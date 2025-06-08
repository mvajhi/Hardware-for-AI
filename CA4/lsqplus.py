import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

CUSTOM_OP_DOMAIN = "lsqplus_ops"
CUSTOM_OPSET_VERSION = 1

class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

    @staticmethod
    def symbolic(g, input):
        return g.op(f"{CUSTOM_OP_DOMAIN}::Round", input).setType(input.type())

class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, beta, g, Qn, Qp):
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp
        clamped_val = torch.div((weight - beta), alpha).clamp(Qn, Qp)
        w_q = Round.apply(clamped_val)
        w_q = w_q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger
        grad_alpha = ((smaller * Qn + bigger * Qp +
            between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, grad_beta, None, None, None

    @staticmethod
    def symbolic(g, weight, alpha, beta, g_tensor, Qn, Qp):
        # just 4 tensor input و 2 attribute
        return g.op(f"{CUSTOM_OP_DOMAIN}::ALSQPlus",
                    weight, alpha, beta, g_tensor,
                    Qn_i=int(Qn),
                    Qp_i=int(Qp)).setType(weight.type())

class WLSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        w_q, wq = None, None
        if per_channel:
            sizes = weight.size()
            weight_reshaped = weight.view(weight.size()[0], -1)
            weight_reshaped = torch.transpose(weight_reshaped, 0, 1)
            alpha_broadcasted = torch.broadcast_to(alpha, weight_reshaped.size())
            clamped_val = torch.div(weight_reshaped, alpha_broadcasted).clamp(Qn, Qp)
            wq_reshaped = Round.apply(clamped_val)
            w_q_reshaped = wq_reshaped * alpha_broadcasted
            w_q = torch.transpose(w_q_reshaped, 0, 1).view(sizes)
            wq = torch.transpose(wq_reshaped, 0, 1).view(sizes)
        else:
            clamped_val = torch.div(weight, alpha).clamp(Qn, Qp)
            wq = Round.apply(clamped_val)
            w_q = wq * alpha
        return w_q, wq.detach()

    @staticmethod
    def backward(ctx, grad_weight, gwq):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight_reshaped = weight.view(weight.size()[0], -1)
            weight_reshaped = torch.transpose(weight_reshaped, 0, 1)
            alpha_broadcasted = torch.broadcast_to(alpha, weight_reshaped.size())
            q_w_reshaped = weight_reshaped / alpha_broadcasted
            q_w = torch.transpose(q_w_reshaped, 0, 1).view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller -bigger
        if per_channel:
            grad_alpha = ((smaller * Qn + bigger * Qp +
                between * Round.apply(q_w) - between * q_w)*grad_weight * g)
            grad_alpha = grad_alpha.view(grad_alpha.size()[0], -1).sum(dim=1)
        else:
            grad_alpha = ((smaller * Qn + bigger * Qp +
                between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, None

    @staticmethod
    def symbolic(g, weight, alpha, g_tensor, Qn, Qp, per_channel):
        # just 3 tensor input و 3 attribute
        outputs = g.op(f"{CUSTOM_OP_DOMAIN}::WLSQPlus",
                    weight, alpha, g_tensor,
                    Qn_i=int(Qn),
                    Qp_i=int(Qp),
                    per_channel_i=int(per_channel),
                    outputs=2)
        
        outputs[0].setType(weight.type())
        outputs[1].setType(weight.type())
        return outputs


class LSQPlusActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False,batch_init = 20):
        super(LSQPlusActivationQuantizer, self).__init__()
        self.a_bits = nn.Parameter(torch.tensor(a_bits), requires_grad=False)
        self.s_bits = 20
        self.n = 1 << 16
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            self.Qn = 0
            self.Qp = 2 ** a_bits - 1
        else:
            self.Qn = - 2 ** (a_bits - 1)
            self.Qp = 2 ** (a_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1).squeeze(0), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.ones(1).squeeze(0), requires_grad=True)
        self.g = torch.nn.Parameter(torch.ones(1).squeeze(0), requires_grad=True)
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, activation):
        if self.training:
            if self.init_state==0:
                self.g.data = torch.tensor(1.0/math.sqrt(activation.numel() * self.Qp))
                mina = torch.min(activation.detach())
                self.s.data = (torch.max(activation.detach()) - mina)/(self.Qp-self.Qn)
                self.beta.data = mina - self.s.data *self.Qn
                self.init_state += 1
            elif self.init_state<self.batch_init:
                mina = torch.min(activation.detach())
                self.s.data = self.s.data*0.9 + 0.1*(torch.max(activation.detach()) - mina)/(self.Qp-self.Qn)
                self.beta.data = self.beta.data*0.9 + 0.1* (mina - self.s.data * self.Qn)
                self.init_state += 1
            elif self.init_state==self.batch_init:
                self.init_state += 1

        if self.a_bits.item() == 32:
            q_a = activation
        elif self.a_bits.item() == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits.item() != 1
        else:
            q_a = ALSQPlus.apply(activation, self.s, self.beta, self.g, self.Qn, self.Qp)
        return q_a

class LSQPlusWeightQuantizer(nn.Module):
    def __init__(self, w_bits, all_positive=False, per_channel=False, batch_init = 20, shape=(1,), saved=True):
        super(LSQPlusWeightQuantizer, self).__init__()
        self.w_bits = nn.Parameter(torch.tensor(w_bits), requires_grad=False)
        self.s_bits = 20
        self.n = 1 << 16
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            self.Qn = 0
            self.Qp = 2 ** w_bits - 1
        else:
            self.Qn = - 2 ** (w_bits - 1)
            self.Qp = 2 ** (w_bits - 1) - 1
        self.per_channel = per_channel
        self.register_buffer('init_state', torch.zeros(1))
        if not per_channel:
           self.s = torch.nn.Parameter(torch.ones(1).squeeze(0), requires_grad=True)
        else:
            self.s = torch.nn.Parameter(torch.ones(shape[0]), requires_grad=True)

        self.saved = saved
        if saved:
            self.wq = torch.nn.Parameter(torch.ones(shape), requires_grad=True)
        self.g = torch.nn.Parameter(torch.ones(1).squeeze(0), requires_grad=True)
        
    def forward(self, weight):
        if self.training:
            if self.init_state==0:
                self.div = 2**self.w_bits.item()-1
                self.g.data = torch.tensor(1.0/math.sqrt(weight.numel() * self.Qp))
                if self.per_channel:
                    weight_tmp = weight.detach().view(weight.size()[0], -1)
                    mean = torch.mean(weight_tmp, dim=1)
                    std = torch.std(weight_tmp, dim=1)
                    v, _ = torch.max(torch.stack([torch.abs(mean-3*std), torch.abs(mean + 3*std)]), dim=0)
                    self.s.data = v/self.div
                else:
                    mean = torch.mean(weight.detach())
                    std = torch.std(weight.detach())
                    self.s.data = max(torch.abs(mean-3*std), torch.abs(mean + 3*std))/self.div
                self.init_state += 1
            elif self.init_state<self.batch_init:
                self.div = 2**self.w_bits.item()-1
                if self.per_channel:
                    weight_tmp = weight.detach().view(weight.size()[0], -1)
                    mean = torch.mean(weight_tmp, dim=1)
                    std = torch.std(weight_tmp, dim=1)
                    v, _ = torch.max(torch.stack([torch.abs(mean-3*std), torch.abs(mean + 3*std)]), dim=0)
                    self.s.data = self.s.data*0.9 + 0.1*v/self.div
                else:
                    mean = torch.mean(weight.detach())
                    std = torch.std(weight.detach())
                    self.s.data = self.s.data*0.9 + 0.1*max(torch.abs(mean-3*std), torch.abs(mean + 3*std))/self.div
                self.init_state += 1
            elif self.init_state==self.batch_init:
                self.init_state += 1

        if self.w_bits.item() == 32:
            w_q = weight
        elif self.w_bits.item() == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits.item() != 1
        else:
            if self.saved:
                w_q, self.wq.data = WLSQPlus.apply(weight, self.s, self.g, self.Qn, self.Qp, self.per_channel)
            else:
                w_q, _ = WLSQPlus.apply(weight, self.s, self.g, self.Qn, self.Qp, self.per_channel)

        return w_q

class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 w_bits=8, a_bits=8, all_positive=False, per_channel=False, batch_init=20):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight_quantizer = LSQPlusWeightQuantizer(w_bits, all_positive, per_channel, batch_init, self.weight.shape)
        self.act_quantizer = LSQPlusActivationQuantizer(a_bits, all_positive, batch_init)
    
    def forward(self, x):
        x_q = self.act_quantizer(x)
        w_q = self.weight_quantizer(self.weight)
        return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, 
                 w_bits=8, a_bits=8, all_positive=False, per_channel=False, batch_init=20):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.weight_quantizer = LSQPlusWeightQuantizer(w_bits, all_positive, per_channel, batch_init, self.weight.shape)
        self.act_quantizer = LSQPlusActivationQuantizer(a_bits, all_positive, batch_init)
    
    def forward(self, x):
        x_q = self.act_quantizer(x)
        w_q = self.weight_quantizer(self.weight)
        return F.linear(x_q, w_q, self.bias)

def prepare(model, w_bits=8, a_bits=8, all_positive=False, per_channel=False, batch_init=20):
    """
    Replaces Conv2d and Linear layers in the model with QuantConv2d and QuantLinear layers.
    
    Args:
        model: The model to be quantized
        w_bits: Number of bits for weight quantization
        a_bits: Number of bits for activation quantization
        all_positive: Whether the quantization is all positive
        per_channel: Whether to use per-channel quantization for weights
        batch_init: Number of batches for quantizer initialization
        
    Returns:
        Quantized model with replaced layers
    """
    quant_model = copy.deepcopy(model)
    
    for name, module in list(quant_model.named_children()):
        # If it's a container (like Sequential), recursively apply to its children
        if len(list(module.children())) > 0:
            setattr(quant_model, name, prepare(module, w_bits, a_bits, all_positive, per_channel, batch_init))
        
        # If it's a Conv2d layer, replace it with QuantConv2d
        elif isinstance(module, nn.Conv2d):
            quant_conv = QuantConv2d(
                module.in_channels, 
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                w_bits,
                a_bits,
                all_positive,
                per_channel,
                batch_init
            )
            
            # Copy weights and bias if exists
            quant_conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                quant_conv.bias.data = module.bias.data.clone()
                
            setattr(quant_model, name, quant_conv)
            
        # If it's a Linear layer, replace it with QuantLinear
        elif isinstance(module, nn.Linear):
            quant_linear = QuantLinear(
                module.in_features,
                module.out_features,
                module.bias is not None,
                w_bits,
                a_bits,
                all_positive,
                per_channel,
                batch_init
            )
            
            # Copy weights and bias if exists
            quant_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                quant_linear.bias.data = module.bias.data.clone()
                
            setattr(quant_model, name, quant_linear)
    
    return quant_model
