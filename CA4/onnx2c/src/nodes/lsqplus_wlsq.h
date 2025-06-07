/* This file is part of onnx2c.
 *
 * LSQPlus WLSQPlus node - Weight quantization
 */
#pragma once
#include "node.h"

namespace toC {

class LSQPlusWLSQ : public Node {
public:
    LSQPlusWLSQ() {
        op_name = "WLSQPlus";
        Qn = 0;         // default values
        Qp = 0;
        per_channel = 0;
    }

    int Qn, Qp, per_channel;

    virtual void parseAttributes(onnx::NodeProto &node) override {
        for (const auto& a : node.attribute()) {
            LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
            if (a.name() == "Qn_i")
                Qn = parse_attribute_int(a);
            else if (a.name() == "Qp_i")
                Qp = parse_attribute_int(a);
            else if (a.name() == "per_channel_i")
                per_channel = parse_attribute_int(a);
            else if (a.name() == "Qn")  // backup for different naming
                Qn = parse_attribute_int(a);
            else if (a.name() == "Qp")  // backup for different naming
                Qp = parse_attribute_int(a);
            else if (a.name() == "per_channel")  // backup for different naming
                per_channel = parse_attribute_int(a);
            else
                LOG(WARNING) << "Ignoring unknown attribute " << a.name() 
                            << " for LSQPlus WLSQPlus node" << std::endl;
        }
        
        LOG(DEBUG) << "WLSQPlus attributes: Qn=" << Qn << ", Qp=" << Qp 
                   << ", per_channel=" << per_channel << std::endl;
    }

    virtual void resolve(void) override {
        // WLSQPlus has 3 tensor inputs + 3 scalar attributes  
        if (get_number_of_inputs() != 3)
            ERROR("WLSQPlus expects exactly 3 inputs: weight, alpha, g");

        name_input(0, "weight");
        name_input(1, "alpha");
        name_input(2, "g");

        const Tensor *weight = get_input_tensor(0);
        
        // First output: quantized weight
        Tensor *w_q = new Tensor;
        w_q->data_dim = weight->data_dim;
        w_q->data_type = weight->data_type;
        register_output(w_q, "w_q");

        // Second output: wq (quantized indices)
        Tensor *wq = new Tensor;
        wq->data_dim = weight->data_dim;
        wq->data_type = weight->data_type;
        register_output(wq, "wq");
    }

    virtual void print(std::ostream &dst) const override {
        const Tensor *weight = get_input_tensor(0);
        std::string type = weight->data_type_str();

        INDT_1 << "/* LSQPlus WLSQPlus quantization */" << std::endl;
        INDT_1 << "/* Qn=" << Qn << ", Qp=" << Qp << ", per_channel=" << per_channel << " */" << std::endl;
        INDT_1 << type << " *w = (" << type << "*)weight;" << std::endl;
        INDT_1 << type << " *w_q_out = (" << type << "*)w_q;" << std::endl;
        INDT_1 << type << " *wq_out = (" << type << "*)wq;" << std::endl;
        INDT_1 << type << " *alpha_ptr = (" << type << "*)alpha;" << std::endl;
        INDT_1 << "int Qn_val = " << Qn << ";" << std::endl;
        INDT_1 << "int Qp_val = " << Qp << ";" << std::endl;
        INDT_1 << "int per_channel_val = " << per_channel << ";" << std::endl;
        
        if (per_channel) {
            INDT_1 << "/* Per-channel quantization */" << std::endl;
            INDT_1 << "for(uint32_t i = 0; i < " << weight->data_num_elem() << "; i++) {" << std::endl;
            INDT_2 << "/* Simplified per-channel implementation */" << std::endl;
            INDT_2 << type << " alpha_val = alpha_ptr[0]; /* TODO: proper per-channel indexing */" << std::endl;
            INDT_2 << type << " clamped = w[i] / alpha_val;" << std::endl;
            INDT_2 << "if(clamped < Qn_val) clamped = Qn_val;" << std::endl;
            INDT_2 << "if(clamped > Qp_val) clamped = Qp_val;" << std::endl;
            INDT_2 << "/* Apply round operation */" << std::endl;
            INDT_2 << type << " sign = (clamped >= 0) ? 1.0f : -1.0f;" << std::endl;
            INDT_2 << type << " rounded = sign * floorf(fabsf(clamped) + 0.5f);" << std::endl;
            INDT_2 << "wq_out[i] = rounded;" << std::endl;
            INDT_2 << "w_q_out[i] = rounded * alpha_val;" << std::endl;
            INDT_1 << "}" << std::endl;
        } else {
            INDT_1 << "/* Global quantization */" << std::endl;
            INDT_1 << type << " alpha_val = alpha_ptr[0];" << std::endl;
            INDT_1 << "for(uint32_t i = 0; i < " << weight->data_num_elem() << "; i++) {" << std::endl;
            INDT_2 << type << " clamped = w[i] / alpha_val;" << std::endl;
            INDT_2 << "if(clamped < Qn_val) clamped = Qn_val;" << std::endl;
            INDT_2 << "if(clamped > Qp_val) clamped = Qp_val;" << std::endl;
            INDT_2 << "/* Apply round operation */" << std::endl;
            INDT_2 << type << " sign = (clamped >= 0) ? 1.0f : -1.0f;" << std::endl;
            INDT_2 << type << " rounded = sign * floorf(fabsf(clamped) + 0.5f);" << std::endl;
            INDT_2 << "wq_out[i] = rounded;" << std::endl;
            INDT_2 << "w_q_out[i] = rounded * alpha_val;" << std::endl;
            INDT_1 << "}" << std::endl;
        }
    }
};

} // namespace toC