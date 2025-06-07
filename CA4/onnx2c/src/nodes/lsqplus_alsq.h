/* This file is part of onnx2c.
 *
 * LSQPlus ALSQPlus node - Activation quantization
 */
#pragma once
#include "node.h"

namespace toC {

class LSQPlusALSQ : public Node {
public:
    LSQPlusALSQ() {
        op_name = "ALSQPlus";
        Qn = 0;  // default values
        Qp = 0;
    }

    int Qn, Qp;

    virtual void parseAttributes(onnx::NodeProto &node) override {
        for (const auto& a : node.attribute()) {
            LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
            if (a.name() == "Qn_i")
                Qn = parse_attribute_int(a);
            else if (a.name() == "Qp_i") 
                Qp = parse_attribute_int(a);
            else if (a.name() == "Qn")  // backup for different naming
                Qn = parse_attribute_int(a);
            else if (a.name() == "Qp")  // backup for different naming
                Qp = parse_attribute_int(a);
            else
                LOG(WARNING) << "Ignoring unknown attribute " << a.name() 
                            << " for LSQPlus ALSQPlus node" << std::endl;
        }
        
        LOG(DEBUG) << "ALSQPlus attributes: Qn=" << Qn << ", Qp=" << Qp << std::endl;
    }

    virtual void resolve(void) override {
        // ALSQPlus has 4 tensor inputs + 2 scalar attributes
        if (get_number_of_inputs() != 4)
            ERROR("ALSQPlus expects exactly 4 inputs: weight, alpha, beta, g");

        name_input(0, "weight");
        name_input(1, "alpha");
        name_input(2, "beta");
        name_input(3, "g");

        const Tensor *weight = get_input_tensor(0);
        
        // Output has same shape and type as weight
        Tensor *output = new Tensor;
        output->data_dim = weight->data_dim;
        output->data_type = weight->data_type;
        register_output(output, "output");
    }

    virtual void print(std::ostream &dst) const override {
        const Tensor *weight = get_input_tensor(0);
        std::string type = weight->data_type_str();

        INDT_1 << "/* LSQPlus ALSQPlus quantization */" << std::endl;
        INDT_1 << "/* Qn=" << Qn << ", Qp=" << Qp << " */" << std::endl;
        INDT_1 << type << " *w = (" << type << "*)weight;" << std::endl;
        INDT_1 << type << " *out = (" << type << "*)output;" << std::endl;
        INDT_1 << type << " alpha_val = *(" << type << "*)alpha;" << std::endl;
        INDT_1 << type << " beta_val = *(" << type << "*)beta;" << std::endl;
        INDT_1 << "int Qn_val = " << Qn << ";" << std::endl;
        INDT_1 << "int Qp_val = " << Qp << ";" << std::endl;
        
        INDT_1 << "for(uint32_t i = 0; i < " << weight->data_num_elem() << "; i++) {" << std::endl;
        INDT_2 << type << " clamped = (w[i] - beta_val) / alpha_val;" << std::endl;
        INDT_2 << "if(clamped < Qn_val) clamped = Qn_val;" << std::endl;
        INDT_2 << "if(clamped > Qp_val) clamped = Qp_val;" << std::endl;
        INDT_2 << "/* Apply round operation */" << std::endl;
        INDT_2 << type << " sign = (clamped >= 0) ? 1.0f : -1.0f;" << std::endl;
        INDT_2 << type << " rounded = sign * floorf(fabsf(clamped) + 0.5f);" << std::endl;
        INDT_2 << "out[i] = rounded * alpha_val + beta_val;" << std::endl;
        INDT_1 << "}" << std::endl;
    }
};

} // namespace toC