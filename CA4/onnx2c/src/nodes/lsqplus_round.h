/* This file is part of onnx2c.
 *
 * LSQPlus Round node - Custom quantization round operation
 */
#pragma once
#include "node.h"

namespace toC {

class LSQPlusRound : public Node {
public:
    LSQPlusRound() {
        op_name = "Round";  // Note: this matches your custom domain
    }

    virtual void parseAttributes(onnx::NodeProto &node) override {
        // No attributes for Round operation
        for (const auto& a : node.attribute()) {
            LOG(WARNING) << "Ignoring unknown attribute " << a.name() 
                        << " for LSQPlus Round node" << std::endl;
        }
    }

    virtual void resolve(void) override {
        if (get_number_of_inputs() != 1)
            ERROR("LSQPlus Round expects exactly 1 input");

        const Tensor *input = get_input_tensor(0);
        name_input(0, "input");

        // Output has same shape and type as input
        Tensor *output = new Tensor;
        output->data_dim = input->data_dim;
        output->data_type = input->data_type;
        register_output(output, "output");
    }

    virtual void print(std::ostream &dst) const override {
        const Tensor *input = get_input_tensor(0);
        std::string type = input->data_type_str();

        INDT_1 << "/* LSQPlus Round operation */" << std::endl;
        INDT_1 << type << " *in = (" << type << "*)input;" << std::endl;
        INDT_1 << type << " *out = (" << type << "*)output;" << std::endl;
        
        INDT_1 << "for(uint32_t i = 0; i < " << input->data_num_elem() << "; i++) {" << std::endl;
        INDT_2 << type << " val = in[i];" << std::endl;
        INDT_2 << type << " sign = (val >= 0) ? 1.0f : -1.0f;" << std::endl;
        INDT_2 << "out[i] = sign * floorf(fabsf(val) + 0.5f);" << std::endl;
        INDT_1 << "}" << std::endl;
    }
};

} // namespace toC