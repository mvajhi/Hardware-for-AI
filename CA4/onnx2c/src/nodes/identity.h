/* This file is part of onnx2c.
 *
 * Identity node.
 * Simply passes input through to output without any change.
 */
#pragma once
#include "node.h"

namespace toC {

class Identity : public Node {
public:
    Identity() {
        op_name = "Identity";
    }

    virtual void parseAttributes(onnx::NodeProto &node) override {
        // Identity has no attributes
        for (const auto& a : node.attribute()) {
            LOG(WARNING) << "Ignoring unknown attribute " << a.name() 
                        << " for Identity node" << std::endl;
        }
    }

    virtual void resolve(void) override {
        if (get_number_of_inputs() != 1)
            ERROR("Identity expects exactly 1 input");

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

        INDT_1 << "/* Identity - pass through */" << std::endl;
        INDT_1 << type << " *in = (" << type << "*)input;" << std::endl;
        INDT_1 << type << " *out = (" << type << "*)output;" << std::endl;
        
        INDT_1 << "for(uint32_t i = 0; i < " << input->data_num_elem() << "; i++) {" << std::endl;
        INDT_2 << "out[i] = in[i];" << std::endl;
        INDT_1 << "}" << std::endl;
    }
};

} // namespace toC