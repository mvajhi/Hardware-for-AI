/* This file is part of onnx2c.
 *
 * Unsqueeze:
 * "Insert single-dimensional entries to the shape of an input tensor"
 * I.e. only re-interprets the shape of the data.
 */ 
namespace toC {

class Unsqueeze : public Node {
	public:
	Unsqueeze() {
		op_name = "Unsqueeze";
	}

	std::vector<int64_t> axes_attr;

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		// In ONNX versions before 12, the axes were passed as a node,
		// in 13 this was changed to pass as a input tensor.
		// TODO: if onnx2c handles ONNX versions, maybe re-write this?
		for( const auto& a : node.attribute() ) {
			if( a.name() == "axes" )
				axes_attr = parse_attribute_ints(a);
			else
				ERROR("Bad attribute " << a.name() << " to unsqueeze");
		}
		return;
	}

	/* Body of the node implementing function */
	virtual void print(std::ostream &dst) const override
	{
		const Tensor *data = get_input_tensor(0);
		std::string type = data->data_type_str();

		dst << "\t/* Unsqueeze */" << std::endl;

		dst << "\t" << type << " *data = (" << type << "*)input;" << std::endl;
		dst << "\t" << type << " *expanded= (" << type << "*)output;" << std::endl;

		// TODO: can't this be a no-op? Check if the compiler can optimize this away?
		//       also if not, can it optimize a memcpy()?
		dst << "\t" << "for( uint32_t i=0; i<" << data->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\t" << "expanded[i] = data[i];" << std::endl;
		dst << std::endl;
	}


	/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
	virtual void resolve(void) override
	{
		const Tensor *data = get_input_tensor(0);
		name_input(0, "input");

		// ONNX13 changed how axes were passed (but not the contents).
		// if axes_attr is set, then the padding axes are passed as attribute
		// otherwise as input tensor. Handle both cases here, so we don't need
		// to know later on how the data was passed in.
		// After this, the axes_attr contains the (raw) axes data in either case.
		// TODO: since axes is now an input tensor - can the contents be dynamic??
		if (axes_attr.size() == 0 ) {
			if( get_number_of_inputs() != 2 )
				ERROR("axes not provided. Malformatted ONNX?");
			const Tensor *axes_tensor = get_input_tensor(1);
			name_input(1, "axes_tensor");
			if (axes_tensor->isConst == false)
				ERROR("provided axes are dynamic, not implmeneted");
			for( unsigned i=0; (int)i<axes_tensor->data_num_elem(); i++) {
				int64_t *rd = (int64_t*)axes_tensor->data_buffer;  // axes data must be int64
				axes_attr.push_back(rd[i]);
			}
		}

		Tensor *t = new Tensor;
		t->data_type = data->data_type;

		// insert the new axis:
		// The input dimensions are copied in-order to output, but then
		// dimensions of 1 are inserted as per the axes data.
		// The axes data are indices into the output data dimensions.
		unsigned expanded_rank = data->rank() + axes_attr.size();
		t->data_dim.resize(expanded_rank); 

		// negative axes data means "count index from end".
		// Convert them here to positive indices.
		std::vector<unsigned> cleaned_axes;
		for ( auto a : axes_attr ) {
			if ( a < 0)
				cleaned_axes.push_back( expanded_rank + a );
			else
				cleaned_axes.push_back( a );
		}

		// "The order of values in axes does not matter and can come in any order."
		std::sort(cleaned_axes.begin(), cleaned_axes.end());

		// Create the expanded dimensions.
		unsigned di, ai;
		di=ai=0;
		for( unsigned i=0; i<expanded_rank; i++ ) {
			if( cleaned_axes[ai] == i ) {
				t->data_dim[i] = 1;
				ai++;
			}
			else {
				t->data_dim[i] = data->data_dim[di];
				di++;
			}
		}

		register_output(t, "output");
	}
};
}

