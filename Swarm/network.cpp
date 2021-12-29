#include "network.h"


const double e = 2.71;



network::network(	vector<int> layers, vector<double> weights) {
	//count up nodes
	num_nodes				= 0;
	num_layers				= 0;
	unsigned int layer	= 0;
	for (layer = 0; layer < layers.size(); ++layer) {
		num_nodes += layers[layer];
	}
	--layer;
	num_layers	= layer + 1;
	num_outputs	= layers[layer];
	num_inputs	= layers[0] - 1;
	
	//error check	
	if (weights.size() != num_nodes * num_nodes) {
		throw "wrong number of weights";
	}

	//assign everything else
	the_layers	= layers;
	the_weights	= weights;
}


network::~network() {
	the_layers.clear();
	the_weights.clear();
}


vector<double> network::run(vector<float> inputs) {
	//error check
	if (inputs.size() != num_inputs) {
		throw "wrong number of inputs";
	}

	vector<double> node_states;	//the vaule of every node
	node_states.resize(num_nodes);
	
	//turn on bias
	node_states[0] = 1.0;

	//copy in inputs
	int first_node			= 1;		//the first node of this layer
	int last_node			= first_node + the_layers[0] - 2;	//the last node of this layer
	int prev_first_node	= -1;		//the first node of the previous layer
	for (int node = 1; node <= num_inputs; ++node) {
		node_states[node] = inputs[node - 1];
	}
	
	//run each layer
	for (int layer = 1; layer < num_layers; ++layer) {
		prev_first_node	= first_node;
		first_node			= last_node + 1;
		last_node			= first_node + the_layers[layer] - 1;
		for (int node = first_node; node <= last_node; ++node) {
			//get inputs from the previous layer and from the bias
			double input	= 0;
			for (int prev = prev_first_node; prev < first_node; ++prev) {
    			input += node_states[prev] * the_weights[node * num_nodes + prev];
   		}
   		input += node_states[0] * the_weights[node * num_nodes];
   		
			//calculate state of node
 			node_states[node] = sigmoid(input);
		}
	}

	//put outputs into output vector
	vector<double> outputs;
	outputs.resize(num_outputs);
	int first_output = num_nodes - num_outputs;
	for (int output = 0; output < num_outputs; ++output) {
		outputs[output] = node_states[output + first_output];
	}

	return outputs;
}


double network::sigmoid(double input) {
	double output;
	output = (1.0 / (1.0 + pow(e, -1.0 * input)));
	return output;
}


ostream& operator << (ostream& file, const network& out) {
	file << out.num_layers << endl;
	for (int i = 0; i < out.num_layers; ++i) {
		file << out.the_layers[i] << ' ';
	}
	file << endl << endl << out.num_nodes << endl;
	for (int i = 0; i < out.num_nodes; ++i) {
		for (int j = 0; j < out.num_nodes; ++j) {
			file << out.the_weights[j * out.num_nodes + i] << ' ';
		}
		file << endl;
	}
	return file;
}

