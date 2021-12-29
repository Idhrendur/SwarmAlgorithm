#ifndef NETWORK_H_
#define NETWORK_H_


#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;



class network {
	public:
		network(vector<int> layers, vector<double> weights);
		~network();
		vector<double> run(vector<float> inputs);
		friend ostream& operator << (ostream& file, const network& out);
	private:
		double sigmoid(double input);
		int				num_layers;
		int				num_nodes;
		int				num_inputs;
		int				num_outputs;
		vector<double>	the_weights;
		vector<int>		the_layers;
};

//	things to note:
//		node zero is the bias node, this is reflected in the weights, and the number
//			of nodes in the first layer.
//		each layer can only be affected by the previous layer and the bias node
//		the state of an input node is simply the input value.  The sigmoid
//			function is not applied.



#endif	// NETWORK_H_
