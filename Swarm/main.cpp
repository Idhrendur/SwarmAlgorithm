#include "network.h"
#include "swarm.h"
#include <fstream>
#include <vector>
using namespace std;



int main()
{
	cout << "How many layers are in the network? ";
	int num_layers;
	cin >> num_layers;
	vector<int> layers;
	layers.resize(num_layers);
	for (int i = 0; i < num_layers; ++i)
	{
		cout << "How many nodes in layer " << i << "? ";
		cin >> layers[i];
	}

	cout << "What is the filename of the training set? ";
	char in_filename[256];
	cin >> in_filename;
	ifstream train;
	train.open(in_filename);

	network_swarm swarm(layers, train);
	train.close();

	cout << "What should the file with the trained network be named? ";
	char out_filename[256];
	cin >> out_filename;
	ofstream trained;
	trained.open(out_filename);

	swarm.train(trained);
	trained.close();

	return 0;
}
