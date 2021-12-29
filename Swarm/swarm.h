#ifndef SWARM_H_
#define SWARM_H_


#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "network.h"
using namespace std;



typedef struct {
	vector<double>	position;
	vector<double>	velocity;
	vector<double>	best_position;
	double			least_error;
} particle;


struct train_pattern {
	vector<float> inputs;
	vector<float> outputs;
};


class network_swarm {
	public:
		network_swarm(vector<int> layers, ifstream& train_file);
		~network_swarm();
		void train(ofstream& output_file);
	private:
		void initialize_particles();
		void update();
		void test_particles();
		network	make_net(particle source);
		
		int							num_nodes;
		int 							num_weights;
		int							num_inputs;
		int							num_outputs;
		int							num_train_sets;
		vector<int>					the_layers;
		vector<train_pattern>	training_data;
		vector<particle>			particles;
		vector<double>				global_best_position;
		double						global_least_error;
		
		ofstream						debug_file;
};

//	To save processing, the swarm will not keep track of redundant network
//		data.  Don't care values will be added by the swarm when a network is
//		initialized.



#endif	// SWARM_H_
