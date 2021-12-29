#include "swarm.h"


const int max_particles		= 20;
const int max_iterations	= 1000;
const float dont_care		= 0.0;
const float v_max				= 2.0;



network_swarm::network_swarm(vector<int> layers, ifstream& train_file) {
	//count nodes & connections
	if (layers.size() < 2) {
		throw "network must have at least two layers!";
	}
	num_weights	= 0;
	num_nodes	= layers[0];
	for (unsigned int layer = 1; layer < layers.size(); ++layer) {
		num_weights	+=	layers[layer] * layers[layer - 1];
		num_nodes	+= layers[layer];
	}
	num_weights	+= num_nodes - layers[1] - layers[0];	//add in bias connections
		
	//count inputs & outputs
	num_inputs	= layers[0] - 1;
	num_outputs	= layers[layers.size() - 1];
	
	//load training data
	if(train_file.eof()) {
		throw "training file ended early";
	}
 	train_file >> num_train_sets;
 	for (int set = 0; set < num_train_sets; ++set) {
   	train_pattern new_pattern;
		new_pattern.inputs.resize(num_inputs);
		new_pattern.outputs.resize(num_outputs);
		for(int in_num = 0; in_num < num_inputs; ++in_num) {
			if(train_file.eof()) {
				throw "training file ended early";
			}
			train_file >> new_pattern.inputs[in_num];
		}
		for(int out_num = 0; out_num < num_outputs; ++out_num) {
			if(train_file.eof()) {
				throw "training file ended early";
			}
			train_file >> new_pattern.outputs[out_num];
		}
		training_data.push_back(new_pattern);
	}
	
	//default initialize other variables
	the_layers	= layers;
	particles.clear();
	global_best_position.clear();
	global_least_error	= 1000.0 * 1000.0 * 1000.0;
	srand((unsigned)time(0));
}


network_swarm::~network_swarm() {
	training_data.clear();
	particles.clear();
	global_best_position.clear();
}


void network_swarm::train(ofstream& output_file) {
	initialize_particles();
	int i	= 0;
	for (i = 0; i < max_iterations; ++i) {
		update();
		if (global_least_error < 0.1) {
			break;
		}
	}
	test_particles();
	particle best_particle;
	best_particle.position = global_best_position;
	network new_net = make_net(best_particle);
	output_file << new_net;
	cout << "Network trained in " << i << " iterations." << endl;
	cout << "Final error on training set: " << global_least_error << endl;
}


void network_swarm::initialize_particles() {
	particles.clear();
	global_best_position.clear();
	global_least_error	= 1000.0 * 1000.0 * 1000.0;
	for(int i = 0; i < max_particles; ++i) {
		particle new_particle;
		new_particle.position.resize(num_weights);
		new_particle.velocity.resize(num_weights);
		new_particle.best_position.resize(num_weights);
		for (int dim = 0; dim < num_weights; ++dim) {
			new_particle.position[dim] 		= -100 + (200 * (rand() / (RAND_MAX + 1.0)));
			new_particle.velocity[dim] 		= -1 + (2 * (rand() / (RAND_MAX + 1.0)));
		}
		new_particle.best_position	= new_particle.position;
		new_particle.least_error	= 1000.0 * 1000.0 * 1000.0;
		particles.push_back(new_particle);
	}
}


void network_swarm::update() {
	test_particles();
	for (int particle = 0; particle < max_particles; ++particle) {
		//update velocity
		double rho_1	= (4.0 * (rand() / (RAND_MAX + 1.0)));
		double rho_2	= 4.0 - rho_1;
		for (int dim = 0; dim < num_weights; ++dim) {
			particles[particle].velocity[dim] +=
				(rho_1 * (particles[particle].best_position[dim] -
    						particles[particle].position[dim])) +
				(rho_2 * (global_best_position[dim] -
    						particles[particle].position[dim]));
       	if (particles[particle].velocity[dim] > v_max) {
       		particles[particle].velocity[dim] = v_max;
      	}
		}
		
		//update position
  		for (int dim = 0; dim < num_weights; ++dim) {
  			particles[particle].position[dim] += particles[particle].velocity[dim];
  		}
	}
}



void network_swarm::test_particles() {
	for (int particle = 0; particle < max_particles; ++particle) {
		//construct neural net
		network new_net = make_net(particles[particle]);
		
		//find total error of net
		double error = 0.0;
		for (int set = 0; set < num_train_sets; ++set) {
   		vector<double> results = new_net.run(training_data[set].inputs);
   		for(int output = 0; output < num_outputs; ++output) {
				double difference;
    			difference	=	1.0 * results[output] -  
									1.0 * training_data[set].outputs[output];
    			error += difference * difference;
   		}
		}
		
		//if error is less than previous best, set new best position
		if (error < particles[particle].least_error) {
			particles[particle].least_error		= error;
			particles[particle].best_position	= particles[particle].position;
		}
		
		//if error is less than the global best position, set new global best
		if (error < global_least_error) {
			global_least_error	= error;
			global_best_position	= particles[particle].position;
   		for(int dim = 0; dim < num_weights; ++dim) {
     			debug_file << global_best_position[dim] << ' ';
			}
		}
	}
}


network network_swarm::make_net(particle source) {
		//set up weights
		vector<double> weights;
		weights.resize(num_nodes * num_nodes);
		for(unsigned int i = 0; i < weights.size(); ++i) {
			weights[i] = dont_care;
		}
		
		//set weights from bias node
		int particle_index	= 0;
		for(int i = the_layers[0]; i < num_nodes; ++i) {
			weights[i * num_nodes] = source.position[particle_index];
			particle_index++;
		}
		
		//set the rest of the weights
		int node_index			= 1;
		int layer_end			= 0;
		int next_layer_end 	= 0;
		int num_layers			= int(the_layers.size());
		for (int i = 0; i < num_layers - 1; ++i) {
			layer_end  		= layer_end + the_layers[i];
			next_layer_end	= layer_end + the_layers[i + 1];
			for (int j = node_index; j < layer_end; ++j) {
				for (int k = layer_end; k < next_layer_end; ++k) {
					weights[k * num_nodes + j] = source.position[particle_index];
					++particle_index;
				}
				++node_index;
			}
		}
		
		//create and return network
		network new_net(the_layers, weights);
		return new_net;
}
