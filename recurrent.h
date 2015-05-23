#ifndef __RECURRENT_H__
#define __RECURRENT_H__

#include <vector>
#include <algorithm>

#include "neuralnet/neuralnet.h"

// Recurrent net
const int NUM_LAYERS = 3;
const int LAYER_SIZE = 3;

// Neural net
const int INPUT_SIZE = 3;
const int HIDDEN_SIZE = 6;
const int OUTPUT_SIZE = 3;

class RecurrentNeuralNetwork
{
  std::vector< std::vector<NeuralNetwork> > rnn;
  std::vector<float> outputs;
public:
  RecurrentNeuralNetwork();
  void update(std::vector<float>);
  std::vector<float> get_outputs();
  void train(std::vector<float>);
};

std::vector<float> vectorMultiplication(std::vector<float>, std::vector<float>);

#endif // __RECURRENT_H__