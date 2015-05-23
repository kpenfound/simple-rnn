#include "recurrent.h"

RecurrentNeuralNetwork::RecurrentNeuralNetwork()
  : rnn (NUM_LAYERS, std::vector<NeuralNetwork> (LAYER_SIZE, NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE))),
  outputs (OUTPUT_SIZE) {}

void RecurrentNeuralNetwork::update(std::vector<float> in)
{
  for(int i = 0; i < NUM_LAYERS; i++)
  {
    for(int j = 0; j < LAYER_SIZE; j++)
    {
      if(i > 0)
      {
        in = rnn[i-1][j].get_outputs();
      }
      if(j > 0)
      {
        in = vectorMultiplication(in, rnn[i][j-1].get_outputs());
      }
      rnn[i][j].set_inputs(in);
      rnn[i][j].update();
    }
  }
  outputs = rnn[NUM_LAYERS - 1][LAYER_SIZE - 1].get_outputs();
}

std::vector<float> RecurrentNeuralNetwork::get_outputs()
{
  return outputs;
}

void RecurrentNeuralNetwork::train(std::vector<float> target)
{
  rnn[NUM_LAYERS - 1][LAYER_SIZE - 1].backpropagate(target);
}

std::vector<float> vectorMultiplication(std::vector<float> a, std::vector<float> b)
{
  int res_size = std::min(a.size(), b.size());
  std::vector<float> result_vector (res_size);
  for(int i = 0; i < res_size; i++)
  {
    result_vector[i] = a[i] * b[i];
  }
  return result_vector;
}
