#include "recurrent.h"
#include <iostream>

int main(void)
{
  srand(43); // Seed random

  int iterations = 1000; // Number of generations

  std::vector<float> inputs(INPUT_SIZE); // Network input
  inputs[0] = 0.1;
  inputs[1] = 0.1;
  inputs[2] = 0.9;

  std::vector<float> targets(OUTPUT_SIZE); // Expected network output
  targets[0] = 1;
  targets[1] = 0;
  targets[2] = 1;

  RecurrentNeuralNetwork rnn = RecurrentNeuralNetwork();

  for(int i = 0; i < iterations; i++)
  {
    rnn.update(inputs);
    rnn.train(targets);
  }

  std::vector<float> outputs = rnn.get_outputs();
  for(int i = 0; i < outputs.size(); i++)
  {
    std::cout << outputs[i] << std::endl;
  }
}
