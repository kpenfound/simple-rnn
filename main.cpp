#include "recurrent.h"
#include <iostream>

/*
  Main executes a simple example use of our RNN.
  Using 0,1,2,3,4 as 3 bit binary numbers, we teach our
  network to recognize the sequence.  In the example seen
  here, we will train a single test case of 1,2,3->2,3,4
*/
int main(void)
{
  srand(43); // Seed random

  int iterations = 100; // Number of generations

  const float zero[] = {0, 0, 0};
  std::vector<float> v_zero (zero, zero + sizeof(zero) / sizeof(float));
  const float one[] = {1, 0, 0};
  std::vector<float> v_one (one, one + sizeof(one) / sizeof(float));
  const float two[] = {0, 1, 0};
  std::vector<float> v_two (two, two + sizeof(two) / sizeof(float));
  const float three[] = {1, 1, 0};
  std::vector<float> v_three (three, three + sizeof(three) / sizeof(float));
  const float four[] = {0, 0, 1};
  std::vector<float> v_four (four, four + sizeof(four) / sizeof(float));

  std::vector< std::vector<float> > inputs(LAYER_SIZE); // Network input
  inputs[0] = v_one;
  inputs[1] = v_two;
  inputs[2] = v_three;

  std::vector< std::vector<float> > targets(LAYER_SIZE); // Expected network output
  targets[0] = v_two;
  targets[1] = v_three;
  targets[2] = v_four;

  RecurrentNeuralNetwork rnn = RecurrentNeuralNetwork();

  // Execution loop: Execute network and then train with backpropagation
  for(int i = 0; i < iterations; i++)
  {
    rnn.update(&inputs);
    rnn.train(targets);
  }

  // Display outputs of the network after iterations have completed
  std::vector< std::vector<float> > outputs = rnn.get_outputs();
  for(int i = 0; i < outputs.size(); i++)
  {
    for(int j = 0; j < outputs[i].size(); j++)
    {
      std::cout << outputs[i][j] << " ";
    }
    std::cout << std::endl;
  }
}
