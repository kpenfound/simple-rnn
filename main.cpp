#include "recurrent.h"
#include <iostream>

int main(void)
{
  srand(43); // Seed random

  int iterations = 10; // Number of generations

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
  const float five[] = {1, 0, 1};
  std::vector<float> v_five (five, five + sizeof(five) / sizeof(float));
  const float six[] = {0, 1, 1};
  std::vector<float> v_six (six, six + sizeof(six) / sizeof(float));
  const float seven[] = {1, 1, 1};
  std::vector<float> v_seven (seven, seven + sizeof(seven) / sizeof(float));

  std::vector< std::vector<float> > inputs(LAYER_SIZE); // Network input
  inputs[0] = v_one;
  inputs[1] = v_two;
  inputs[2] = v_three;

  std::vector< std::vector<float> > targets(LAYER_SIZE); // Expected network output
  targets[0] = v_two;
  targets[1] = v_three;
  targets[2] = v_four;

  RecurrentNeuralNetwork rnn = RecurrentNeuralNetwork();

  for(int i = 0; i < iterations; i++)
  {
    rnn.update(&inputs);
    rnn.train(targets);
  }

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
