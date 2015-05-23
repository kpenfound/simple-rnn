#include <iostream>
#include "neuralnet/neuralnet.h"

int main(void)
{
  NeuralNetwork nn (INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

  std::vector<float> inputs(INPUT_SIZE);
  inputs[0] = 0.2;
  inputs[1] = 1.4;
  inputs[2] = 2.1;
  std::vector<float> targets(OUTPUT_SIZE);
  inputs[0] = 0.8;
  inputs[1] = 0.1;
  inputs[2] = 0.2;
  int iterations = 1;

  nn.set_inputs(inputs);

  for(int i = 0; i < iterations; i++)
  {
    nn.update();
    nn.backpropagate(targets);
  }

  std::vector<float> out = nn.get_outputs();

  for(int i = 0; i < out.size(); i++)
  {
    std::cout << out[i] << std::endl;
  }
}
