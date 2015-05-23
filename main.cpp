#include "neuralnet/neuralnet.h"

int main(void)
{
  NeuralNetwork nn (INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

  std::vector<float> inputs(INPUT_SIZE);
  inputs[0] = 0.1;
  inputs[1] = 0.1;
  inputs[2] = 0.9;
  std::vector<float> targets(OUTPUT_SIZE);
  targets[0] = 1;
  targets[1] = 0;
  targets[2] = 1;
  int iterations = 100;

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
