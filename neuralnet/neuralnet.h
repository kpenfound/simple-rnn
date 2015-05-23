#ifndef __NEURALNET_H__
#define __NEURALNET_H__

#include <vector>
#include <cmath>
#include <cstdlib>

class Neuron
{
  std::vector<float> weights;
  float output;
public:
  Neuron(int);
  void set_weights(std::vector<float>);
  float get_output();
  float get_weight(int);
  void update(std::vector<float>);
};

class NeuronLayer
{
  int layer_size;
  std::vector<Neuron> neurons;
public:
  NeuronLayer(int,int);
  int get_size();
  Neuron get_neuron(int);
  void set_neuron(int, Neuron);
  void update(std::vector<float>);
  std::vector<float> get_outputs();
};

class NeuralNetwork
{
  std::vector<float> inputs;
  NeuronLayer hidden_layer;
  NeuronLayer output_layer;
public:
  NeuralNetwork(int,int,int);
  void set_inputs(std::vector<float>);
  std::vector<float> get_outputs();
  void update();
  static float nonlinearFunction(float);
  static float nonlinearDerivative(float);
  void backpropagate(std::vector<float>);
};

float randomWeight();

#endif
