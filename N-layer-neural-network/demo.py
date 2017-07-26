from numpy import *


class NLayerNeuralNetwork:
    def __init__(self):
        self.layer = 1
        self.synaptic_weights = {}
        self.adjustment = {}
        self.neurons_in_layers = {}
        self.hidden_layer_value = {}

    # activation function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def neurons_in_input_layer(self, inputs):
        self.neurons_in_layers[self.layer] = inputs

    def output_neuron(self, number):
        self.synaptic_weights[self.layer] = 2 * random.random(
            (number, self.neurons_in_layers[self.layer])) - 1
        # print(self.neurons_in_layers[self.layer])
        self.adjustment[self.layer + 1] = zeros((number, self.neurons_in_layers[self.layer]))
        self.neurons_in_layers[self.layer + 1] = 1

    def add_layer(self, no_of_sigmoid_neurons):
        self.layer += 1
        # adding random weight to new layer
        self.synaptic_weights[self.layer - 1] = 2 * random.random(
            (no_of_sigmoid_neurons, self.neurons_in_layers[self.layer - 1])) - 1
        # setting initial adjustment to zero
        self.adjustment[self.layer] = zeros((no_of_sigmoid_neurons, 1))
        self.neurons_in_layers[self.layer] = no_of_sigmoid_neurons

    def predict(self, inputs):
        value = None
        for layer in self.neurons_in_layers:
            if layer != len(self.neurons_in_layers):
                for weights in self.synaptic_weights[layer]:
                    # print(weights)
                    # print(inputs)
                    value = self.__sigmoid(dot(inputs, weights.T))
                    if self.hidden_layer_value.get(layer) is None:
                        self.hidden_layer_value[layer] = value.T
                    else:
                        self.hidden_layer_value[layer] = stack((self.hidden_layer_value[layer], value.T))
                inputs = self.hidden_layer_value[layer].T
        return self.hidden_layer_value

    def train(self, training_set_inputs, training_set_outputs, learning_rate=1, stop_accuracy=1e-5,
              number_of_iterations=1):
        for iteration in range(number_of_iterations):
            output = self.predict(training_set_inputs)
            # print(type(training_set_input))
            # print(training_set_outputs-output)

            error = training_set_outputs - array([output[3]]).T
            print(error)
            # adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            #
            # self.synaptic_weights[1] += adjustment.T


if __name__ == "__main__":
    neural_network = NLayerNeuralNetwork()
    neural_network.neurons_in_input_layer(3)
    neural_network.add_layer(2)
    neural_network.add_layer(2)
    neural_network.output_neuron(1)
    # print(neural_network.synaptic_weights)
    training_set_input = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_input, training_set_outputs)
    # print('Predicting')
    # print('Consider new situation [1, 0, 0] -> ?: ')
    # print(neural_network.predict(array([1, 0, 0])))
