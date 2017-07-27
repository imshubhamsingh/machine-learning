from numpy import *


class NLayerNeuralNetwork:
    def __init__(self):
        self.layer = 1
        self.synaptic_weights = {}
        self.adjustment = {}
        self.neurons_in_layers = {}
        self.hidden_layer_value = {}
        self.output_layer_value = {}
        self.delta = {}

    # activation function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def neurons_in_input_layer(self, inputs):
        self.neurons_in_layers[self.layer] = inputs

    def output_layer_neuron(self, input):
        self.layer += 1
        self.synaptic_weights[self.layer - 1] = 2 * random.random(
            (self.neurons_in_layers[self.layer - 1], input)) - 1
        # print(self.neurons_in_layers[self.layer])
        self.adjustment[self.layer] = zeros((self.neurons_in_layers[self.layer - 1], input))
        self.neurons_in_layers[self.layer] = input

    def add_layer(self, no_of_sigmoid_neurons):
        self.layer += 1
        # adding random weight to new layer
        self.synaptic_weights[self.layer - 1] = 2 * random.random(
            (self.neurons_in_layers[self.layer - 1], no_of_sigmoid_neurons)) - 1
        # setting initial adjustment to zero
        self.adjustment[self.layer] = zeros((self.neurons_in_layers[self.layer - 1], no_of_sigmoid_neurons))
        self.neurons_in_layers[self.layer] = no_of_sigmoid_neurons

    def backward_propagation(self, training_set_inputs):
        for layer in sorted(self.neurons_in_layers.keys(), reverse=True):
            if layer != len(self.neurons_in_layers) and layer != 1:
                self.delta[layer] = self.delta[layer + 1].dot(
                    self.synaptic_weights[layer].T) * self.__sigmoid_derivative(
                    self.hidden_layer_value[layer - 1])

        for layer in sorted(self.neurons_in_layers.keys(), reverse=True):
            if layer == len(self.neurons_in_layers):
                self.synaptic_weights[layer - 1] += self.output_layer_value.T.dot(self.delta[layer])
            elif layer < len(self.neurons_in_layers) and layer > 2:
                self.synaptic_weights[layer - 1] += self.hidden_layer_value[layer - 1].T.dot(self.delta[layer])
            elif layer == 1:
                self.synaptic_weights[layer] += training_set_inputs.T.dot(self.delta[layer + 1])
            else:
                pass

    def predict(self, inputs):
        value = None
        for layer in self.neurons_in_layers:
            if layer != len(self.neurons_in_layers):
                value = self.__sigmoid(dot(inputs, self.synaptic_weights[layer]))
                inputs = value
                if layer < len(self.neurons_in_layers) - 1:
                    self.hidden_layer_value[layer] = value
                else:
                    self.output_layer_value = value
        return self.output_layer_value

    def train(self, training_set_inputs, training_set_outputs, number_of_iterations=200000):
        output = None
        for iteration in range(number_of_iterations):
            output = self.predict(training_set_inputs)
            error_in_output = training_set_outputs - output
            # print("error in output: ", error_in_output)
            self.delta[len(self.neurons_in_layers)] = error_in_output * self.__sigmoid_derivative(output)
            self.backward_propagation(training_set_inputs)
        print("output: ", output)


if __name__ == "__main__":
    neural_network = NLayerNeuralNetwork()
    neural_network.neurons_in_input_layer(2)
    # output layer
    neural_network.output_layer_neuron(1)
    # print(neural_network.synaptic_weights)
    training_set_input = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_set_outputs = array([[1], [0], [0], [1]])

    neural_network.train(training_set_input, training_set_outputs)
    # print('Predicting')
    # print('Consider new situation [1, 0, 0] -> ?: ')
    # print(neural_network.predict(array([1, 0, 0])))
