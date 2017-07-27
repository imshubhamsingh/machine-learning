from numpy import *


class NLayerNeuralNetwork:
    def __init__(self):
        self.layer = 1
        self.synaptic_weights = {}
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
        self.synaptic_weights[self.layer - 1] = random.uniform(
            size=(self.neurons_in_layers[self.layer - 1], input))
        # # print(self.neurons_in_layers[self.layer-1])
        self.neurons_in_layers[self.layer] = input

    def add_layer(self, no_of_sigmoid_neurons):
        self.layer += 1
        # adding random weight to new layer
        self.synaptic_weights[self.layer - 1] = random.uniform(
            size=(self.neurons_in_layers[self.layer - 1], no_of_sigmoid_neurons))
        # setting initial adjustment to zero
        self.neurons_in_layers[self.layer] = no_of_sigmoid_neurons

    def backward_propagation(self, training_set_inputs):
        for layer in sorted(self.neurons_in_layers.keys(), reverse=True):
            if layer != len(self.neurons_in_layers) and layer != 1:
                self.delta[layer] = self.delta[layer + 1].dot(
                    self.synaptic_weights[layer].T) * self.__sigmoid_derivative(
                    self.hidden_layer_value[layer])

        for layer in sorted(self.neurons_in_layers.keys(), reverse=True):
            if layer > 2:
                self.synaptic_weights[layer - 1] += self.hidden_layer_value[layer - 1].T.dot(self.delta[layer])
            elif layer == 2:
                self.synaptic_weights[layer - 1] += training_set_inputs.T.dot(self.delta[layer])
            else:
                pass

    def predict(self, inputs):
        for layer in self.neurons_in_layers:
            # # print(layer)
            if layer != len(self.neurons_in_layers):
                value = self.__sigmoid(dot(inputs, self.synaptic_weights[layer]))
                inputs = value
                if layer < len(self.neurons_in_layers) - 1:
                    self.hidden_layer_value[layer + 1] = value
                else:
                    self.output_layer_value = value
        return self.output_layer_value

    def train(self, training_set_inputs, training_set_outputs, number_of_iterations=6000):
        for iteration in range(number_of_iterations):
            output = self.predict(training_set_inputs)
            error_in_output = training_set_outputs - output
            self.delta[len(self.neurons_in_layers)] = error_in_output * self.__sigmoid_derivative(output)
            self.backward_propagation(training_set_inputs)


if __name__ == "__main__":
    neural_network = NLayerNeuralNetwork()
    neural_network.neurons_in_input_layer(3)
    # hidden layer
    neural_network.add_layer(3)
    neural_network.add_layer(4)
    # output layer
    neural_network.output_layer_neuron(1)

    # training_set_input = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # training_set_outputs = array([[0], [1], [1], [0]])
    training_set_input = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0], [1], [1], [0]])

    neural_network.train(training_set_input, training_set_outputs)
    print('Predicting')
    print('Consider new situation [1, 0, 0] -> ?: ')
    print(neural_network.predict(array([1, 0, 0])))
