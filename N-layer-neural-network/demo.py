from numpy import *


class NLayerNeuralNetwork:
    def __init__(self):
        self.layer = 1
        self.synaptic_weights = {}
        self.adjustment = {}
        self.neurons_in_layers = {}

    # activation function
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def neurons_in_input_layer(self, inputs):
        self.neurons_in_layers[self.layer] = inputs

    def final_neuron(self, number):
        self.synaptic_weights[self.layer] = 2 * random.random(
            (number, self.neurons_in_layers[self.layer - 1])) - 1
        self.adjustment[self.layer + 1] = zeros((number, self.neurons_in_layers[self.layer - 1]))
        self.neurons_in_layers[self.layer + 1] = 1

    def add_layer(self, no_of_sigmoid_neurons):
        self.layer += 1
        # adding random weight to new layer
        self.synaptic_weights[self.layer - 1] = 2 * random.random(
            (no_of_sigmoid_neurons, self.neurons_in_layers[self.layer - 1])) - 1
        # setting initial adjustment to zero
        self.adjustment[self.layer] = zeros((no_of_sigmoid_neurons, 1))
        self.neurons_in_layers[self.layer] = no_of_sigmoid_neurons

    def predict(self, data):
        pass

    def train(self, inputs, targets, learning_rate=1, stop_accuracy=1e-5):
        error = []


if __name__ == "__main__":
    neural_network = NLayerNeuralNetwork()
    neural_network.neurons_in_input_layer(3)
    neural_network.add_layer(2)
    neural_network.add_layer(2)
    neural_network.final_neuron(1)
    print(neural_network.synaptic_weights)
    training_set_input = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T
