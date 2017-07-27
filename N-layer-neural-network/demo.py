from numpy import *


class NLayerNeuralNetwork:
    def __init__(self):
        # initializing neural network with default input layer
        self.layer = 1
        self.synaptic_weights = {}  # synaptic weight value of links between layers
        self.neurons_in_layers = {}  # number of neurons in each layer
        self.hidden_layer_value = {}  # values of neurons in each layer
        self.output_layer_value = {}  # value of neuron(s) in output layer
        self.delta = {}  # delta output sum

    # activation function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # sigmoid prime
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # input layer
    def neurons_in_input_layer(self, inputs):
        self.neurons_in_layers[self.layer] = inputs

    # output layer
    def output_layer_neuron(self, input):
        self.layer += 1
        self.synaptic_weights[self.layer - 1] = random.uniform(
            size=(self.neurons_in_layers[self.layer - 1], input))
        # # print(self.neurons_in_layers[self.layer-1])
        self.neurons_in_layers[self.layer] = input

    # hidden layer
    def add_layer(self, no_of_sigmoid_neurons):
        self.layer += 1
        # adding random weight to new layer
        self.synaptic_weights[self.layer - 1] = random.uniform(
            size=(self.neurons_in_layers[self.layer - 1], no_of_sigmoid_neurons))
        self.neurons_in_layers[self.layer] = no_of_sigmoid_neurons

    # backward propagation to adjust weights
    def backward_propagation(self, training_set_inputs):
        # calculating Delta weights for hidden layers
        for layer in sorted(self.neurons_in_layers.keys(), reverse=True):
            # as for first layer there is no need for Delta weights and final layer Delta weights are calculated before
            if layer != len(self.neurons_in_layers) and layer != 1:
                self.delta[layer] = self.delta[layer + 1].dot(
                    self.synaptic_weights[layer].T) * self.__sigmoid_derivative(
                    self.hidden_layer_value[layer])

        # adjusting weights for all layers
        for layer in sorted(self.neurons_in_layers.keys(), reverse=True):
            # adjusting weights till second layer from final layer as second layer adjustment depends on first layer
            if layer > 2:
                self.synaptic_weights[layer - 1] += self.hidden_layer_value[layer - 1].T.dot(self.delta[layer])
            elif layer == 2:
                # adjusting weight between second layer and first layer
                self.synaptic_weights[layer - 1] += training_set_inputs.T.dot(self.delta[layer])
            else:
                pass  # pass when layer is 1

    def predict(self, inputs):
        for layer in self.neurons_in_layers:
            # I'm not checking the final layer because for every n layer there will only be n-1 links
            if layer != len(self.neurons_in_layers):
                value = self.__sigmoid(dot(inputs, self.synaptic_weights[layer]))
                # value of neurons in next layer depends on previous layer
                inputs = value

                if layer < len(self.neurons_in_layers) - 1:
                    # saving the value for hidden layer neurons
                    self.hidden_layer_value[layer + 1] = value
                else:
                    # on second last layer, saving the value for final layer neuron(s)
                    self.output_layer_value = value
        # return output layer value
        return self.output_layer_value

    # train neural network
    def train(self, training_set_inputs, training_set_outputs, number_of_iterations=6000):
        # the network will under go 6000 iteration to improve synaptic_weights
        for iteration in range(number_of_iterations):
            # get the output from output layer after computing it from input layer
            output = self.predict(training_set_inputs)
            # get the output sum margin of error
            error_in_output = training_set_outputs - output
            # delta output sum
            self.delta[len(self.neurons_in_layers)] = error_in_output * self.__sigmoid_derivative(output)
            # adjust the weights using backward propagation
            self.backward_propagation(training_set_inputs)


if __name__ == "__main__":
    # created instance of NLayerNeuralNetwork class
    neural_network = NLayerNeuralNetwork()

    # added input layer with 3 sigmoid perceptron as test set has three inputs
    neural_network.neurons_in_input_layer(3)

    # added 2 hidden layer with 3, 4 sigmoid perceptron
    neural_network.add_layer(3)
    neural_network.add_layer(4)

    # output layer with one neuron as the test set has one output
    neural_network.output_layer_neuron(1)

    # training_set_input = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # training_set_outputs = array([[0], [1], [1], [0]])
    training_set_input = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0], [1], [1], [0]])

    # training the neural network for the above test data
    neural_network.train(training_set_input, training_set_outputs)

    # this is pretty clear
    print('Predicting')
    print('Consider new situation [1, 0, 0] -> ?: ')
    print(neural_network.predict(array([1, 0, 0])))
