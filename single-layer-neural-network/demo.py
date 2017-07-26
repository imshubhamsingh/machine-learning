from numpy import array, random, exp, dot


class NeuralNetwork:
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def _sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
        for iteration in range(number_of_iterations):
            output = self.predict(training_set_inputs)

            error = training_set_outputs - output

            adjustment = dot(training_set_inputs.T, error * self._sigmoid_derivative(output))
            # print(output)

            self.synaptic_weights += adjustment

    def predict(self, inputs):
        print(self.synaptic_weights)
        return self._sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    # initialize a single neuron neural networkself.__sigmoid_derivative(output)
    neural_network = NeuralNetwork()

    print('Random starting synaptic weights')
    print(neural_network.synaptic_weights)

    # The training set. We have 4 example, each consisting of 3 input value
    # 0 and 1 output value

    training_set_input = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # train the neural network using training set
    # do it 10,000 times and make small adjustments each time

    neural_network.train(training_set_input, training_set_outputs, 10000)

    print('New synaptic weights after training')
    print(neural_network.synaptic_weights)

    # test the neural network with new values
    print('Predicting')
    print('Consider new situation [1, 0, 0] -> ?: ')
    print(neural_network.predict(array([1, 0, 0])))
