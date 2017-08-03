import numpy as np


def evaluate_gradient(c_current, m_current, points, learning_rate):
    b_gradient = 0
    a_gradient = 0
    n = 1
    x = points[0]
    y = points[1]
    b_gradient += -(2 / n) * (y - ((m_current * x * x) + c_current))
    a_gradient += -(2 / n) * x * x * (y - ((m_current * x * x) + c_current))
    new_b = c_current - (learning_rate * b_gradient)
    new_a = m_current - (learning_rate * a_gradient)
    return [new_b, new_a]


def stochastic_gradient_descent(points, params, learning_rate, num_iteration):
    [b, a] = params
    for i in range(num_iteration):
        np.random.shuffle(points)
        for point in points:
            [b, a] = evaluate_gradient(b, a, point, learning_rate)
    return [b, a]


def run():
    x_data = np.linspace(-2, 2, 3000)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = 0.3 * np.square(x_data) + 0.4 + noise
    points = np.concatenate((x_data, y_data), axis=1)
    learning_rate = 0.05  # hyperparameter tuning knobs for model
    # y = ax^2 + b
    initial_b = 0
    initial_a = 0
    params = [initial_b, initial_a]
    num_iteration = 2
    [b, a] = stochastic_gradient_descent(points, params, learning_rate, num_iteration)
    print(b, a)


if __name__ == '__main__':
    run()
