import numpy as np


def evaluate_gradient(c_current, m_current, points, learning_rate):
    c_gradient = 0
    m_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        c_gradient += -(2 / n) * (y - ((m_current * x) + c_current))
        m_gradient += -(2 / n) * x * (y - ((m_current * x) + c_current))
    new_c = c_current - (learning_rate * c_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_c, new_m]


def stochastic_gradient_descent(points, initial_c, initial_m, learning_rate, num_iteration):
    c = initial_c
    m = initial_m
    for i in range(num_iteration):
        np.random.shuffle(points)
        for point in points:
            [c, m] = evaluate_gradient(c, m, point, learning_rate)
    return [c, m]


def run():
    x_data = np.linspace(-2, 2, 3000)[:, np.newaxis]
    y_data = 0.3 * x_data + 0.4
    points = np.array([x_data, y_data])
    learning_rate = 0.0001  # hyperparameter tuning knobs for model
    # y = mx + c
    initial_c = 0
    initial_m = 0
    num_iteration = 1000
    [c, m] = stochastic_gradient_descent(points, initial_c, initial_m, learning_rate, num_iteration)
    print(c, m)


if __name__ == '__main__':
    run()
