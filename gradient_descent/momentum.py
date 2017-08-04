import numpy as np


def evaluate_gradient(c_current, m_current, points, learning_rate):
    c_v = 0
    m_v = 0
    aplha = 0.9
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        c_gradient = -(2 / n) * (y - ((m_current * x) + c_current))
        m_gradient = -(2 / n) * x * (y - ((m_current * x) + c_current))
        c_v = c_v * aplha + learning_rate * c_gradient
        m_v = m_v * aplha + learning_rate * m_gradient
        c_current = c_current - c_v
        m_current = m_current - m_v
    return [c_current, m_current]


def momentum(params, points, epoch, learning_rate):
    [c, m] = params
    for i in range(0, epoch):
        [c, m] = evaluate_gradient(c, m, points, learning_rate)
    return [c, m]


def run():
    x_data = np.linspace(-2, 2, 3000)[:, np.newaxis]
    # y = mx +c
    y_data = 0.7 * x_data + 0.3
    points = np.concatenate((x_data, y_data), axis=1)
    initial_m = 0
    initial_c = 0
    params = [initial_c, initial_m]
    epoch = 7
    learning_rate = 0.05
    params = momentum(params, points, epoch, learning_rate)
    print(params)


if __name__ == "__main__":
    run()
