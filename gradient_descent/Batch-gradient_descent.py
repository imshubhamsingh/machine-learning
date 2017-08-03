from numpy import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def compute_error_for_given_points(c, m, points):
    # gradient descent
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + c)) ** 2
    return total_error / float(len(points))


def step_gradient(c_current, m_current, points, learning_rate):
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


def gradient_descent_runner(points, initial_c, initial_m, learning_rate, num_iteration):
    c = initial_c
    m = initial_m
    for i in range(num_iteration):
        c, m = step_gradient(c, m, array(points), learning_rate)
    return [c, m]


def run():
    points = genfromtxt('data.csv', delimiter=',')
    learning_rate = 0.0001  # hyperparameter tuning knobs for model
    # y = mx + c
    initial_c = 0
    initial_m = 0
    num_iteration = 1000
    [c, m] = gradient_descent_runner(points, initial_c, initial_m, learning_rate, num_iteration)
    print(c, m)
    print(compute_error_for_given_points(c, m, points))
    plt.scatter(points[0:, 0], points[0:, 1])
    x = array(range(100))
    plt.plot(x, m * x + c)
    plt.xlabel('Hours')
    plt.ylabel('Marks')
    plt.show()


if __name__ == '__main__':
    run()
