import numpy as np


def initialize_parameters(num_agents):
    theta = np.random.rand(num_agents)
    x = np.random.rand(num_agents)
    D = np.diag(np.random.randint(1, 5, size=num_agents))
    A = np.random.randint(0, 2, size=(num_agents, num_agents))
    A = (A + A.T) / 2  # 保证邻接矩阵是对称的
    S = np.eye(num_agents)  # 调控随机扰动的方向和幅度，简单设为单位矩阵
    c = np.random.rand(num_agents)
    q = np.random.rand(num_agents) * 0.1  # 保证 q < 1
    num_iterations = 100
    initial_h = 0.01
    h = initial_h
    return theta, x, D, A, S, c, q, h, num_iterations


def generate_laplacian_noise(b):
    return np.random.laplace(0, b, size=b.shape)


def distributed_differential_privacy(theta, x, h, D, A, S, c, q, num_iterations):
    d_max = np.max(D)
    if h >= 1 / d_max:
        raise ValueError("Step size h must be less than (d_max)^{-1}")

    L = D - A  # Laplacian matrix
    n = len(theta)

    for k in range(num_iterations):
        # 调整 h, c, q
        h = min(h * 1.01, 1 / d_max)  # 每次迭代逐渐增加 h，最大值为 1 / d_max
        c = c * 0.99  # 每次迭代逐渐减小 c
        q = q * 1.01  # 每次迭代逐渐增加 q

        eta = np.array([generate_laplacian_noise(c_i * q_i ** k) for c_i, q_i in zip(c, q)])
        theta = theta - h * L @ x + S @ eta
        x = theta + eta

    return theta, x
