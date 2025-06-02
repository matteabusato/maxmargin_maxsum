import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Parameters:
    def __init__(self, N, M, THRESHOLD, iterations, setting_phi_down, setting_q, r):
        self.N = N
        self.M = M
        self.THRESHOLD = THRESHOLD
        self.iterations = iterations
        self.setting_phi_down = setting_phi_down
        self.setting_q = setting_q
        self.r = r

def timing_decorator(func):
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        end = time.perf_counter()
        print(f"Function {func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

class MaxSum:
    def __init__(self, param):
        self.N = param.N
        self.M = param.M
        self.THRESHOLD = param.THRESHOLD
        self.ITERATIONS = param.iterations
        self.setting_phi_down = param.setting_phi_down
        self.setting_q = param.setting_q
        self.R = param.r
        self.time = 0

        self.weights = np.random.choice([-1, 1], size=self.N)
        self.patterns = np.random.choice([-1, 1], size=(self.M, self.N))
        self.q_up = np.zeros((self.N, self.M))
        self.psi_up = np.zeros((self.M, self.N+1))
        self.phi_up = np.zeros(self.N+1)
        self.phi_down = np.zeros(self.N+1)
        self.psi_down = np.zeros((self.M, self.N+1))
        self.q_down = np.zeros((self.N, self.M))
        self.q_singlesite = np.zeros(self.N)
        self.xi = np.zeros((self.N+1, self.M))
        self.gamma = np.zeros((self.N+1, self.M))
        self.ipsilon = np.zeros((self.M, self.N+1))
        self.delta_tilde = np.zeros(self.M)
        self.U_plus = [[] for _ in range(self.M)]
        self.U_minus = [[] for _ in range(self.M)]
        self.I_plus = [[] for _ in range(self.M)]
        self.I_minus = [[] for _ in range(self.M)]
        self.m_plus = np.zeros((self.M, 2))
        self.M_plus = np.zeros(self.M)
        self.m_minus = np.zeros((self.M, 2))
        self.M_minus = np.zeros(self.M)
        self.delta_star = np.zeros((self.M, 2))
        self.k_star = np.zeros((self.M, 2))
        self.q_up_old = np.zeros((self.N, self.M))
        self.psi_up_old = np.zeros((self.M, self.N+1))
        self.phi_up_old = np.zeros(self.N+1)
        self.psi_down_old = np.zeros((self.M, self.N+1))
        self.q_down_old = np.zeros((self.N, self.M))
        self.q_singlesite_old = np.zeros(self.N)
        self.weights_old = np.zeros(self.N)
        self.weights_best = self.weights.copy()
        self.delta_star_max = -self.N
        self.delta_star_convergence = -self.N
        self.convergence_iteration = param.iterations
        self.COUNTERS = [0, 0]
        self.POSSIBLE_DELTA_UNDERLINED = [-self.N+1+i*2 for i in range(self.N)]
        self.converged = False

    def delta_to_ind(self, delta):
        return int((delta + self.N) // 2)

    def ind_to_s(self, i):
        return -1 + i * 2

    def normalize(self, array):
        return array - np.max(array)

    def sign_arr(self, x):
        x = np.asarray(x)
        return np.where(x >= 0, 1, -1)

    @timing_decorator
    def update_q_up(self):
        self.q_up[:, :] = np.sum(self.q_down, axis=1, keepdims=True) - self.q_down
        if self.setting_q == 1:
            self.q_up += self.R * self.time * self.q_singlesite_old[:, np.newaxis]

    @timing_decorator
    def update_psi_up(self):
        pass  # Replace with the real logic

    @timing_decorator
    def update_phi_up(self):
        pass  # Replace with the real logic

    @timing_decorator
    def update_phi_down(self):
        pass  # Replace with the real logic

    @timing_decorator
    def update_psi_down(self):
        pass  # Replace with the real logic

    @timing_decorator
    def update_q_down(self):
        pass  # Replace with the real logic

    @timing_decorator
    def update_weights(self):
        self.q_singlesite[:] = np.sum(self.q_down, axis=1)
        if self.setting_q == 1:
            self.q_singlesite += self.R * self.time * self.q_singlesite_old
        self.weights[:] = self.sign_arr(self.q_singlesite)

    def check_convergence(self):
        if np.array_equal(self.weights, self.weights_old):
            self.COUNTERS[0] += 1
        else:
            self.COUNTERS[0] = 0
        return self.COUNTERS[0] >= 10

    def store(self):
        self.q_up_old = np.copy(self.q_up)
        self.psi_up_old = np.copy(self.psi_up)
        self.phi_up_old = np.copy(self.phi_up)
        self.psi_down_old = np.copy(self.psi_down)
        self.q_down_old = np.copy(self.q_down)
        self.q_singlesite_old = np.copy(self.q_singlesite)
        self.weights_old = np.copy(self.weights)
        current_delta_star = np.min(np.dot(self.weights, self.patterns.T))
        if current_delta_star > self.delta_star_max:
            self.delta_star_max = current_delta_star
            self.weights_best = self.weights.copy()

    def forward_pass(self):
        self.update_q_up()
        self.update_psi_up()
        self.update_phi_up()

    def backward_pass(self):
        self.update_phi_down()
        self.update_psi_down()
        self.update_q_down()

    def converge(self):
        for i in range(self.ITERATIONS):
            self.forward_pass()
            self.backward_pass()
            self.update_weights()
            print(f"ITERATION: {i}, weights: {self.weights}")
            if self.check_convergence():
                self.convergence_iteration = i
                self.delta_star_convergence = min(np.dot(self.weights, self.patterns.T))
                return True
            self.store()
            self.time += 1
        return False

def run_simulations():
    N = 101
    M = 60
    THRESHOLD = 1e-4
    ITERATIONS = 10000
    SETTING_PHI_DOWN = 1
    SETTING_Q = 0
    R = 0.01

    for i in range(2):
        seed = i * 5
        np.random.seed(seed)
        param = Parameters(N, M, THRESHOLD, ITERATIONS, SETTING_PHI_DOWN, SETTING_Q, R)
        model = MaxSum(param)
        converged = model.converge()
        print(f"Trial {i}, Converged: {converged}, Iterations: {model.convergence_iteration}")

run_simulations()
