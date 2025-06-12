import numpy as np

class Parameters:
    def __init__(self, N, M, THRESHOLD, iterations, setting_phi_down, setting_q, r):
        self.N = N
        self.M = M
        self.THRESHOLD = THRESHOLD
        self.iterations = iterations
        self.setting_phi_down = setting_phi_down
        self.setting_q = setting_q
        self.r = r

class MaxSum:
    def __init__(self, param):
        self.N = param.N
        self.M = param.M
        self.THRESHOLD =  param.THRESHOLD
        self.ITERATIONS = param.iterations
        self.WEIGHTS_MAX_ITERATIONS = 10
        self.MESSAGES_MAX_ITERATIONS = 10
        self.POSSIBLE_DELTA_UNDERLINED = [-self.N+1+i*2 for i in range(self.N)]
        self.setting_phi_down = param.setting_phi_down
        self.setting_q = param.setting_q
        self.R = param.r
        self.COUNTERS = [0,0]
        self.convergence_iteration = param.iterations
        self.time = 0

        self.weights = np.random.choice([-1, 1], size=self.N)
        self.patterns = np.random.choice([-1, 1], size=(self.M, self.N))

        # data structures for messages
        self.q_up = np.zeros((self.N, self.M))
        self.psi_up = np.zeros((self.M, self.N+1))
        self.phi_up = np.zeros(self.N+1)
        self.phi_down = np.zeros(self.N+1)
        self.psi_down = np.zeros((self.M, self.N+1))
        self.q_down = np.zeros((self.N, self.M))

        # single-site quantities
        self.q_singlesite = np.zeros(self.N)

        # data structures for auxiliary messages
        self.xi = np.zeros((self.N+1, self.M))
        self.gamma = np.zeros((self.N+1, self.M))
        self.ipsilon = np.zeros((self.M, self.N+1))

        # data structures for auxiliary quantities
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

        # data structures to store messages and weights
        self.q_up_old = np.zeros((self.N, self.M))
        self.psi_up_old = np.zeros((self.M, self.N+1))
        self.phi_up_old = np.zeros(self.N+1)
        self.phi_down_old = np.zeros(self.N+1)
        self.psi_down_old = np.zeros((self.M, self.N+1))
        self.q_down_old = np.zeros((self.N, self.M))

        self.q_singlesite_old = np.zeros(self.N)

        self.weights_old = np.zeros(self.N)
        self.weights_best = self.weights.copy()
        self.delta_star_max = -self.N
        self.delta_star_convergence = -self.N

        self.converged = False

    def delta_to_ind(self, delta):
        return int((delta + self.N) // 2)

    def ind_to_s(self, i):
        return -1 + i * 2

    def normalize(self, array):
        return array - np.max(array)

    def sign_num(self, x):
        return +1 if x >= 0 else -1

    def sign_arr(self, x):
        x = np.asarray(x)
        return np.where(x >= 0, 1, -1)
    
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

    def update_phi_up(self):
        self.update_xi()
        self.update_gamma()
        self.phi_up = np.sum(self.xi, axis=1) + np.max(self.gamma, axis=1)

        # NORMALIZE
        self.phi_up = self.normalize(self.phi_up)

    def update_xi(self):
        self.xi[:, :] = np.maximum.accumulate(self.psi_up[:, ::-1], axis=1)[:, ::-1].T

    def update_gamma(self):
        self.gamma[:, :] = self.psi_up.T - self.xi

    def update_psi_up(self):
        for mu in range(self.M):
            weights_tilde = self.sign_arr(self.q_up.T[mu])
            self.delta_tilde[mu] = int(np.dot(self.patterns[mu], weights_tilde))
            psi_up_tilde = np.sum(np.abs(self.q_up.T[mu]))
            delta_index_tilde = self.delta_to_ind(self.delta_tilde[mu])
            self.psi_up[mu][delta_index_tilde] = psi_up_tilde

            self.U_plus[mu] = [i for i in range(self.N) if self.patterns[mu][i] == self.sign_num(self.q_up[i][mu])]
            self.U_minus[mu] = [i for i in range(self.N) if self.patterns[mu][i] != self.sign_num(self.q_up[i][mu])]

            self.I_plus[mu] = sorted(self.U_plus[mu].copy(), key=lambda i: np.abs(self.q_up[i][mu]))
            self.I_minus[mu] = sorted(self.U_minus[mu].copy(), key=lambda i: np.abs(self.q_up[i][mu]))

            psi_temp = psi_up_tilde
            delta_index = delta_index_tilde
            for i in self.I_minus[mu]:
                delta_index += 1
                psi_temp -= np.abs(2 * self.q_up[i][mu])
                self.psi_up[mu][delta_index] = psi_temp

            psi_temp = psi_up_tilde
            delta_index = delta_index_tilde
            if self.I_plus[mu] is not None:
                for i in self.I_plus[mu]:
                    delta_index -= 1
                    psi_temp -= 2 * np.abs(self.q_up[i][mu])
                    self.psi_up[mu][delta_index] = psi_temp

            # NORMALIZE
            self.psi_up[mu] = self.psi_up[mu] - np.sum(np.abs(self.q_up.T[mu]))

    def update_q_up(self):
        self.q_up[:, :] = np.sum(self.q_down, axis=1, keepdims=True) - self.q_down

        if self.setting_q == 1:
            self.q_up += self.R * self.time * self.q_singlesite_old[:, np.newaxis]

    def update_phi_down(self):
        match self.setting_phi_down:
            case 0:  # linear spacing
                delta = 1 / self.N
                for i in range(self.N + 1):
                    self.phi_down[i] = -1 + delta * i
            case 1:  # quadratic spacing
                x = np.linspace(0, 1, self.N + 1)
                x2 = x**2
                x2_norm = (x2 - x2[0]) / (x2[-1] - x2[0]) - 1
                self.phi_down[:] = x2_norm
            case 2:  # exponential spacing
                x = np.linspace(0, 1, self.N + 1)
                exp_x = np.exp(x)
                exp_x_norm = (exp_x - exp_x[0]) / (exp_x[-1] - exp_x[0]) - 1
                self.phi_down[:] = exp_x_norm

    def update_psi_down(self):
        rho_star = np.full(self.N+1, -1)
        omega = np.full(self.N + 1, -np.inf)
        omega_prime = np.full(self.N + 1, -np.inf)
        gamma_overlined = np.zeros((self.M, self.N + 1))
        lambda_delta = np.zeros((self.M, self.N + 1))

        self.update_ipsilon()

        for delta in range(self.N + 1):
            for rho in range(self.M):
                if self.gamma[delta][rho] > omega[delta]:
                    omega[delta] = self.gamma[delta][rho]
                    rho_star[delta] = rho
            for rho in range(self.M):
                if rho != rho_star[delta] and self.gamma[delta][rho] > omega_prime[delta]:
                    omega_prime[delta] = self.gamma[delta][rho]

        for mu in range(self.M):
            for delta in range(self.N + 1):
                if rho_star[delta] == mu:
                    gamma_overlined[mu][delta] = omega[delta]
                else:
                    gamma_overlined[mu][delta] = omega_prime[delta]

        for mu in range(self.M):
            for delta in range(self.N + 1):
                if delta == 0:
                    lambda_delta[mu][delta] = -np.inf
                else:
                    lambda_delta[mu][delta] = max(gamma_overlined[mu][delta - 1] + self.ipsilon[mu][delta - 1], lambda_delta[mu][delta - 1])

        for mu in range(self.M):
            for delta_mu in range(self.N + 1):
                self.psi_down[mu][delta_mu] = max(lambda_delta[mu][delta_mu], self.ipsilon[mu][delta_mu])

        self.psi_down = self.normalize(self.psi_down)

    def update_ipsilon(self):
        for mu in range(self.M):
            for delta_index in range(self.N + 1):
                self.ipsilon[mu][delta_index] = np.sum(self.xi[delta_index]) - self.xi[delta_index][mu] + self.phi_down[delta_index]

    def update_q_down(self):
        for mu in range(self.M):
            abs_q_up_mu = np.abs(self.q_up[:, mu])

            k_tilde = np.full(self.N, -1)
            for k, j in enumerate(self.I_minus[mu]):
                k_tilde[j] = k

            for s_index in range(2):
                s = self.ind_to_s(s_index)
                delta_under = np.array(self.POSSIBLE_DELTA_UNDERLINED)
                idx1 = np.array([self.delta_to_ind(d + 1) for d in delta_under])
                idx2 = np.array([self.delta_to_ind(d + s) for d in delta_under])

                values = self.psi_up[mu][idx1] + self.psi_down[mu][idx2]
                self.m_plus[mu][s_index] = np.max(values)

            self.M_plus[mu] = (self.m_plus[mu][1] - self.m_plus[mu][0]) / 2

            for s_index in range(2):
                s = self.ind_to_s(s_index)
                max1 = -np.inf
                argmax1 = -self.N + 1
                for d in self.POSSIBLE_DELTA_UNDERLINED:
                    idx1 = self.delta_to_ind(d - 1)
                    idx2 = self.delta_to_ind(d + s)
                    val = self.psi_up[mu][idx1] + self.psi_down[mu][idx2]
                    if val > max1:
                        max1 = val
                        argmax1 = d
                self.delta_star[mu][s_index] = argmax1
                idx1 = self.delta_to_ind(argmax1 - 1)
                idx2 = self.delta_to_ind(argmax1 + s)
                self.m_minus[mu][s_index] = max1
                self.k_star[mu][s_index] = (argmax1 - self.delta_tilde[mu] - 1) // 2

            self.M_minus[mu] = (self.m_minus[mu][1] - self.m_minus[mu][0]) / 2

            for j in self.U_plus[mu]:
                self.q_down[j][mu] = self.patterns[mu][j] * self.M_plus[mu]

            I_minus_mu = self.I_minus[mu]
            k_star_max = max(self.k_star[mu])

            for j in self.U_minus[mu]:
                if k_tilde[j] > k_star_max:
                    self.q_down[j][mu] = self.patterns[mu][j] * self.M_minus[mu]
                else:
                    m_hat = np.zeros(2)
                    for s_index in range(2):
                        s = self.ind_to_s(s_index)

                        idx1 = self.delta_to_ind(self.delta_tilde[mu])
                        idx2 = self.delta_to_ind(self.delta_tilde[mu] + 1 + s)
                        max1 = self.psi_up[mu][idx1] + self.psi_down[mu][idx2]

                        k_star_val = int(self.k_star[mu][s_index])
                        for k in range(1, k_star_val + 1):
                            d1 = self.delta_to_ind(self.delta_tilde[mu] + 2 * k)
                            d2 = self.delta_to_ind(self.delta_tilde[mu] + 2 * k + 1 + s)
                            temp_sum = self.psi_up[mu][d1] + self.psi_down[mu][d2]

                            if k > k_tilde[j]:
                                q_k = abs_q_up_mu[I_minus_mu[k]]
                                q_j = abs_q_up_mu[j]
                                temp_sum -= 2 * (q_k - q_j)

                            if temp_sum > max1:
                                max1 = temp_sum

                        m_hat[s_index] = max1 - abs_q_up_mu[j]

                    M_hat = (m_hat[1] - m_hat[0]) / 2
                    self.q_down[j][mu] = self.patterns[mu][j] * M_hat

    def update_singlesite(self):
        self.q_singlesite[:] = np.sum(self.q_down, axis=1)
        if self.setting_q == 1:
            self.q_singlesite += self.R * self.time * self.q_singlesite_old

    def update_weights(self):
        self.update_singlesite()
        self.weights[:] = self.sign_arr(self.q_singlesite)

    def check_convergence(self):
        self.check_weights()

        if self.COUNTERS[0] >= self.WEIGHTS_MAX_ITERATIONS:
            return True
        return False

    def check_weights(self):
        if np.array_equal(self.weights, self.weights_old):
            self.COUNTERS[0] += 1
        else:
            self.COUNTERS[0] = 0

    def check_differences(self):
        differences = []
        differences.append(np.max(np.abs(self.q_up - self.q_up_old)))
        differences.append(np.max(np.abs(self.psi_up - self.psi_up_old)))
        differences.append(np.max(np.abs(self.phi_up - self.phi_up_old)))
        differences.append(np.max(np.abs(self.psi_down - self.psi_down_old)))
        differences.append(np.max(np.abs(self.q_down - self.q_down_old)))

        if np.max(differences) < self.THRESHOLD:
            self.COUNTERS[1] += 1
        else:
            self.COUNTERS[1] = 0
        
    def forward_pass(self):
        self.update_q_up()
        self.update_psi_up()
        self.update_phi_up()

    def backward_pass(self):
        self.update_phi_down()
        self.update_psi_down()
        self.update_q_down()

    def converge(self):
        convergence = False
        for i in range(self.ITERATIONS):
            self.forward_pass()
            self.backward_pass()
            self.update_weights()

            print("ITERATION: ", i, "weights: ", self.weights)

            if self.check_convergence():
                convergence = True
                self.convergence_iteration = i
                self.delta_star_convergence = min(np.dot(self.weights, self.patterns.T))
                break

            self.store()
            self.time += 1

        return convergence
    
def run_simulations(spd, sq):
    N = 1001
    M = 700
    THRESHOLD = 1e-4
    ITERATIONS = 10000
    SETTING_PHI_DOWN = spd   # 0: linear, 1: squared, 2: exponential
    SETTING_Q = sq          # 0: non-forced, 1: forced
    R = 0.001

    file_name = "new_results/results_"
    if SETTING_Q == 0:
        file_name += "non_"
    file_name += "forced_"
    if SETTING_PHI_DOWN == 0:
        file_name += "linear.txt"
    elif SETTING_PHI_DOWN == 1:
        file_name += "squared.txt"
    else:
        file_name += "exponential.txt"

    print(file_name)

    for i in range(100):
        seed = i + 612
        np.random.seed(seed)

        param = Parameters(
            N=N,
            M=M,
            THRESHOLD=THRESHOLD,
            iterations=ITERATIONS,
            setting_phi_down=SETTING_PHI_DOWN,
            setting_q=SETTING_Q,
            r=R
        )
        model = MaxSum(param)
        converged = model.converge()

        with open(file_name, 'a') as f:
            f.write(f"\n{N}  {M}  {M/N}  {ITERATIONS}  {converged}  {model.convergence_iteration}  {model.delta_star_max}  {model.delta_star_convergence}  {seed}")
            f.flush()

if __name__ == "__main__":
    run_simulations(2, 1)
