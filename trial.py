import numpy as np
from simulationpy import MaxSum, Parameters


class MaxSumVerifier(MaxSum):
    def __init__(self, param):
        super().__init__(param)

    def update_psi_down_original(self):
        self.update_ipsilon()
        for mu in range(self.M):
            for delta_mu_index in range(self.N + 1):
                max1 = -np.inf
                for delta_star_index in range(delta_mu_index):
                    max2 = -np.inf
                    for ro in range(self.M):
                        if ro != mu:
                            max2 = max(max2, self.gamma[delta_star_index][ro])
                    max1 = max(max1, max2 + self.ipsilon[mu][delta_star_index])
                self.psi_down[mu][delta_mu_index] = max(max1, self.ipsilon[mu][delta_mu_index])
        self.psi_down = self.normalize(self.psi_down)

    def update_psi_down_optimized(self):
        self.update_ipsilon()
        
        # Precompute max_gamma
        max_gamma = np.full((self.N + 1, self.M), -np.inf)
        for delta_star_index in range(self.N + 1):
            for mu in range(self.M):
                mask = np.arange(self.M) != mu
                max_gamma[delta_star_index][mu] = np.max(self.gamma[delta_star_index][mask])

        for mu in range(self.M):
            for delta_mu_index in range(self.N + 1):
                if delta_mu_index > 0:
                    temp = max_gamma[:delta_mu_index, mu] + self.ipsilon[mu][:delta_mu_index]
                    max1 = np.max(temp)
                else:
                    max1 = -np.inf
                self.psi_down[mu][delta_mu_index] = max(max1, self.ipsilon[mu][delta_mu_index])

        self.psi_down = self.normalize(self.psi_down)

    def test_equivalence(self):
        # Prepare data
        self.update_xi()
        self.update_gamma()

        # Run original
        self.update_psi_down_original()
        psi_down_orig = self.psi_down.copy()

        # Run optimized
        self.update_psi_down_optimized()
        psi_down_opt = self.psi_down.copy()

        # Compare
        if np.allclose(psi_down_orig, psi_down_opt, atol=1e-8):
            print("✅ Optimized and original `update_psi_down` produce the same result.")
        else:
            diff = np.max(np.abs(psi_down_orig - psi_down_opt))
            print("❌ Results differ! Max difference:", diff)
            print("Sample difference matrix:\n", psi_down_orig - psi_down_opt)


param = Parameters(N=101, M=60, THRESHOLD=1e-4, iterations=100, setting_phi_down=1, setting_q=1, r=0.01)
model = MaxSumVerifier(param)

np.random.seed(0)  # Ensure repeatability
model.q_up = np.random.randn(model.N, model.M)
model.psi_up = np.random.randn(model.M, model.N+1)

model.test_equivalence()
