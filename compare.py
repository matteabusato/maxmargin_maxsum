import numpy as np
import itertools
np.random.seed(12)

# constants for testing
N = 5    # dimension of patterns
M = 3    # number of patterns M = alpha * N
THRESHOLD = 1e-4
ITERATIONS = 5
POSSIBLE_DELTA_MUS = [-N+i*2 for i in range(N+1)]
POSSIBLE_DELTA_UNDERLINED = [-N+1+i*2 for i in range(N)]
POSSIBLE_DELTA_MUS_SETS = list(itertools.product(POSSIBLE_DELTA_MUS, repeat=M))
POSSIBLE_WEIGHTS = list(itertools.product([-1, 1], repeat=N))
SETTING_PHI_DOWN = 0
COUNTERS = np.zeros(3)

# data structures for weights and patterns
weights = np.random.choice([-1, 1], size=N)
patterns = np.random.choice([-1, 1], size=(M, N))

# data structures for messages
q_up = np.zeros((N,M))
psi_up = np.zeros((M,N+1))
phi_up = np.zeros(N+1)

phi_down = np.zeros(N+1)
psi_down = np.zeros((M,N+1))
q_down = np.zeros((N,M))

# single-site quantities
q_singlesite = np.zeros(N)

# data structures for auxiliary messages
q_down_pos = np.zeros((N,M))
q_down_neg = np.zeros((N,M))
xi = np.zeros((M,N+1))
gamma = np.zeros((M,N+1))
ipsilon = np.zeros((M,N+1))

# data structures for auxiliary quantities
delta_tilde = np.zeros(M)
U_plus = [[] for _ in range(M)]
U_minus = [[] for _ in range(M)]
I_plus = [[] for _ in range(M)]
I_minus = [[] for _ in range(M)]
m_plus = np.zeros((M, 2))
M_plus = np.zeros(M)
m_minus = np.zeros((M,2))
M_minus = np.zeros(M)
delta_star = np.zeros((M,2))
k_star = np.zeros((M,2))

# data structures to store messages and weights  
q_up_old = np.zeros((N,M))
psi_up_old = np.zeros((M,N+1))
phi_up_old = np.zeros(N+1)

phi_down_old = np.zeros(N+1)
psi_down_old = np.zeros((M,N+1))
q_down_old = np.zeros((N,M))

weights_old = np.zeros(N)
weights_best = weights.copy()


# ------------------------------------------------------------------
# USEFUL FUNCTIONS

def delta_to_ind(delta):
    return int((delta + N) // 2)

def ind_to_s(i):
    return -1+i*2

def normalize(array):
    max_value = np.max(array)
    return array - max_value 

def sign_arr(x):
    array = x.copy()
    for i in range(len(array)):
        array[i] = sign_num(array[i])
    return array

def sign_num(x):
    return +1 if x >= 0 else -1

def store():
    global q_up_old
    global psi_up_old
    global phi_up_old
    global psi_down_old
    global q_down_old

    q_up_old = q_up.copy()
    psi_up_old = psi_up.copy()
    phi_up_old = phi_up.copy()
    psi_down_old = psi_down.copy()
    q_down_old = q_down.copy()


# ------------------------------------------------------------------
# SINGLE-SITE QUANTITIES and WEIGHTS UPDATES

def update_singlesite():
    global q_singlesite
    for i in range(N):
        q_singlesite[i] = sum(q_down[i])

def update_weights():
    global weights
    update_singlesite()
    for i in range(N):
        weights[i] = sign_num(q_singlesite[i])

# ------------------------------------------------------------------
# FORWARD PASS

def update_phi_up():
    global phi_up
    update_xi()
    update_gamma()
    for i in range(N+1):
        phi_up[i] = sum(xi.T[i]) + max(gamma.T[i])

    #NORMALIZE
    phi_up = normalize(phi_up)

def update_xi():
    global xi
    for mu in range(M):
        for delta_index in range(N+1):
            max1 = -np.inf
            for delta_prime_index in range(delta_index,N+1):
                max1 = max(max1, psi_up[mu][delta_prime_index])
            xi[mu][delta_index] = max1

def update_gamma():
    global gamma
    for mu in range(M):
        for delta_index in range(N+1):
            gamma[mu][delta_index] = psi_up[mu][delta_index] - xi[mu][delta_index]


def update_phi_up2():
    global phi_up
    for delta_star in POSSIBLE_DELTA_MUS:
        index_delta_star = delta_to_ind(delta_star)
        max1 = -np.inf
        for delta_mu_set in POSSIBLE_DELTA_MUS_SETS:
            if min(delta_mu_set) == delta_star:
                temp_sum = 0
                for mu in range(M):
                    temp_index = delta_to_ind(delta_mu_set[mu])
                    temp_sum += psi_up[mu][temp_index]
                max1 = max(max1, temp_sum)
        phi_up[index_delta_star] = max1

    #NORMALIZE
    phi_up = normalize(phi_up)

# ----------------------

def update_psi_up():
    global psi_up
    global U_minus
    global U_plus
    global I_minus
    global I_plus
    global delta_tilde
    for mu in range(M):
        weights_tilde = sign_arr(q_up.T[mu])
        delta_tilde[mu] = int(np.dot(patterns[mu], weights_tilde))
        psi_up_tilde = np.sum(np.abs(q_up.T[mu]))
        delta_index_tilde = delta_to_ind(delta_tilde[mu])
        psi_up[mu][delta_index_tilde] = psi_up_tilde

        U_plus[mu] = [i for i in range(N) if patterns[mu][i] == sign_num(q_up[i][mu])]
        U_minus[mu] = [i for i in range(N) if patterns[mu][i] != sign_num(q_up[i][mu])]
        I_plus[mu] = sorted(U_plus[mu].copy(),key=lambda i: np.abs(q_up[i][mu]))
        I_minus[mu] = sorted(U_minus[mu].copy(),key=lambda i: np.abs(q_up[i][mu]))

        psi_temp = psi_up_tilde
        delta_index = delta_index_tilde
        if I_minus[mu] is not None:
            for i in I_minus[mu]:
                delta_index += 1
                psi_temp -= np.abs(2*q_up[i][mu])
                psi_up[mu][delta_index] = psi_temp

        psi_temp = psi_up_tilde
        delta_index = delta_index_tilde
        if I_plus[mu] is not None:
            for i in I_plus[mu]:
                delta_index -= 1
                psi_temp -= 2*np.abs(q_up[i][mu])
                psi_up[mu][delta_index] = psi_temp
 
    #NORMALIZE
    psi_up = psi_up - np.sum(np.abs(q_up.T[mu]))

def update_psi_up2():
    global psi_up
    for mu in range(M):
        for delta in POSSIBLE_DELTA_MUS:
            delta_index = delta_to_ind(delta)
            max1 = -np.inf
            for poss_weights in POSSIBLE_WEIGHTS:
                if np.dot(poss_weights, patterns[mu]) == delta:
                    max1 = max(max1, np.dot(poss_weights, q_up.T[mu]))
            psi_up[mu][delta_index] = max1

        #NORMALIZE
    psi_up[mu] = normalize(psi_up[mu])

# ----------------------

def update_q_up():
    global q_up
    for i in range(N):
        total_q_down = sum(q_down[i])
        for mu in range(M):
            q_up[i][mu] = total_q_down - q_down[i][mu]

def update_q_up2():
    global q_up
    for i in range(N):
        total_q_down = sum(q_down[i])
        for mu in range(M):
            q_up[i][mu] = total_q_down - q_down[i][mu]

# ----------------------

def forward_pass():
    update_q_up()
    update_psi_up()
    update_phi_up()

def forward_pass2():
    update_q_up2()
    update_psi_up2()
    update_phi_up2()

# ------------------------------------------------------------------
# BACKWARD PASS

def update_phi_down():
    global phi_down
    match SETTING_PHI_DOWN:
        case 0:
            delta = 1 / N
            for i in range(N+1):
                phi_down[i] = -1 + delta*i
        case 1: # for exponential phi down
            pass

def update_phi_down2():
    global phi_down
    match SETTING_PHI_DOWN:
        case 0:
            delta = 1 / N
            for i in range(N+1):
                phi_down[i] = -1 + delta*i
        case 1: # for exponential phi down
            pass

# ----------------------

def update_psi_down():
    global psi_down
    update_ipsilon()
    for mu in range(M):
        for delta_mu_index in range(N+1):
            max1 = -np.inf
            for delta_star_index in range(delta_mu_index):
                max2 = -np.inf
                for ro in range(M):
                    if ro != mu:
                        max2 = max(max2, gamma[ro][delta_star_index])
                max1 = max(max1, max2 + ipsilon[mu][delta_star_index])
            psi_down[mu][delta_mu_index] = max(max1, ipsilon[mu][delta_mu_index])   

    #NORMALIZE
    psi_down = normalize(psi_down)

def update_ipsilon():
    global ipsilon
    for mu in range(M):
        for delta_index in range(N+1):
            ipsilon[mu][delta_index] = sum(xi.T[delta_index]) - xi[mu][delta_index] + phi_down[delta_index]

def update_psi_down2():
    global psi_down
    for mu in range(M):
        for delta_mu in POSSIBLE_DELTA_MUS:
            index_delta_mu = delta_to_ind(delta_mu)
            max1 = -np.inf
            for delta_star in POSSIBLE_DELTA_MUS[:index_delta_mu + 1]:
                index_delta_star = delta_to_ind(delta_star)
                max2 = -np.inf
                for delta_nu_set in POSSIBLE_DELTA_MUS_SETS:
                    if min(delta_nu_set) == delta_star:
                        temp_sum = 0
                        for nu in range(M):
                            if nu != mu:
                                temp_index = delta_to_ind(delta_nu_set[nu])
                                temp_sum += psi_up[nu][temp_index]
                        max2 = max(max2, temp_sum)
                max1 = max(max1, max2 + phi_down[index_delta_star])
            psi_down[mu][index_delta_mu] = max1

    #NORMALIZE
    psi_down = normalize(psi_down)

# ----------------------

def update_q_down():
    global q_down
    global m_plus
    global M_plus
    global m_minus
    global M_minus
    global delta_star
    global k_star
    for mu in range(M):
        k_tilde = np.array([-1]*N)
        if I_minus[mu] is not None:
            for (k,j) in enumerate(I_minus[mu]):
                k_tilde[j] = k

        for s_index in range(2):
            s = ind_to_s(s_index)
            max1 = -np.inf
            for delta_under in POSSIBLE_DELTA_UNDERLINED:
                index1 = delta_to_ind(delta_under + 1)
                index2 = delta_to_ind(delta_under + s)
                max1 = max(max1, psi_up[mu][index1] + psi_down[mu][index2])
            m_plus[mu][s_index] = max1
        M_plus[mu] = (m_plus[mu][1] - m_plus[mu][0]) / 2

        for s_index in range(2):
            s = ind_to_s(s_index)
            max1 = -np.inf
            argmax1 = -N+1
            for delta_under in POSSIBLE_DELTA_UNDERLINED:
                index1 = delta_to_ind(delta_under - 1)
                index2 = delta_to_ind(delta_under + s)
                if max1 < psi_up[mu][index1] + psi_down[mu][index2]:
                    max1 = psi_up[mu][index1] + psi_down[mu][index2]
                    argmax1 = delta_under
            delta_star[mu][s_index] = argmax1

            index1 = delta_to_ind(delta_star[mu][s_index] - 1)
            index2 = delta_to_ind(delta_star[mu][s_index] + s)
            m_minus[mu][s_index] = max1
            k_star[mu][s_index] = (delta_star[mu][s_index] - delta_tilde[mu] - 1) // 2
        M_minus[mu] = (m_minus[mu][1] - m_minus[mu][0]) / 2

        for j in U_plus[mu]:
            q_down[j][mu] = patterns[mu][j] * M_plus[mu]
        for j in U_minus[mu]:
            if k_tilde[j] > max(k_star[mu]):
                q_down[j][mu] = patterns[mu][j] * M_minus[mu]
            else:
                m_hat = np.zeros(2)
                for s_index in range(2): 
                    s = ind_to_s(s_index)

                    index1 = delta_to_ind(delta_tilde[mu])
                    index2 = delta_to_ind(delta_tilde[mu] + 1 + s)
                    max1 = psi_up[mu][index1] + psi_down[mu][index2]

                    for k in range(1, int(k_star[mu][s_index])+1):
                        index1 = delta_to_ind(delta_tilde[mu] + 2*k)
                        index2 = delta_to_ind(delta_tilde[mu] + 2*k + 1 + s)
                        temp_sum = psi_up[mu][index1] + psi_down[mu][index2]
                        if k > k_tilde[j]:
                            temp_sum -= 2*(np.abs(q_up[I_minus[mu][k]][mu]) - np.abs(q_up[j][mu]))
                        max1 = max(max1, temp_sum)
                    m_hat[s_index] = max1 - np.abs(q_up[j][mu])

                M_hat = (m_hat[1] - m_hat[0]) / 2
                q_down[j][mu] = patterns[mu][j] * M_hat

def update_q_down2():
    global q_down
    for mu in range(M):
        for i in range(N):
            update_q_down_pos(i,mu)
            update_q_down_neg(i,mu)
            q_down[i][mu] = (q_down_pos[i][mu] - q_down_neg[i][mu]) / 2

def update_q_down_pos(i,mu):
    global q_down_pos
    max1 = -np.inf
    for delta in POSSIBLE_DELTA_MUS:
        index1 = delta_to_ind(delta)
        max2 = -np.inf
        for poss_weights in POSSIBLE_WEIGHTS:
            if (np.dot(poss_weights, patterns[mu]) - poss_weights[i]*patterns[mu][i] + (+1)*patterns[mu][i]) == delta:
                max2 = max(max2, np.dot(poss_weights, q_up.T[mu]) - poss_weights[i]*q_up[i][mu])
        max1 = max(max1, max2 + psi_down[mu][index1])
    q_down_pos[i][mu] = max1

def update_q_down_neg(i,mu):
    global q_down_neg
    max1 = -np.inf
    for delta in POSSIBLE_DELTA_MUS:
        index1 = delta_to_ind(delta)
        max2 = -np.inf
        for poss_weights in POSSIBLE_WEIGHTS:
            if (np.dot(poss_weights, patterns[mu]) - poss_weights[i]*patterns[mu][i] + (-1)*patterns[mu][i]) == delta:
                max2 = max(max2, np.dot(poss_weights, q_up.T[mu]) - poss_weights[i]*q_up[i][mu])
        max1 = max(max1, max2 + psi_down[mu][index1])
    q_down_neg[i][mu] = max1

# ----------------------

def backward_pass():
    update_phi_down()
    update_psi_down()
    update_q_down()

def backward_pass2():
    update_phi_down2()
    update_psi_down2()
    update_q_down2()

# ------------------------------------------------------------------
# CONVERGENCE ITERATIONS

def check_differences():
    differences = []
    differences.append(max(abs(q_up - q_up_old)))
    differences.append(max(abs(psi_up - psi_up_old)))
    differences.append(max(abs(phi_up - phi_up_old)))
    differences.append(max(abs(psi_down - psi_down_old)))
    differences.append(max(abs(q_down - q_down_old)))
    max_delta = max(differences)

    if max_delta < THRESHOLD:
        COUNTERS[0] += 1

def check_weights():
    if weights == weights_old:
        COUNTERS[1] += 1

def converge():
    convergence = False

    for i in range(ITERATIONS):
        forward_pass()
        backward_pass()
        store()
        update_weights()

        print("ITERATION: ", i)
        print("q_up: ", q_up)
        print("psi_up: ", psi_up)
        print("phi_up: ", phi_up)
        print("phi_down: ", phi_down)
        print("psi_down: ", psi_down)
        print("q_down: ", q_down)
        print("weights: ", weights)
        print()
    
    return convergence
    
    
# ------------------------------------------------------------------
# MAIN
  
if __name__ == '__main__':
    # q_up = np.random.randint(-100, 100, (N, M))
    # psi_up = np.random.randint(-100, 100, (M, N+1))
    # phi_up = np.random.randint(-100, 100, (N+1))

    # phi_down = np.random.randint(-100, 100, (N+1))
    # psi_down = np.random.randint(-100, 100, (M, N+1))
    # q_down = np.zeros((N,M))

    print("The patterns are: ")
    print(patterns)
    print()
    print("The weights are: ")
    print(weights)
    print()

    # print("The messages are: ")
    # print("q_up: ", q_up)
    # print("psi_up: ", psi_up)
    # print("phi_up: ", phi_up)
    # print("phi_down: ", phi_down)
    # print("psi_down: ", psi_down)
    # print("q_down: ", q_down)

    for j in range(10):
        np.random.seed(12+j)
        weights = np.random.choice([-1, 1], size=N)
        patterns = np.random.choice([-1, 1], size=(M, N))
        for i in range(ITERATIONS):
            print("Iteration: ", i)
            update_q_up()
            test1 = q_up.copy()
            update_q_up2()
            test2 = q_up.copy()
            if (test1-test2).all()>10e-5:
                print("test1: ", test1)
                print("test2", test2)

            update_psi_up()
            test1 = psi_up.copy()
            update_psi_up2()
            test2 = psi_up.copy()
            if (test1-test2).all()>10e-5:
                print("test1: ", test1)
                print("test2", test2)

            update_phi_up()
            test1 = phi_up.copy()
            update_phi_up2()
            test2 = phi_up.copy()
            if (test1-test2).all()>10e-5:
                print("test1: ", test1)
                print("test2", test2)

            update_phi_down()
            test1 = phi_down.copy()
            update_phi_down2()
            test2 = phi_down.copy()
            if (test1-test2).all()>10e-5:
                print("test1: ", test1)
                print("test2", test2)

            update_psi_down()
            test1 = psi_down.copy()
            update_psi_down2()
            test2 = psi_down.copy()
            if (test1-test2).all()>10e-5:
                print("test1: ", test1)
                print("test2", test2)

            update_q_down()
            test1 = q_down.copy()
            update_q_down2()
            test2 = q_down.copy()
            if (test1-test2).all()>10e-5:
                print("test1: ", test1)
                print("test2", test2)

            update_weights()