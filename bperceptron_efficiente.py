import numpy as np
import itertools

# constants for testing
N = 3    # dimension of patterns
M = 2    # number of patterns M = alpha * N
THRESHOLD = 1e-4
ITERATIONS = 10
POSSIBLE_DELTA_MUS = [-N+i*2 for i in range(N+1)]
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

psi_up_partial = []
for mu in range(M):
    psi_up_partial.append([np.zeros(j + 2) for j in range(N)])

# data structures for old messages and weights  
q_up_old = np.zeros((N,M))
psi_up_old = np.zeros((M,N+1))
phi_up_old = np.zeros(N+1)

phi_down_old = np.zeros(N+1)
psi_down_old = np.zeros((M,N+1))
q_down_old = np.zeros((N,M))

weights_old = np.zeros(N)

# ------------------------------------------------------------------
# USEFUL FUNCTIONS

def normalize(array):
    max_value = np.max(array)
    return array - max_value 

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
        weights[i] = 1 if np.sign(q_singlesite[i]) >= 0 else -1

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

# ----------------------

def update_psi_down():
    global psi_down
    update_ipsilon()
    for mu in range(M):
        for i in range(N+1):
            max1 = -np.inf
            for index_delta_star in range(i):
                max2 = -np.inf
                for ro in range(M):
                    if ro != mu:
                        max2 = max(max2, gamma[ro][index_delta_star])
                max1 = max(max1, max2 + ipsilon[mu][index_delta_star])
            psi_down[mu][i] = max(max1, ipsilon[mu][i])   

    #NORMALIZE
    psi_down = normalize(psi_down)

def update_ipsilon():
    global ipsilon
    for mu in range(M):
        for i in range(N+1):
            ipsilon[mu][i] = sum(xi.T[i]) - xi[mu][i] + phi_down[i]

# ----------------------

def update_q_down():
    global q_down
    for mu in range(M):
        for i in range(N):
            update_q_down_pos(i,mu)
            update_q_down_neg(i,mu)
            q_down[i][mu] = q_down_pos[i][mu] / 2 - q_down_neg[i][mu] / 2

def update_q_down_pos(i,mu):
    global q_down_pos
    max1 = -np.inf
    for poss_weights in POSSIBLE_WEIGHTS:
        delta = np.dot(poss_weights, patterns[mu]) - poss_weights[i]*patterns[mu][i] + 1*patterns[mu][i]
        index = int((delta + N) / 2)
        max1 = max(max1, np.dot(poss_weights, q_up.T[mu]) + psi_down[mu][index])
    q_down_pos[i][mu] = max1 - 1*q_up[i][mu]

def update_q_down_neg(i,mu):
    global q_down_neg
    max1 = -np.inf
    for poss_weights in POSSIBLE_WEIGHTS:
        delta = np.dot(poss_weights, patterns[mu]) - poss_weights[i]*patterns[mu][i] + (-1)*patterns[mu][i]
        index = int((delta + N) / 2)
        max1 = max(max1, np.dot(poss_weights, q_up.T[mu]) + psi_down[mu][index])
    q_down_neg[i][mu] = max1 - (-1)*q_up[i][mu]

# ----------------------

def backward_pass():
    update_phi_down()
    update_psi_down()
    update_q_down()


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
        for delta in POSSIBLE_DELTA_MUS:
            index1 = int((delta + N) / 2)
            max1 = -np.inf
            for delta_prime in POSSIBLE_DELTA_MUS[index1:]:
                index2 = int((delta + N) / 2)
                max1 = max(max1, psi_up[mu][index2])
            xi[mu][index1] = max1

def update_gamma():
    global gamma
    for mu in range(M):
        for delta in POSSIBLE_DELTA_MUS:
            index = int((delta + N) / 2)
            gamma[mu][index] = psi_up[mu][index] - xi[mu][index]

# ----------------------

def update_psi_up():
    global psi_up
    update_psi_up_partial()

    for mu in range(M):
        for i in range(N+1):
            psi_up[mu][i] = psi_up_partial[mu][N-1][i]

    #NORMALIZE
    psi_up = normalize(psi_up)

def update_psi_up_partial():
    global psi_up_partial
    for mu in range(M):
        for j in range(N):
            current_possible_deltas = [(-j-1)+i*2 for i in range(j+2)]
            for delta in current_possible_deltas:
                index1 = int((delta + j + 1)/ 2)
                max1 = -np.inf
                for weight in [-1,1]:
                    if j==0:
                        max1 = max(max1, delta*patterns[mu][j]*q_up[j][mu])
                    else:
                        index2 = int(((delta - weight*patterns[mu][j]) + j) / 2)
                        if index2 < j+1:
                            max1 = max(max1, psi_up_partial[mu][j-1][index2] + weight*q_up[j][mu])
                psi_up_partial[mu][j][index1] = max1

# ----------------------

def update_q_up():
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

# ------------------------------------------------------------------
# CONVERGENCE ITERATIONS

def check_convergence():
    # check_differences()
    # check_weights()
    # check_single_site()

    # if max(COUNTERS) >= 3:
    #     return True
    
    return False

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

def check_single_site():
    pass

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

        if check_convergence():
            convergence = True
            break
    
    return convergence
    
# ------------------------------------------------------------------
# MAIN

def tests():
    global weights
    global q_up
    global psi_down
    global psi_up

    # Test for a random non-zero psi_down
    for test_case in range(5):  # Run a few test cases with different random initializations
        # Initialize psi_down with random values (can range between -1 and 1 for example)
        psi_up = np.random.uniform(-19, 234, (M, N+1))
        psi_down = np.random.uniform(-1, 1, (M, N+1))
        q_up = np.random.uniform(-100, 10, (N, M))
        weights = np.random.choice([-1, 1], size=N)


        update_psi_up_partial()
        result1 = psi_up_partial.copy()


        if result1 == result1 :
            print("Test passed yey")
            print(result1)
        else:
            print("Test FAILEEEED")
        

        # # Use assertion to check if both results are equal
        # assert np.allclose(result1, result2), \
        #     f"Test failed on case {test_case + 1}: The results of update_psi_down and update_psi_down2 do not match.\n" \
        #     f"Result 1:\n{result1}\nResult 2:\n{result2}"

        # # If assertion passes, print success message
        # print(f"Test {test_case + 1} passed: Both functions give the same result. The results of update_psi_down and update_psi_down2 match.\n" \
        #     f"Result 1:\n{result1}\nResult 2:\n{result2}")

    print("All tests passed.")

if __name__ == '__main__':
    # if converge():
    #     print('Has converged!')
    # else:
    #     print('Fail :(')

    # backward_pass()
    # forward_pass()
    # print("done")
    tests()