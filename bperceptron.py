import numpy as np
import itertools
np.random.seed(12)

# constants for testing
N = 5    # dimension of patterns
M = 3    # number of patterns M = alpha * N
THRESHOLD = 1e-5
ITERATIONS = 2
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

def delta_to_ind(delta):
    return (delta + N) // 2

def normalize(array):
    max_value = np.max(array)
    return array - max_value 

def store():
    global q_up_old
    global psi_up_old
    global phi_up_old
    global psi_down_old
    global q_down_old

    q_up_old = q_up
    psi_up_old = psi_up
    phi_up_old = phi_up
    psi_down_old = psi_down
    q_down_old = q_down

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
# FORWARD PASS

def update_phi_up():
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

def update_psi_up():
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
    psi_up = normalize(psi_up)
                        
def update_q_up():
    global q_up
    for i in range(N):
        total_q_down = sum(q_down[i])
        for mu in range(M):
            q_up[i][mu] = total_q_down - q_down[i][mu]

def forward_pass():
    update_q_up()
    update_psi_up()
    update_phi_up()

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

def update_psi_down():
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
            
def update_q_down():
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

def backward_pass():
    update_phi_down()
    update_psi_down()
    update_q_down()

# ------------------------------------------------------------------
# CONVERGENCE ITERATIONS

def check_convergence():
    check_differences()
    check_weights()

    if max(COUNTERS) >= 2:
        return True
    
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
if __name__ == '__main__':
    converge()