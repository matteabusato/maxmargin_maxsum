import numpy as np
from itertools import permutations
import itertools

# constants for initial testing
N = 3
M = 2
THRESHOLD = 1e-4
ITERATIONS = 5

# ------------------------------------------------------------------
# SINGLE-SITE QUANTITIES and WEIGHTS UPDATES (calculated with formulas in 1505.05401v2)

# formula 6 diventa la 15 
# la 5 divent ala 14
def calculate_single_site_q_up(i, weight, q_down):
    temp_sum = 0
    for mu in range(M):
        temp_sum += q_down[mu][i](weight) # FIX DEPENDENCE ON WEIGHT

    return temp_sum

def update_weights(weights, q_down):
    possible_weights = [1, -1]
    for i in range(N):
        weights[i] = np.sign(calculate_single_site_q_up(i, possible_weights[0], q_down) - calculate_single_site_q_up(i, possible_weights[1], q_down))
    
    return weights

# ------------------------------------------------------------------
# BACKWARD PASS

def update_phi_down(phi_up):
    for i in range(N+1):
        phi_up[i] = phi_up[i] # 
    return phi_up

def update_psi_down(mu, psi_up, phi_down):
    possible_deltas = [-N+i*2 for i in range(N+1)] 
    possible_delta_nu = [-N+i*2 for i in range(N+1)] 
    temp_array1 = []
    psi_down = []

    for delta in possible_deltas:
        possible_delta_star = [-N+i*2 for i in range(N+1) if -N+i*2 <= delta]
        for delta_star in possible_delta_star:     
            temp_array2 = []            
            for delta_nus in list(itertools.product(possible_delta_nu, repeat=M)):
                if min(delta_nus) == delta_star:
                    # temp_array2.append(sum(psi_up))
                    temp_sum = 0
                    for i in range(M):
                        if i != mu:
                            index = int(abs(delta_nus[i] + N) / 2)
                            temp_sum += psi_up[i][index]
                    temp_array2.append(temp_sum)

            index =  int(abs(delta_star + N) / 2)
            temp_array1.append(max(temp_array2) + phi_down[index])

        psi_down.append(max(temp_array1))

    # NORMALIZE !!!

    return psi_down

def update_q_down(mu, i, patterns, q_up, psi_down):
    possible_delta = [-N+i*2 for i in range(N+1)] 
    possible_weights = [1, -1]
    temp_array_plus = []
    temp_array_minus = []

    for delta in possible_delta:
        temp_sum_plus = []
        temp_sum_minus = []
        index = abs(delta - N) / 2
        for weights in list(itertools.product(possible_weights, repeat=N)):
            if weights[i] == 1 and np.dot(weights, np.transpose(patterns[mu])) == delta:
                temp_sum_plus.append(np.dot(weights, np.transpose(q_up)[mu]) - weights[i]*np.transpose(q_up)[mu][i] + psi_down[mu][index])
            if weights[i] == -1 and np.dot(weights, np.transpose(patterns[mu])) == delta:
                temp_sum_minus.append(np.dot(weights, np.transpose(q_up)[mu]) - weights[i]*np.transpose(q_up)[mu][i] + psi_down[mu][index])

        if temp_sum_minus:
            temp_array_plus.append(max(temp_sum_plus))
        if temp_sum_minus:
            temp_array_minus.append(max(temp_sum_minus))

    q_plus = max(temp_array_plus)
    q_minus = max(temp_array_minus)

    return q_plus - q_minus

def backward_pass(binary_patterns, weights, messages):
    q_up, q_down, psi_up, psi_down, phi_up, phi_down = messages["q_up"], messages["q_down"], messages["psi_up"], messages["psi_down"], messages["phi_up"], messages["phi_down"]

    # update phi_down
    phi_down = update_phi_down(phi_up)
    messages["phi_down"] = phi_down
    
    # update psi_down
    for mu in range(M):
        psi_down[mu] = update_psi_down(mu, psi_up, phi_down)
    messages["psi_down"] = psi_down

    # update q_down
    # for mu in range(M):
    #     for i in range(N):
    #         q_down[mu][i] = update_q_down(mu, i, binary_patterns, q_up, psi_down)
    # messages["q_down"] = q_down

    return messages

# ------------------------------------------------------------------
# FORWARD PASS

def update_phi_up(psi_up):
    possible_delta_star = [-N+i*2 for i in range(N+1)]
    possible_delta_mus = [-N+i*2 for i in range(N+1)]
    phi_up = []

    temp_array = []
    for delta_star in possible_delta_star:
        for delta_mus in list(itertools.product(possible_delta_mus, repeat=M)):
            if min(delta_mus) == delta_star:
                temp_sum = 0    
                for i in range(M):
                    index = int(abs(delta_mus[i] + N) / 2)
                    temp_sum += psi_up[i][index]
                temp_array.append(temp_sum)
        phi_up.append(max(temp_array))

    # NORMALIZE !!! 

    return phi_up

def update_psi_up(mu, patterns, q_up):
    possible_deltas = [-N+i*2 for i in range(N+1)]
    possible_weights = [1, -1]

    psi_up = []

    for delta in possible_deltas:
        temp_array = []

        for weights in list(itertools.product(possible_weights, repeat=N)):
            if np.dot(weights, np.transpose(patterns[mu])) == delta:
                temp_array.append(np.dot(weights, np.transpose(q_up)[mu])) 

        
        psi_up.append(max(temp_array))

    # NORMALIZE !!! sottrarre il max del singolo messaggio indipendentemente dagli altri 

    return psi_up

def update_q_up(i, mu, weight_i, q_down): # from 1505.05401v2
    temp_sum = 0

    for mu_prime in range(M):
        if mu_prime != mu: ## sommare tutti e poi rimuoverne uno
            temp_sum += q_down[mu_prime][i](weight_i) # FIX DEPENDENCE ON WEIGHT_i

    # NORMALIZE !!!

    return temp_sum

def forward_pass(binary_patterns, weights, messages):
    q_up, q_down, psi_up, psi_down, phi_up, phi_down = messages["q_up"], messages["q_down"], messages["psi_up"], messages["psi_down"], messages["phi_up"], messages["phi_down"]

    # update q_up
    for i in range(N):
        for mu in range(M):
            q_up[i][mu] = update_q_up(i, mu, weights[i], q_down)
    messages["q_up"] = q_up
    
    # update psi_up
    for mu in range(M):
        psi_up[mu] = update_psi_up(mu, binary_patterns, q_up)
    messages["psi_up"] = psi_up

    # update phi_up
    phi_up = update_phi_up(psi_up)
    messages["phi_up"] = phi_up

    return messages
    
# ------------------------------------------------------------------
# CONVERGENCE ITERATIONS

def check_convergence():
    # WRITE LOGIC
    # se rimaNE STABILE d
    return False

def converge(binary_patterns, weights, messages):
    for i in range(ITERATIONS):
        # forward pass
        messages = forward_pass(binary_patterns, weights, messages)
        # backward pass
        messages = backward_pass(binary_patterns, weights, messages)
        # update weights
        weights = update_weights(weights, messages)
        # check if increment in delta_star is relevant
        if check_convergence():
            return True

    return False

# ------------------------------------------------------------------
# INITIALIZATION

def initialize_messages():
    messages = {}

    messages["q_up"] = np.zeros((N, M))
    messages["q_down"] = np.zeros((M, N))

    messages["psi_up"] = np.zeros((M, N+1))
    messages["psi_down"] = np.zeros((M, N+1))

    messages["phi_up"] = np.zeros(N+1)
    messages["phi_down"] = np.zeros(N+1)

    return messages       

def initialize_weights():
    return np.random.choice([-1, 1], size=(1, N))

def generate_patterns():
    return np.random.choice([-1, 1], size=(M, N))

# ------------------------------------------------------------------
def main(args):
    binary_patterns = generate_patterns(M, N)

    weights = initialize_weights(N)
    messages = initialize_messages(M, N)

    if converge(binary_patterns, weights, messages):
        print("Has converged :)")
    else:
        print("Has not converged :(")

if __name__ == "__main__":
    # M = 5, N = 3
    # args = []
    # main(args)

    # # TEST FOR INITIALIZATION
    # print(generate_patterns())
    # print(initialize_weights())
    # print(initialize_messages())

    # # TEST FOR FUNCTIONS IN FORWARD PASS
    # binary_patterns = generate_patterns()
    # weights = initialize_weights()
    # messages = initialize_messages()
    # forward_pass(binary_patterns, weights, messages)

    # # # update_q_up
    # q_down = np.zeros((M, N))
    # q_up = update_q_up(0, 0, 1, q_down)

    # # update_psi_up
    # weights = [1, -1, 1, -1, 1]
    # patterns = [[-1, 1, -1, 1, -1],
    #             [1, -1, 1, -1, 1],
    #             [-1, 1, -1, 1, -1]]
    # q_up = np.zeros((N, M))
    # # print(update_psi_up(0, patterns, q_up))

    # # # update_phi_up
    # psi_up = []
    # for i in range(M):
    #     psi_up.append(update_psi_up(i, patterns, q_up))
    # print(psi_up)
    # print(update_phi_up(psi_up))

    # # TEST FOR FUNCTIONS IN BACKWARD PASS
    binary_patterns = generate_patterns()
    weights = initialize_weights()
    messages = initialize_messages()
    backward_pass(binary_patterns, weights, messages)

    # # update_phi_down
    # print(update_phi_down(1))
    
    # # update_psi_down
    # mu = 0
    # delta_mu = 3
    # psi_up = [1, 2, 3]
    # phi_down = 1
    # print(update_psi_down(mu, delta_mu, psi_up, phi_down))

    # # update_q_down
    # mu = 0
    # i = 0
    # patterns = [[-1, 1, -1, 1, -1],
    #             [1, -1, 1, -1, 1],
    #             [-1, 1, -1, 1, -1]]
    # q_up = [[11, 12, 13], 
    #           [21, 22, 23],
    #           [31, 32, 33],
    #           [41, 42, 43],
    #           [51, 52, 53]]
    # psi_down = [1, 2, 3]
    # print(update_q_down(mu, i, patterns, q_up, psi_down))

    # # TEST FOR WEIGHT UPDATES
    # weights = [1, -1, 1, -1, 1]
    # q_down = np.transpose([[11, 12, 13], 
    #           [21, 22, 23],
    #           [31, 32, 33],
    #           [41, 42, 43],
    #           [51, 52, 53]])
    # print(update_weights(weights, q_down))
    pass