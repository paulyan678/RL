import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
# problem problem
a_cap = 4
b_cap = 4
transfer_reward = -2
rent_reward = 10
discount = 0.9
max_transfer = 5
a_lam_return = 3
b_lam_return = 3
a_lam_out = 3
b_lam_out = 3

def possion(lam, x):
    res = np.exp(-lam) * lam**x / sp.factorial(x)
    norm_res = res / np.sum(res)
    return norm_res

# array that represents the probability of returning and renting cars at locations
a_return_prob = possion(a_lam_return, np.arange(a_cap+1))
b_return_prob = possion(b_lam_return, np.arange(b_cap+1))
a_out_prob = possion(a_lam_out, np.arange(a_cap+1))
b_out_prob = possion(b_lam_out, np.arange(b_cap+1))

def comp_reward():
    '''
    Compute the reward matrix using the probability of returning and renting cars at locations
    return: reward matrix 3D of shape [a_cap+1, b_cap+1, max_transfer*2+1]
    '''
    global a_return_prob, b_return_prob, a_out_prob, b_out_prob, a_cap, b_cap, max_transfer, rent_reward, transfer_reward
    
    reward = np.zeros((a_cap+1, b_cap+1, max_transfer*2+1))
    
    for num_car_return_a in range(a_cap+1):
        for num_car_return_b in range(b_cap+1):
            for num_car_out_a in range(a_cap+1):
                for num_car_out_b in range(b_cap+1):
                    reward_a = np.minimum(np.arange(a_cap+1)[:, None] + num_car_return_a, num_car_out_a)
                    reward_b = np.minimum(np.arange(b_cap+1)[None, :] + num_car_return_b, num_car_out_b)
                    total_reward = (reward_a + reward_b) * rent_reward
                    
                    prob = (
                        a_return_prob[num_car_return_a]
                        * b_return_prob[num_car_return_b]
                        * a_out_prob[num_car_out_a]
                        * b_out_prob[num_car_out_b]
                    )
                    
                    reward[:, :, max_transfer] += prob * total_reward

    for num_transfer in range(1, max_transfer+1):
        reward[:, :, max_transfer+num_transfer] = reward[:, :, max_transfer] + transfer_reward * num_transfer
        reward[:, :, max_transfer-num_transfer] = reward[:, :, max_transfer] + transfer_reward * num_transfer
    
    return reward
    
    
def comp_reward1():
    '''
    Compute the reward matrix using the probability of returning and renting cars at locations
    return: reward matrix 3D of shape [a_cap+1, b_cap+1, max_transfer*2+1]
    '''
    global a_return_prob, b_return_prob, a_out_prob, b_out_prob
    reward = np.zeros((a_cap+1, b_cap+1, max_transfer*2+1))
    for a in range(a_cap+1):
        for b in range(b_cap+1):
            
            for num_car_return_a in range(a_cap+1):
                for num_car_return_b in range(b_cap+1):
                    for num_car_out_a in range(a_cap+1):
                        for num_car_out_b in range(b_cap+1):
                            reward_a = min(a + num_car_return_a, num_car_out_a)
                            reward_b = min(b + num_car_return_b, num_car_out_b)
                            reward[a, b, max_transfer] += a_return_prob[num_car_return_a] * b_return_prob[num_car_return_b] * a_out_prob[num_car_out_a] * b_out_prob[num_car_out_b] * (reward_a + reward_b)*rent_reward
    
    for num_transfer in range(1, max_transfer+1):
        reward[:, :, max_transfer+num_transfer] = reward[:, :, max_transfer] + transfer_reward * num_transfer
        reward[:, :, max_transfer-num_transfer] = reward[:, :, max_transfer] + transfer_reward * num_transfer
    return reward

vect_reward = comp_reward()
# not_vect_reward = comp_reward1()
# # print every ele side by side
# for i in range(a_cap+1):
#     for j in range(b_cap+1):
#         for k in range(max_transfer*2+1):
#             print(vect_reward[i, j, k], not_vect_reward[i, j, k])
# show the image made up of the reward without transfer
# plt.imshow(vect_reward[:, :, max_transfer])
# plt.show()
# plt.imshow(not_vect_reward[:, :, max_transfer])
# plt.show()

def comp_transition_prob():
    '''
    Compute the transition probability matrix using the probability of returning and renting cars at locations
    P[S_t+1 = (a2, b2) | S_t = (a1, b1), A_t = num_transfer]
    return: reward matrix %D of shape [a_cap+1, b_cap+1, a_cap+1, b_cap+1, max_transfer*2+1]
    '''
    global a_return_prob, b_return_prob, a_out_prob, b_out_prob, a_cap, b_cap, max_transfer, rent_reward, transfer_reward
    transition_prob = np.zeros((a_cap+1, b_cap+1, a_cap+1, b_cap+1, max_transfer*2+1))
    
    for a1 in range(a_cap+1):
        for b1 in range(b_cap+1):
            for a2 in range(a_cap+1):
                for b2 in range(b_cap+1):
                    for num_transfer in range(-max_transfer, max_transfer+1):
                        # compute the total probability that would give before transfer car number from a1 and b1
                        prob = 0
                        for num_car_return_a in range(a_cap+1):
                            for num_car_return_b in range(b_cap+1):
                                for num_car_out_a in range(a_cap+1):
                                    for num_car_out_b in range(b_cap+1):
                                        if a1 + num_car_return_a - num_car_out_a == a2 + num_transfer and b1 + num_car_return_b - num_car_out_b == b2 - num_transfer:
                                            prob += a_return_prob[num_car_return_a] * b_return_prob[num_car_return_b] * a_out_prob[num_car_out_a] * b_out_prob[num_car_out_b]
                        transition_prob[a1, b1, a2, b2, num_transfer+max_transfer] = prob
    return transition_prob




def comp_transition_prob_vec():
    '''
    Compute the transition probability matrix using the probability of returning and renting cars at locations
    P[S_t+1 = (a2, b2) | S_t = (a1, b1), A_t = num_transfer]
    return: reward matrix %D of shape [a_cap+1, b_cap+1, a_cap+1, b_cap+1, max_transfer*2+1]
    '''
    global a_return_prob, b_return_prob, a_out_prob, b_out_prob, a_cap, b_cap, max_transfer, rent_reward, transfer_reward
    transition_prob = np.zeros((a_cap+1, b_cap+1, a_cap+1, b_cap+1, max_transfer*2+1))
    
    a_range = np.arange(a_cap+1)
    b_range = np.arange(b_cap+1)
    
    a1, b1, a2, b2 = np.meshgrid(a_range, b_range, a_range, b_range, indexing='ij')
    
    for num_transfer in range(-max_transfer, max_transfer+1):
        before_transfer_a = a2 + num_transfer
        before_transfer_b = b2 - num_transfer
        
        for num_car_return_a in range(a_cap+1):
            for num_car_return_b in range(b_cap+1):
                for num_car_out_a in range(a_cap+1):
                    for num_car_out_b in range(b_cap+1):
                        valid_transition_a = a1 + num_car_return_a - num_car_out_a == before_transfer_a
                        valid_transition_b = b1 + num_car_return_b - num_car_out_b == before_transfer_b
                        
                        valid_transitions = valid_transition_a & valid_transition_b
                        
                        prob = (
                            a_return_prob[num_car_return_a]
                            * b_return_prob[num_car_return_b]
                            * a_out_prob[num_car_out_a]
                            * b_out_prob[num_car_out_b]
                        )
                        
                        transition_prob[..., num_transfer+max_transfer] += valid_transitions * prob
    
    return transition_prob

prob_transition = comp_transition_prob()
prob_transition_vec = comp_transition_prob_vec()

for a1 in range(a_cap+1):
    for b1 in range(b_cap+1):
        for a2 in range(a_cap+1):
            for b2 in range(b_cap+1):
                for num_transfer in range(1*max_transfer+1):
                    print(f"({a1}, {b1}) -> ({a2}, {b2}) under {num_transfer} transfers is iterative{prob_transition[a1,b1,a2,b2,num_transfer]} vectorized {prob_transition_vec[a1,b1,a2,b2,num_transfer]}")
                    if prob_transition[a1,b1,a2,b2,num_transfer] != prob_transition_vec[a1,b1,a2,b2,num_transfer]:
                        print("Error")
                        break

def policy_evaluation():
    pass

def policy_improvement():
    pass

def policy_iteration():
    pass

#run 
#plot


