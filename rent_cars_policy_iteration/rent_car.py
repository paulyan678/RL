import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
class rent_car_problem:
    def __init__(self, max_cars_a, max_cars_b, max_transfer, avg_rent_a, avg_return_a, avg_rent_b, avg_return_b, discount, transfer_reward, rent_reward):
        self.max_cars_a = max_cars_a
        self.max_cars_b = max_cars_b
        self.max_transfer = max_transfer
        self.avg_rent_a = avg_rent_a
        self.avg_return_a = avg_return_a
        self.avg_rent_b = avg_rent_b
        self.avg_return_b = avg_return_b
        self.discount = discount
        self.transfer_reward = transfer_reward
        self.rent_reward = rent_reward

        self.rent_a_dist = np.arange(0, max_cars_a+1)
        self.rent_b_dist = np.arange(0, max_cars_b+1)
        self.return_a_dist = np.arange(0, max_cars_a+1)
        self.return_b_dist = np.arange(0, max_cars_b+1)
        
        self.rent_a_dist = stats.poisson.pmf(self.rent_a_dist, avg_rent_a)
        self.rent_b_dist = stats.poisson.pmf(self.rent_b_dist, avg_rent_b)
        self.return_a_dist = stats.poisson.pmf(self.return_a_dist, avg_return_a)
        self.return_b_dist = stats.poisson.pmf(self.return_b_dist, avg_return_b)

        self.rent_a_dist[-1] += 1 - np.sum(self.rent_a_dist)
        self.rent_b_dist[-1] += 1 - np.sum(self.rent_b_dist)
        self.return_a_dist[-1] += 1 - np.sum(self.return_a_dist)
        self.return_b_dist[-1] += 1 - np.sum(self.return_b_dist)
        
        self.state_value = np.zeros((max_cars_a+1, max_cars_b+1))
        self.policy = np.zeros((max_cars_a+1, max_cars_b+1), dtype=np.int32)

        self.policy_stable = False
    

def action_reward(state, action, problem: rent_car_problem, state_value):
    reward = 0
    reward += problem.transfer_reward * np.abs(action)

    num_car_after_move_a = state[0] - action
    num_car_after_move_b = state[1] + action
    
    if num_car_after_move_a < 0 or num_car_after_move_b < 0 or num_car_after_move_a > problem.max_cars_a or num_car_after_move_b > problem.max_cars_b:
        return -np.inf
    
    num_empty_spot_after_move_a = problem.max_cars_a - num_car_after_move_a
    num_empty_spot_after_move_b = problem.max_cars_b - num_car_after_move_b

    last_return_a_prob = np.sum(problem.return_a_dist[num_empty_spot_after_move_a:])

    last_return_b_prob = np.sum(problem.return_b_dist[num_empty_spot_after_move_b:])
    

    for num_car_return_a in range(num_empty_spot_after_move_a+1):
        for num_car_return_b in range(num_empty_spot_after_move_b+1):
            num_car_after_return_a = num_car_after_move_a + num_car_return_a
            num_car_after_return_b = num_car_after_move_b + num_car_return_b
            
            last_rent_a_prob = np.sum(problem.rent_a_dist[num_car_after_return_a:])
            last_rent_b_prob = np.sum(problem.rent_b_dist[num_car_after_return_b:])

            for num_car_rent_a in range(num_car_after_return_a+1):
                for num_car_rent_b in range(num_car_after_return_b+1):
                    
                    prob = 1

                    if num_car_rent_a == num_car_after_return_a:
                        prob *= last_rent_a_prob
                    else:
                        prob *= problem.rent_a_dist[num_car_rent_a]
                    
                    if num_car_rent_b == num_car_after_return_b:
                        prob *= last_rent_b_prob
                    else:
                        prob *= problem.rent_b_dist[num_car_rent_b]

                    if num_car_return_a == num_empty_spot_after_move_a:
                        prob *= last_return_a_prob
                    else:
                        prob *= problem.return_a_dist[num_car_return_a]
                    
                    if num_car_return_b == num_empty_spot_after_move_b:
                        prob *= last_return_b_prob
                    else:
                        prob *= problem.return_b_dist[num_car_return_b]
                    
                    num_after_rent_a = num_car_after_return_a - num_car_rent_a
                    num_after_rent_b = num_car_after_return_b - num_car_rent_b

                    reward += prob * ((num_car_rent_a + num_car_rent_b)*problem.rent_reward + problem.discount * state_value[num_after_rent_a, num_after_rent_b])
    return reward

def eval_state_value(problem: rent_car_problem):
    for _ in range(1):
        old_state_value = np.copy(problem.state_value)
        for i in range(problem.max_cars_a+1):
            for j in range(problem.max_cars_b+1):
                state = [i, j]
                action = problem.policy[i, j]
                problem.state_value[i, j] = action_reward(state, action, problem, old_state_value)

def policy_improvement(problem: rent_car_problem):
    problem.policy_stable = True
    for i in range(problem.max_cars_a+1):
        for j in range(problem.max_cars_b+1):
            state = [i, j]
            old_action = problem.policy[i, j]
            action_list = np.arange(-problem.max_transfer, problem.max_transfer+1)
            action_reward_list = [action_reward(state, action, problem, problem.state_value) for action in action_list]
            new_action = action_list[np.argmax(action_reward_list)]
            if old_action != new_action:
                problem.policy_stable = False
                problem.policy[i, j] = new_action

def policy_iteration(problem: rent_car_problem):
    i = 0
    while not problem.policy_stable:
        eval_state_value(problem)
        
        # plt.imshow(problem.state_value)
        # plt.colorbar()
        # plt.show()
        print("state value")
        
        policy_improvement(problem)
        
        plt.imsave(f"policy{i}.png", problem.policy)
        i += 1
    return problem.policy

# test the action reward function
problem = rent_car_problem(10, 10, 2, 1, 1, 1, 1, 0.9, -3, 10)
# print(action_reward([2, 2], 1, problem))
opt_policy = policy_iteration(problem)
