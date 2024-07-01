import numpy as np
import random
from scipy.stats import multivariate_normal

# Parameters
alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
epsilon = 0.1 # exploration-exploitation trade-off
num_episodes = 1000
n_steps = 100

# Environment
n, m = 5, 5 # dimensions of the grid

# Mean and covariance matrix for the 2D normal distribution
mean = [n//2, m//2]
cov = [[1.0, 0.0], [0.0, 1.0]]

# Generate a 2D normal distribution
x, y = np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, m-1, m))
pos = np.dstack((x, y))
rv = multivariate_normal(mean, cov)
grid = rv.pdf(pos)

# Normalize the grid so that values are between 0 and 1
grid = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))

# Actions
actions = ['up', 'down', 'left', 'right']

# Q-table
Q = np.zeros((n, m, len(actions)))

# Function to get next state
def get_next_state(x, y, action):
    if action == 'up':
        return max(0, x-1), y
    elif action == 'down':
        return min(n-1, x+1), y
    elif action == 'left':
        return x, max(0, y-1)
    elif action == 'right':
        return x, min(m-1, y+1)

# Function to choose an action using epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(Q[state[0], state[1], :])]

# Q-learning algorithm
for episode in range(num_episodes):
    x, y = np.random.randint(n), np.random.randint(m) # start from a random position
    total_reward = 0

    for step in range(n_steps):
        action = choose_action((x, y))
        next_x, next_y = get_next_state(x, y, action)
        
        reward = grid[next_x, next_y]
        total_reward += reward * (gamma ** step)
        
        next_action = choose_action((next_x, next_y))
        
        Q[x, y, actions.index(action)] += alpha * (reward + gamma * np.max(Q[next_x, next_y, :]) - Q[x, y, actions.index(action)])
        
        x, y = next_x, next_y
        
        if step == n_steps - 1:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Learned policy
policy = np.zeros((n, m), dtype=str)
for i in range(n):
    for j in range(m):
        policy[i, j] = actions[np.argmax(Q[i, j, :])]

print("The grid")
print(grid)
print("Learned Policy:")
print(policy)
