import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from scipy.stats import multivariate_normal

# Parameters
alpha = 0.001  # learning rate
gamma = 0.9    # discount factor
epsilon = 1.0  # exploration-exploitation trade-off (initially high)
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 1000
n_steps = 100
batch_size = 32
memory_size = 2000

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
num_actions = len(actions)

# Replay memory
memory = deque(maxlen=memory_size)

# Neural Network for Q-learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
def choose_action(state, policy_net):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return actions[torch.argmax(q_values).item()]

# Initialize the neural networks and optimizer
policy_net = DQN(n * m, num_actions)
target_net = DQN(n * m, num_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
criterion = nn.MSELoss()

# Function to optimize the model
def optimize_model():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(states)
    actions = torch.LongTensor([actions.index(a) for a in actions]).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    current_q_values = policy_net(states).gather(1, actions)
    max_next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
    
    loss = criterion(current_q_values, expected_q_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# DQN algorithm
for episode in range(num_episodes):
    x, y = np.random.randint(n), np.random.randint(m) # start from a random position
    total_reward = 0
    
    state = grid.flatten()
    
    for step in range(n_steps):
        action = choose_action(state, policy_net)
        next_x, next_y = get_next_state(x, y, action)
        
        reward = grid[next_x, next_y]
        total_reward += reward * (gamma ** step)
        
        next_state = grid.flatten()
        done = (step == n_steps - 1)
        
        memory.append((state, action, reward, next_state, done))
        
        state = next_state
        x, y = next_x, next_y
        
        optimize_model()
        
        if done:
            break
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Learned policy
policy = np.zeros((n, m), dtype=str)
for i in range(n):
    for j in range(m):
        state = grid.flatten()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        policy[i, j] = actions[torch.argmax(q_values).item()]

print("Learned Policy:")
print(policy)
