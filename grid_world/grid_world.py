import numpy as np
import matplotlib.pyplot as plt
num_rows = 4
num_cols = 4
value_function = np.zeros((num_rows, num_cols))

# action [up, down, left, right]
policy = np.ones((num_rows, num_cols, 4)) * 0.25

#set terminal states
policy[0][0] = [0, 0, 0, 0] 
policy[-1][-1] = [0, 0, 0, 0]

def next_position(y, x, action):
    if action == 0: # up
        return max(y - 1, 0), x
    elif action == 1: # down
        return min(y + 1, num_rows - 1), x
    elif action == 2: # left
        return y, max(x - 1, 0)
    elif action == 3: # right
        return y, min(x + 1, num_cols - 1)
    
def reward(y, x, action):
    if y == 0 and x == 0:
        return 0
    elif y == num_rows - 1 and x == num_cols - 1:
        return 0
    else:
        return -1
    
def evaluate_policy(steps=5):
    global value_function
    for _ in range(steps):
        new_value_function = np.copy(value_function)
        for y in range(num_rows):
            for x in range(num_cols):
                for action in range(4):
                    y_next, x_next = next_position(y, x, action)
                    new_value_function[y, x] += policy[y][x][int(action)] * (reward(y, x, action) + value_function[y_next, x_next])
        value_function = new_value_function

def update_policy():
    global policy
    for y in range(num_rows):
        for x in range(num_cols):
            best_actions = [reward(y, x, action) + value_function[next_position(y, x, action)] for action in range(4)]
            # set the max values to 1 and the rest to 0
            max_value = max(best_actions)
            # get the index of all of the max values
            index = [i for i, j in enumerate(best_actions) if j == max_value]
            best_actions = np.zeros(4)
            best_actions[index] = 1
            
            policy[y][x] = best_actions/ np.sum(best_actions)


def plot_policy():
    """
    Function to plot a policy grid.
    
    Args:
        policy (numpy.ndarray): A 3D numpy array where the third dimension has four elements
                                corresponding to the four possible actions (up, down, left, right).
    """
    # Define the arrow directions
    arrow_directions = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    # Get the grid size
    n_rows, n_cols, _ = policy.shape
    
    # Create a plot
    plt.figure(figsize=(n_cols, n_rows))
    ax = plt.gca()
    
    # Iterate through each cell in the grid
    for row in range(n_rows):
        for col in range(n_cols):
            if row == 0 and col == 0:
                continue
            if row == n_rows - 1 and col == n_cols - 1:
                continue
            if policy[row, col, 0] != 0.0:
                ax.text(col, row, arrow_directions[0], ha='center', va='center', fontsize=12)
            if policy[row, col, 1] != 0.0:
                ax.text(col, row, arrow_directions[1], ha='center', va='center', fontsize=12)
            if policy[row, col, 2] != 0.0:
                ax.text(col, row, arrow_directions[2], ha='center', va='center', fontsize=12)
            if policy[row, col, 3] != 0.0:
                ax.text(col, row, arrow_directions[3], ha='center', va='center', fontsize=12)

    # set the terminating state to a gray color solid block
    ax.text(0, 0, 'X', ha='center', va='center', fontsize=12, color='gray')
    ax.text(n_cols - 1, n_rows - 1, 'X', ha='center', va='center', fontsize=12, color='gray')
    # Set grid
    ax.set_xticks(np.arange(-0.5, n_cols, 1))
    ax.set_yticks(np.arange(-0.5, n_rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="both")
    plt.gca().invert_yaxis()

    # Show plot
    plt.show()

def plot_value_function():
    plt.imshow(value_function, cmap='hot', interpolation='nearest')
    plt.show()

for _ in range(10):
    evaluate_policy()
    update_policy()
plot_value_function()
plot_policy()


