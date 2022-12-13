################################################################################
#                           1 Import packages                                  #
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
# Use a double ended queue (deque) for memory
# When memory is full, this will replace the oldest value with the new one
from collections import deque
# Supress all warnings (e.g. deprecation warnings) for regular use
import warnings
warnings.filterwarnings("ignore")

################################################################################
#                           Model parameters for all DDQ                       #
################################################################################

# Set whether to display on screen (slows model)
DISPLAY_ON_SCREEN = False
# Discount rate of future rewards
GAMMA = 0.95
# Learing rate for neural network
LEARNING_RATE = 0.0003
# Maximum number of game steps (state, action, reward, next state) to keep
MEMORY_SIZE = 1500000
# Sample batch size for policy network update
BATCH_SIZE = 3

# Number of game steps to play before starting training (all random actions)
REPLAY_START_SIZE = 10000
# Time step between actions
TIME_STEP = 1
# Number of steps between policy -> target network update
SYNC_TARGET_STEPS = 365

# Exploration rate (episolon) is probability of choosign a random action
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
# Reduction in epsilon with each game step
EXPLORATION_DECAY = 0.999

# Simulation duration
#SIM_DURATION = 365
SIM_DURATION = 20000

# Training episodes
TRAINING_EPISODES = 5


################################################################################
#                    4 Define policy net training function                     #
################################################################################

def optimize(policy_net, target_net, memory):
    """
    Update  model by sampling from memory.
    Uses policy network to predict best action (best Q).
    Uses target network to provide target of Q for the selected next action.
    """
      
    # Do not try to train model if memory is less than reqired batch size
    if len(memory) < BATCH_SIZE:
        return    
 
    # Reduce exploration rate (exploration rate is stored in policy net)
    policy_net.exploration_rate *= EXPLORATION_DECAY
    policy_net.exploration_rate = max(EXPLORATION_MIN, 
                                      policy_net.exploration_rate)
    # Sample a random batch from memory
    batch = random.sample(memory, BATCH_SIZE)
    for state, action, reward, state_next, terminal in batch:
        state_action_values = policy_net(torch.FloatTensor(state))
        
        # Get target Q for policy net update
       
        if not terminal:
            # For non-terminal actions get Q from policy net
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach next state values from gradients to prevent updates
            expected_state_action_values = expected_state_action_values.detach()
            # Get next state action with best Q from the policy net (double DQN)
            policy_next_state_values = policy_net(torch.FloatTensor(state_next))
            policy_next_state_values = policy_next_state_values.detach()
            best_action = np.argmax(policy_next_state_values[0].numpy())
            # Get target net next state
            next_state_action_values = target_net(torch.FloatTensor(state_next))
            # Use detach again to prevent target net gradients being updated
            next_state_action_values = next_state_action_values.detach()
            best_next_q = next_state_action_values[0][best_action].numpy()
            updated_q = reward + (GAMMA * best_next_q)      
            expected_state_action_values[0][action] = updated_q
        else:
            # For termal actions Q = reward (-1)
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach values from gradients to prevent gradient update
            expected_state_action_values = expected_state_action_values.detach()
            # Set Q for all actions to reward (-1)
            expected_state_action_values[0] = reward
 
        # Set net to training mode
        policy_net.train()
        # Reset net gradients
        policy_net.optimizer.zero_grad()  
        # calculate loss
        loss_v = nn.MSELoss()(state_action_values, expected_state_action_values)
        # Backpropogate loss
        loss_v.backward()
        # Update network gradients
        policy_net.optimizer.step()  

    return

################################################################################
#                            5 Define memory class                             #
################################################################################

class Memory():
    """
    Replay memory used to train model.
    Limited length memory (using deque, double ended queue from collections).
      - When memory full deque replaces oldest data with newest.
    Holds, state, action, reward, next state, and episode done.
    """
    
    def __init__(self):
        """Constructor method to initialise replay memory"""
        self.memory = deque(maxlen=MEMORY_SIZE)

    def remember(self, state, action, reward, next_state, done):
        """state/action/reward/next_state/done"""
        self.memory.append((state, action, reward, next_state, done))

################################################################################
#                       6  Define results plotting function                    #
################################################################################

def plot_results(run, exploration, score, next_maint_t, wear_accum_t):
    """Plot and report results at end of run"""
    
    # Get beds and patirents from run_detals DataFrame
    # next_maint = run_details['next_maint']
    # conv_wear = run_details['conv_wear']
    #last_fail = run_details['last_fail']
    
    # Set up chart (ax1 and ax2 share x-axis to combine two plots on one graph)
    fig = plt.figure(figsize=(9,5))
    ax1 = fig.add_subplot(121)
    ax2 = ax1.twinx()
    
    # Plot results
    average_rewards = np.array(score)/SIM_DURATION
    ax1.plot(run, exploration, label='exploration', color='g')
    ax2.plot(run, average_rewards, label='average reward', color='r')
    
    # Set axes
    ax1.set_xlabel('run')
    ax1.set_ylabel('exploration', color='g')
    ax2.set_ylabel('average reward', color='r')
    
    # Show last run tracker of beds and patients

    ax3 = fig.add_subplot(122)
    step = np.arange(len(next_maint_t))
    ax3.plot(step, next_maint_t, label='next_maint', color='g')
    #ax3.plot(step, wear_accum_t, label='wear_accum', color='r')

    # day = np.arange(len(next_maint))*TIME_STEP
    # ax3.plot(day, next_maint, label='next_maint', color='g')
    # ax3.plot(day, conv_wear, label='conv_wear', color='r')
    #ax3.plot(day, last_fail, label='last_fail', color='b')
    
    # Set axes
    ax3.set_xlabel('run')
    ax3.set_ylabel('main/wear')
    ax3.set_ylim(0)
    ax3.legend()
    ax3.grid()
    # Show
    
    plt.tight_layout(pad=2)
    plt.show()
    
    # Calculate summary results
    # results = pd.Series()
    # beds = np.array(beds)
    # patients = np.array(patients)
    # results['days under capacity'] = np.sum(patients > beds)
    # results['days over capacity'] = np.sum(beds > patients)
    # results['average patients'] = np.round(np.mean(patients), 0)
    # results['average beds'] = np.round(np.mean(beds), 0)
    # results['% occupancy'] = np.round((patients.sum() / beds.sum() * 100), 1)
    # print (results);