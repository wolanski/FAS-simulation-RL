################################################################################
#                           1 Import packages                                  #
################################################################################

import numpy as np
import random
import torch
import torch.nn as nn
# Supress all warnings (e.g. deprecation warnings) for regular use
import warnings
import modules.rl_agents.common as cm
warnings.filterwarnings("ignore")

################################################################################
#                      3 Define DQN (Deep Q Network) class                     #
#                    (Used for both policy and target nets)                    #
################################################################################

class DQN(nn.Module):

    """Deep Q Network. Udes for both policy (action) and target (Q) networks."""

    def __init__(self, observation_space, action_space, neurons_per_layer=48):
        """Constructor method. Set up neural nets."""

        # Set starting exploration rate
        self.exploration_rate = cm.EXPLORATION_MAX
        
        # Set up action space (choice of possible actions)
        self.action_space = action_space
              
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, action_space)
            )
        
    def act(self, state):
        """Act either randomly or by redicting action that gives max Q"""
        
        # Act randomly if random number < exploration rate
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_space)
            
        else:
            # Otherwise get predicted Q values of actions
            q_values = self.net(torch.FloatTensor(state))
            # Get index of action with best Q
            action = np.argmax(q_values.detach().numpy()[0])
        
        return  action
   
    def forward(self, x):
        """Forward pass through network"""
        return self.net(x)
