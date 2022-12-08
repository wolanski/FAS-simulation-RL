################################################################################
#                           1 Import packages                                  #
################################################################################

import numpy as np
import random
import torch
import torch.nn as nn
import math
# Supress all warnings (e.g. deprecation warnings) for regular use
import warnings
import modules.rl_agents.common as cm
warnings.filterwarnings("ignore")


################################################################################
#                 3 Define DQN (Duelling Deep Q Network) class                 #
#                    (Used for both policy and target nets)                    #
################################################################################

"""
Code for nosiy layers comes from:

Lapan, M. (2020). Deep Reinforcement Learning Hands-On: Apply modern RL methods 
to practical problems of chatbots, robotics, discrete optimization, 
web automation, and more, 2nd Edition. Packt Publishing.
"""


class NoisyLinear(nn.Linear):
    """
    Noisy layer for network.
    
    For every weight in the layer we have a random value that we draw from the
    normal distribution.Paraemters for the noise, sigma, are stored within the
    layer and get trained as part of the standard back-propogation.
    
    'register_buffer' is used to create tensors in the network that are not
    updated during back-propogation. They are used to create normal 
    distributions to add noise (multiplied by sigma which is a paramater in the
    network).
    """
    
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + self.weight
        return F.linear(input, v, bias)
    

class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise. This reduces the number of
    random numbers to be sampled (so less computationally expensive). There are 
    two random vectors. One with the size of the input, and the other with the 
    size of the output. A random matrix is create by calculating the outer 
    product of the two vectors.
    
    'register_buffer' is used to create tensors in the network that are not
    updated during back-propogation. They are used to create normal 
    distributions to add noise (multiplied by sigma which is a paramater in the
    network).
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z1 = torch.zeros(1, in_features)
        self.register_buffer("epsilon_input", z1)
        z2 = torch.zeros(out_features, 1)
        self.register_buffer("epsilon_output", z2)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        v = self.weight + self.sigma_weight * noise_v
        return F.linear(input, v, bias)

class DQN(nn.Module):

    """Deep Q Network. Udes for both policy (action) and target (Q) networks."""

    def __init__(self, observation_space, action_space, neurons_per_layer=48):
        """Constructor method. Set up neural nets."""

        # Set starting exploration rate
        self.exploration_rate = cm.EXPLORATION_MAX
        
        # Set up action space (choice of possible actions)
        self.action_space = action_space
              
        
        # First layerswill be common to both Advantage and value
        super(DQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(observation_space, neurons_per_layer),
            nn.ReLU()            
            )
        
        # Advantage has same number of outputs as the action space
        self.advantage = nn.Sequential(
            NoisyFactorizedLinear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            NoisyFactorizedLinear(neurons_per_layer, action_space)
            )
        
        # State value has only one output (one value per state)
        self.value = nn.Sequential(
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, 1)
            )        
        
    def act(self, state):
        """Act either randomly or by redicting action that gives max Q"""
        
        # Act randomly if random number < exploration rate
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_space)
            
        else:
            # Otherwise get predicted Q values of actions
            q_values = self.forward(torch.FloatTensor(state))
            # Get index of action with best Q
            action = np.argmax(q_values.detach().numpy()[0])
        
        return  action
        
  
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        action_q = value + advantage - advantage.mean()
        return action_q