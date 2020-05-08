import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import itertools as it

from AC_modules import Layers, Networks

class StateEncoder(nn.Module):
    def __init__(self, map_size, in_channels, n_features):
        super(StateEncoder, self).__init__()
        self.net = Networks.OheNet(map_size, in_channels, n_features=n_features)
        
    def forward(self, x):
        """
        in = (batch, in_channels, map_size+2, map_size+2) or (in_channels, map_size+2, map_size+2)
        out = (batch, n_features) or (1, n_features) if batch dim not present.
        """
        return self.net(x)

class ActionEncoder(nn.Module):
    def __init__(self, n_actions, action_features, hidden_dim=256):
        super(ActionEncoder, self).__init__()
        self.embed = nn.Embedding(n_actions, action_features)
        self.residual = Layers.ResidualLayer(action_features, hidden_dim)
        
    def forward(self, a):
        """
        in = (batch)
        out = (batch, action_features)
        """
        x = self.embed(a)
        return self.residual(x)

class Decoder(nn.Module):
    def __init__(self, n_features, action_features, state_channels, state_resolution, n_possible_rewards):
        super(Decoder, self).__init__()
        
        self.state_decoder = Networks.SpatialNet(n_features+action_features, state_resolution, state_channels)
        self.action_decoder = nn.Sequential(
                                nn.Linear(n_features+action_features, 256),
                                nn.ReLU(),
                                nn.Linear(256, 64),
                                nn.ReLU(),
                                nn.Linear(64, n_possible_rewards))
        
    def forward(self, state_repr, action_repr):
        """
        In:
          state_repr = (batch, n_features)
          action_repr = (batch, action_features)
        Out:
          next_state = (batch, state_channels, state_resolution, state_resolution) - RECONSTRUCTION TASK
          reward_log_probs = (batch, n_possible_rewards) - CLASSIFICATION TASK
        """
        x = torch.cat([state_repr, action_repr], axis=1)
        next_state = self.state_decoder(x)
        reward_log_probs = F.log_softmax(self.action_decoder(x), dim=-1)
        return next_state, reward_log_probs

class SharedActor(nn.Module):
    def __init__(self, action_space, n_features):
        super(SharedActor, self).__init__()
        self.linear = nn.Sequential(
                        nn.Linear(n_features, 256),
                        nn.ReLU(),
                        nn.Linear(256, action_space))

    def forward(self, shared_repr):
        log_probs = F.log_softmax(self.linear(shared_repr), dim=1)
        return log_probs
    
class SharedCritic(nn.Module):
    def __init__(self, n_features):
        super(SharedCritic, self).__init__()
        self.net = nn.Sequential(
                        nn.Linear(n_features, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1))

    def forward(self, shared_repr):
        V = self.net(shared_repr)
        return V

class SupervisedAC(nn.Module):
    def __init__(self, n_actions, n_features, state_resolution, in_channels, n_possible_rewards, action_features):
        super(SupervisedAC, self).__init__()
        self.state_encoder = StateEncoder(state_resolution-2, in_channels, n_features)
        self.action_encoder = ActionEncoder(n_actions, action_features)
        self.decoder = Decoder(n_features, action_features, in_channels, state_resolution, n_possible_rewards)
        self.shared_actor = SharedActor(n_actions, n_features)
        self.shared_critic = SharedCritic(n_features)
        
    def supervised_params(self):
        params_iter = it.chain(self.state_encoder.parameters(), self.action_encoder.parameters(), self.decoder.parameters())
        return params_iter
    
    def RL_params(self):
        params_iter = it.chain(self.shared_actor.parameters(), self.shared_critic.parameters())
        return params_iter
            
    def encode_state(self, state):
        return self.state_encoder(state)
    
    def predict_state_reward(self, states, actions):
        state_repr = self.state_encoder(states)
        action_repr = self.action_encoder(actions)
        next_state, reward_log_probs = self.decoder(state_repr, action_repr)
        return next_state, reward_log_probs
    
    def pi(self, x, full_pass=True):
        if full_pass:
            with torch.no_grad():
                state_repr = self.encode_state(x)
        else:
            state_repr = x
            
        log_probs = self.shared_actor(state_repr)
        return log_probs
    
    def V_critic(self, x, full_pass=True):
        if full_pass:
            with torch.no_grad():
                state_repr = self.encode_state(x)
        else:
            state_repr = x
            
        V = self.shared_critic(state_repr)
        return V
        
        
        
        
        
        
        
        