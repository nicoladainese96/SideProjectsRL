import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from AC_modules.ArchitectureBlocks import *

debug = False

class SupervisedActorCritic(nn.Module):
    def __init__(self, action_space, n_features, gamma=0.99, 
                 tau = 1., H=1e-3, n_steps = 1, device='cpu',**HPs):
        super(SupervisedActorCritic, self).__init__()
        
        # Parameters
        self.gamma = gamma
        self.n_actions = action_space
        self.tau = tau
        self.H = H
        self.n_steps = n_steps
        
        self.AC = SupervisedAC(action_space, n_features, **HPs)
        self.device = device 
        self.AC.to(self.device) 
        
        self.state_loss_fn = nn.MSELoss()
        self.reward_loss_fn = nn.NLLLoss()
        
        if debug:
            print("="*10 +" A2C HyperParameters "+"="*10)
            print("Discount factor: ", self.gamma)
            print("Action space: ", self.n_actions)
            print("Update critic target factor: ", self.tau)
            print("n_steps for TD: ", self.n_steps)
            print("Device used: ", self.device)
            print("\n\n"+"="*10 +" A2C Architecture "+"="*10)
            print("Actor Critic architecture: \n", self.AC)

    def get_action(self, state, return_log=False):
        state = torch.from_numpy(state).float().to(self.device)
        log_probs = self.AC.pi(state)
        probs = torch.exp(log_probs)
        action = Categorical(probs).sample().item()
        if return_log:
            return action, log_probs.view(-1)[action], probs
        else:
            return action
        
    def compute_supervised_loss(self, rewards, actions, states):
        rewards = torch.LongTensor(rewards).to(self.device)
        old_states = torch.tensor(states[:-1]).float().to(self.device)
        new_states = torch.tensor(states[1:]).float().to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        predicted_states, rewards_log_probs = self.AC.predict_state_reward(old_states, actions)
        state_loss = self.state_loss_fn(predicted_states, new_states)
        reward_loss = self.reward_loss_fn(rewards_log_probs, rewards)
        reconstruction_loss = state_loss + reward_loss
        return reconstruction_loss, state_loss.item(), reward_loss.item()
                                
    def compute_ac_loss(self, rewards, log_probs, distributions, states, done, bootstrap=None): 
        ### Compute n-steps rewards, states, discount factors and done mask ###
        
        n_step_rewards = self.compute_n_step_rewards(rewards)
        if debug:
            print("n_step_rewards.shape: ", n_step_rewards.shape)
            print("rewards.shape: ", rewards.shape)
            print("n_step_rewards: ", n_step_rewards)
            print("rewards: ", rewards)
            print("bootstrap: ", bootstrap)
                
        if bootstrap is not None:
            done[bootstrap] = False 
        if debug:
            print("done.shape: (before n_steps)", done.shape)
            print("done: (before n_steps)", done)
        
        old_states = torch.tensor(states[:-1]).float().to(self.device)

        new_states, Gamma_V, done = self.compute_n_step_states(states, done)
        new_states = torch.tensor(new_states).float().to(self.device)

        if debug:
            print("done.shape: (after n_steps)", done.shape)
            print("Gamma_V.shape: ", Gamma_V.shape)
            print("done: (after n_steps)", done)
            print("Gamma_V: ", Gamma_V)
            print("old_states.shape: ", old_states.shape)
            print("new_states.shape: ", new_states.shape)
            
        ### Wrap variables into tensors ###
        
        done = torch.LongTensor(done.astype(int)).to(self.device)
        if debug: print("log_probs: ", log_probs)
            
        log_probs = torch.stack(log_probs).to(self.device)
        if debug: print("log_probs: ", log_probs)
            
        distributions = torch.stack(distributions, axis=0).to(self.device)
        mask = (distributions == 0).nonzero()
        distributions[mask[:,0], mask[:,1]] = 1e-5
        if debug: print("distributions: ", distributions)
            
        n_step_rewards = torch.tensor(n_step_rewards).float().to(self.device)
        Gamma_V = torch.tensor(Gamma_V).float().to(self.device)
        
        ### Update critic and then actor ###
        critic_loss = self.compute_critic_loss(n_step_rewards, new_states, old_states, done, Gamma_V)

        actor_loss, entropy = self.compute_actor_loss(n_step_rewards, log_probs, distributions, 
                                                       new_states, old_states, done, Gamma_V)

        return critic_loss, actor_loss, entropy
    
    def compute_critic_loss(self, n_step_rewards, new_states, old_states, done, Gamma_V):
        
        # Compute loss 
        if debug: print("Updating critic...")
        with torch.no_grad():
            V_trg = self.AC.V_critic(new_states).squeeze()
            if debug:
                print("V_trg.shape (after critic): ", V_trg.shape)
            V_trg = (1-done)*Gamma_V*V_trg + n_step_rewards
            if debug:
                print("V_trg.shape (after sum): ", V_trg.shape)
            V_trg = V_trg.squeeze()
            if debug:
                print("V_trg.shape (after squeeze): ", V_trg.shape)
                print("V_trg.shape (after squeeze): ", V_trg)
            
        V = self.AC.V_critic(old_states).squeeze()
        if debug: 
            print("V.shape: ",  V.shape)
            print("V: ",  V)
        loss = F.mse_loss(V, V_trg)

        return loss
    
    def compute_actor_loss(self, n_step_rewards, log_probs, distributions, new_states, old_states, done, Gamma_V):
        
        # Compute gradient 
        if debug: print("Updating actor...")
        with torch.no_grad():
            V_pred = self.AC.V_critic(old_states).squeeze()
            V_trg = (1-done)*Gamma_V*self.AC.V_critic(new_states).squeeze()  + n_step_rewards
        
        A = V_trg - V_pred
        policy_gradient = - log_probs*A
        if debug:
            print("V_trg.shape: ",V_trg.shape)
            print("V_trg: ", V_trg)
            print("V_pred.shape: ",V_pred.shape)
            print("V_pred: ", V_pred)
            print("A.shape: ", A.shape)
            print("A: ", A)
            print("policy_gradient.shape: ", policy_gradient.shape)
            print("policy_gradient: ", policy_gradient)
        policy_grad = torch.mean(policy_gradient)
        if debug: print("policy_grad: ", policy_grad)
            
        # Compute negative entropy (no - in front)
        entropy = torch.mean(distributions*torch.log(distributions))
        if debug: print("Negative entropy: ", entropy)
        
        loss = policy_grad + self.H*entropy
        if debug: print("Actor loss: ", loss)
             
        return loss, entropy
                
    def compute_n_step_rewards(self, rewards):
        """
        Computes n-steps discounted reward padding with zeros the last elements of the trajectory.
        This means that the rewards considered are AT MOST n, but can be less for the last n-1 elements.
        """
        T = len(rewards)
        
        # concatenate n_steps zeros to the rewards -> they do not change the cumsum
        r = np.concatenate((rewards,[0 for _ in range(self.n_steps)])) 
        
        Gamma = np.array([self.gamma**i for i in range(r.shape[0])])
        
        # reverse everything to use cumsum in right order, then reverse again
        Gt = np.cumsum(r[::-1]*Gamma[::-1])[::-1]
        
        G_nstep = Gt[:T] - Gt[self.n_steps:] # compute n-steps discounted return
        
        Gamma = Gamma[:T]
        
        assert len(G_nstep) == T, "Something went wrong computing n-steps reward"
        
        n_steps_r = G_nstep / Gamma
        
        return n_steps_r
    
    def compute_n_step_states(self, states, done):
        """
        Computes n-steps target states (to be used by the critic as target values together with the
        n-steps discounted reward). For last n-1 elements the target state is the last one available.
        Adjusts also the `done` mask used for disabling the bootstrapping in the case of terminal states
        and returns Gamma_V, that are the discount factors for the target state-values, since they are 
        n-steps away (except for the last n-1 states, whose discount is adjusted accordingly).
        
        Return
        ------
        new_states, Gamma_V, done: arrays with first dimension = len(states)-1
        """
        
        # Compute indexes for (at most) n-step away states 
        
        n_step_idx = np.arange(len(states)-1) + self.n_steps
        diff = n_step_idx - len(states) + 1
        mask = (diff > 0)
        n_step_idx[mask] = len(states) - 1
        
        # Compute new states
        
        new_states = states[n_step_idx]
        
        # Compute discount factors
        
        pw = np.array([self.n_steps for _ in range(len(new_states))])
        pw[mask] = self.n_steps - diff[mask]
        Gamma_V = self.gamma**pw
        
        # Adjust done mask
        
        mask = (diff >= 0)
        done[mask] = done[-1]
        
        return new_states, Gamma_V, done
            
