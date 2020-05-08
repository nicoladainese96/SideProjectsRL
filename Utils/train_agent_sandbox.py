import numpy as np
import torch
from Utils import test_env
import time

debug = False

def play_episode(agent, env, max_steps):

    # Start the episode
    state = env.reset()
    if debug: print("state.shape: ", state.shape)
    rewards = []
    log_probs = []
    distributions = []
    states = [state]
    done = []
    bootstrap = []
    actions = []
        
    steps = 0
    while True:
     
        action, log_prob, distrib = agent.get_action(state, return_log = True)
        actions.append(action)
        new_state, reward, terminal, info = env.step(action)
        if debug: print("state.shape: ", new_state.shape)
        rewards.append(reward)
        log_probs.append(log_prob)
        distributions.append(distrib)
        states.append(new_state)
        done.append(terminal)
        
        # Still unclear how to retrieve max steps from the game itself
        if terminal is True and steps == max_steps:
            bootstrap.append(True)
        else:
            bootstrap.append(False) 
        
        if terminal is True:
            #print("steps: ", steps)
            #print("Bootstrap needed: ", bootstrap[-1])
            break
            
        state = new_state
        steps += 1
        
    rewards = np.array(rewards)
    states = np.array(states)
    if debug: print("states.shape: ", states.shape)
    done = np.array(done)
    bootstrap = np.array(bootstrap)
    actions = np.array(actions)

    return rewards, log_probs, distributions, np.array(states), done, bootstrap, actions

def random_start(X=10, Y=10):
    s1, s2 = np.random.choice(X*Y, 2, replace=False)
    initial = [s1//X, s1%X]
    goal = [s2//X, s2%X]
    return initial, goal

def supervised_update(agent, optimizer, rewards, actions, states):
    reconstruction_loss, state_loss, reward_loss = agent.compute_supervised_loss(rewards, actions, states)
    optimizer.zero_grad()
    reconstruction_loss.backward()
    optimizer.step()
    return state_loss, reward_loss

def RL_update(agent, optimizer, rewards, log_probs, distributions, states, done, bootstrap):
    critic_loss, actor_loss, entropy = agent.compute_ac_loss(rewards, log_probs, distributions, states, done, bootstrap)
    loss = critic_loss + actor_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return critic_loss.item(), actor_loss.item(), entropy.item()
    
def train_sandbox(agent, game_params, supervised_lr, RL_lr, n_episodes = 1000, 
                  max_steps=120, return_agent=False, random_init=True):
    performance = []
    steps_to_solve = []
    time_profile = []
    critic_losses = [] 
    actor_losses = []
    entropies = []
    state_losses = []
    reward_losses = []
    
    supervised_optimizer = torch.optim.Adam(agent.AC.supervised_params(), lr=supervised_lr)
    RL_optimizer = torch.optim.Adam(agent.AC.RL_params(), lr=RL_lr)
    
    update_supervised = True
    for e in range(n_episodes):
        
        if random_init:
            # Change game params
            initial, goal = random_start(game_params["x"], game_params["y"])

            # All game parameters
            game_params["initial"] = initial
            game_params["goal"] = goal

        t0 = time.time()
        env = test_env.Sandbox(**game_params)
        rewards, log_probs, distributions, states, done, bootstrap, actions = play_episode(agent, env, max_steps)
        t1 = time.time()

        performance.append(np.sum(rewards))
        steps_to_solve.append(len(rewards))
        if (e+1)%10 == 0:
            print("Episode %d - reward: %.2f - steps to solve: %.2f"%(e+1, np.mean(performance[-10:]), np.mean(steps_to_solve[-10:])))
        
        if update_supervised:
            state_loss, reward_loss = supervised_update(agent, supervised_optimizer, rewards, actions, states)
            state_losses.append(state_loss)
            reward_losses.append(reward_loss)
            update_supervised = False
        else:
            critic_loss, actor_loss, entropy = RL_update(agent, RL_optimizer, rewards, log_probs, 
                                                         distributions, states, done, bootstrap)
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            entropies.append(entropy)
            update_supervised = True
    
        t2 = time.time()
        #print("Time updating the agent: %.2f s"%(t2-t1))
            
        time_profile.append([t1-t0, t2-t1])
        
    performance = np.array(performance)
    time_profile = np.array(time_profile)
    steps_to_solve = np.array(steps_to_solve)
    L = n_episodes // 6 # consider last sixth of episodes to compute agent's asymptotic performance
    losses = dict(critic_losses=critic_losses, actor_losses=actor_losses, entropies=entropies, 
                  state_loss=state_losses, reward_loss=reward_losses )
    if return_agent:
        return performance, performance[-L:].mean(), performance[-L:].std(), agent, time_profile, losses, steps_to_solve
    else:
        return performance, performance[-L:].mean(), performance[-L:].std(), losses, steps_to_solve
