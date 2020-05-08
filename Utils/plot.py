import numpy as np
import matplotlib.pyplot as plt

def plot_results(results, moving_average=True, average_window=100):
    
    score, asymptotic_score, asymptotic_err, trained_agent, time_profile, losses, steps_to_solve = results
    t_play = time_profile[:,0].mean()
    t_update = time_profile[:,1].mean()

    print("Average time for playing one episode: %.2f s"%t_play)
    print("Average time for updating the agent: %.2f s"%t_update)
    print("Asymptotic score: %.3f +/- %.3f"%(asymptotic_score, asymptotic_err))
    
    if moving_average:
        n_epochs = np.arange(average_window, len(score))
    else:
        n_epochs = np.arange(len(score))
        
    ### plot score ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(score[i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, score)
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.show()
    
    ### steps to solve ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(steps_to_solve[i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, steps_to_solve)
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Steps to solve", fontsize=16)
    plt.show()
    
    if moving_average:
        n_epochs = np.arange(average_window, len(losses['critic_losses']))
    else:
        n_epochs = np.arange(len(losses['critic_losses']))
        
    ### plot critic loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['critic_losses'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, losses['critic_losses'])

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Critic loss", fontsize=16)
    plt.show()

    ### plot actor loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['actor_losses'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, losses['actor_losses'])

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Actor loss", fontsize=16)
    plt.show()
    
    ### plot entropy ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['entropies'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, -average_score)
    else:
        plt.plot(n_epochs, -np.array(losses['entropies']))

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Entropy term", fontsize=16)
    plt.show()
    
    if moving_average:
        n_epochs = np.arange(average_window, len(losses['state_loss']))
    else:
        n_epochs = np.arange(len(losses['state_loss']))
        
    ### plot state loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['state_loss'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, np.array(losses['state_loss']))

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("State loss", fontsize=16)
    plt.show()
    
    ### plot reward loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['reward_loss'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, np.array(losses['reward_loss']))

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Reward loss", fontsize=16)
    plt.show()
    

def compare_results(results1, results2, label1, label2, moving_average=True, average_window=100):
    
    score1, asymptotic_score1, asymptotic_err1, trained_agent1, time_profile1, losses1, steps_to_solve1 = results1
    score2, asymptotic_score2, asymptotic_err2, trained_agent2, time_profile2, losses2, steps_to_solve2 = results2
    
    t_play1 = time_profile1[:,0].mean()
    t_update1 = time_profile1[:,1].mean()

    print("Average time (1) for playing one episode: %.2f s"%t_play1)
    print("Average time (1) for updating the agent: %.2f s"%t_update1)
    print("Asymptotic score (1): %.3f +/- %.3f"%(asymptotic_score1, asymptotic_err1))
    
    t_play2 = time_profile2[:,0].mean()
    t_update2 = time_profile2[:,1].mean()

    print("Average time (2) for playing one episode: %.2f s"%t_play2)
    print("Average time (2) for updating the agent: %.2f s"%t_update2)
    print("Asymptotic score (2): %.3f +/- %.3f"%(asymptotic_score2, asymptotic_err2))
    
    if moving_average:
        n_epochs = np.arange(average_window, len(score1))
    else:
        n_epochs = np.arange(len(score1))
        
    ### plot score ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score1 = np.array([np.mean(score1[i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score1, label=label1)
        average_score2 = np.array([np.mean(score2[i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score2, label=label2)
    else:
        plt.plot(n_epochs, score1, label=label1)
        plt.plot(n_epochs, score2, label=label2)
        
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.legend(fontsize=13)
    plt.show()
    
    ### steps to solve ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score1 = np.array([np.mean(steps_to_solve1[i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score1, label=label1)
        average_score2 = np.array([np.mean(steps_to_solve2[i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score2, label=label2)
    else:
        plt.plot(n_epochs, steps_to_solve1, label=label1)
        plt.plot(n_epochs, steps_to_solve2, label=label2)
        
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Steps to solve", fontsize=16)
    plt.legend(fontsize=13)
    plt.show()
    
    ### plot critic loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score1 = np.array([np.mean(losses1['critic_losses'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score1, label=label1)
        average_score2 = np.array([np.mean(losses2['critic_losses'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score2, label=label2)
    else:
        plt.plot(n_epochs, losses1['critic_losses'], label=label1)
        plt.plot(n_epochs, losses2['critic_losses'], label=label2)
        
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Critic loss", fontsize=16)
    plt.legend(fontsize=13)
    plt.show()

    ### plot actor loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score1 = np.array([np.mean(losses1['actor_losses'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score1, label=label1)
        average_score2 = np.array([np.mean(losses2['actor_losses'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score2, label=label2)
    else:
        plt.plot(n_epochs, losses1['actor_losses'], label=label1)
        plt.plot(n_epochs, losses2['actor_losses'], label=label2)

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Actor loss", fontsize=16)
    plt.legend(fontsize=13)
    plt.show()
    
    ### plot entropy ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score1 = np.array([np.mean(losses1['entropies'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, -average_score1, label=label1)
        average_score2 = np.array([np.mean(losses2['entropies'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, -average_score2, label=label2)
    else:
        plt.plot(n_epochs, -np.array(losses1['entropies']), label=label1)
        plt.plot(n_epochs, -np.array(losses2['entropies']), label=label2)
        
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Entropy term", fontsize=16)
    plt.legend(fontsize=13)
    plt.show()
    
    
def plot_session(score, steps_to_solve, losses, moving_average=True, average_window=100):
    
    if moving_average:
        n_epochs = np.arange(average_window, len(score))
    else:
        n_epochs = np.arange(len(score))
        
    ### plot score ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(score[i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, score)
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.show()
    
    ### steps to solve ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(steps_to_solve[i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, steps_to_solve)
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Steps to solve", fontsize=16)
    plt.show()
    
    ### plot critic loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['critic_losses'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, losses['critic_losses'])

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Critic loss", fontsize=16)
    plt.show()

    ### plot actor loss ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['actor_losses'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, average_score)
    else:
        plt.plot(n_epochs, losses['actor_losses'])

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Actor loss", fontsize=16)
    plt.show()
    
    ### plot entropy ###
    plt.figure(figsize=(8,6))

    if moving_average:
        average_score = np.array([np.mean(losses['entropies'][i:i+average_window]) for i in range(len(n_epochs))])
        plt.plot(n_epochs, -average_score)
    else:
        plt.plot(n_epochs, -np.array(losses['entropies']))

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Entropy term", fontsize=16)
    plt.show()
    
def compare_sessions(list_of_scores, list_of_steps, list_of_losses, list_of_labels, moving_average=True, average_window=100):
    
    if moving_average:
        n_epochs = np.arange(average_window, len(list_of_scores[0]))
    else:
        n_epochs = np.arange(len(list_of_scores[0]))
        
    ### plot score ###
    plt.figure(figsize=(8,6))
    
    for i, score in enumerate(list_of_scores):
        if moving_average:
            average_score = np.array([np.mean(score[i:i+average_window]) for i in range(len(n_epochs))])
            plt.plot(n_epochs, average_score, label=list_of_labels[i])
        else:
            plt.plot(n_epochs, score, label=list_of_labels[i])
        
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.legend(fontsize=13)
    plt.show()
    
    ### steps to solve ###
    plt.figure(figsize=(8,6))

    for i, steps in enumerate(list_of_steps):
        if moving_average:
            average_steps = np.array([np.mean(steps[i:i+average_window]) for i in range(len(n_epochs))])
            plt.plot(n_epochs, average_steps, label=list_of_labels[i])
        else:
            plt.plot(n_epochs, steps, label=list_of_labels[i])
        
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Steps to solve", fontsize=16)
    plt.legend(fontsize=13)
    plt.show()
    
    ### plot critic loss ###
    plt.figure(figsize=(8,6))

    for i, losses in enumerate(list_of_losses):
        if moving_average:
            average_loss = np.array([np.mean(losses['critic_losses'][i:i+average_window]) for i in range(len(n_epochs))])
            plt.plot(n_epochs, average_loss, label=list_of_labels[i])
        else:
            plt.plot(n_epochs, losses['critic_losses'], label=list_of_labels[i])
        
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Critic loss", fontsize=16)
    plt.legend(fontsize=13)
    plt.show()

    ### plot actor loss ###
    plt.figure(figsize=(8,6))

    for i, losses in enumerate(list_of_losses):
        if moving_average:
            average_loss = np.array([np.mean(losses['actor_losses'][i:i+average_window]) for i in range(len(n_epochs))])
            plt.plot(n_epochs, average_loss, label=list_of_labels[i])
        else:
            plt.plot(n_epochs, losses['actor_losses'], label=list_of_labels[i])

    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Actor loss", fontsize=16)
    plt.legend(fontsize=13)
    plt.show()
    
    ### plot entropy ###
    plt.figure(figsize=(8,6))

    for i, losses in enumerate(list_of_losses):
        if moving_average:
            average_loss = np.array([np.mean(losses['entropies'][i:i+average_window]) for i in range(len(n_epochs))])
            plt.plot(n_epochs, -average_loss, label=list_of_labels[i])
        else:
            plt.plot(n_epochs, -np.array(losses['entropies']), label=list_of_labels[i])
        
    plt.xlabel("Number of epochs", fontsize=16)
    plt.ylabel("Entropy term", fontsize=16)
    plt.legend(fontsize=13)
    plt.show()