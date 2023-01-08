from collections import deque
import numpy as np
import common.helper as h
import wandb
import os

# cross entropy method
def cem(agent,
        cfg,
        print_every=10):
    """The Implementation of the cross-entropy method.
        
    Params
    ======
        Agent (object): agent instance
        cfg (dict): the configuration dict
        print_every (int): how often to print average score (over last 100 episodes)
    """

    # set the number of elite
    n_elite = int(cfg['pop_size'] * cfg['elite_frac'])
     
    # set variable store scores
    scores_deque = deque(maxlen=cfg['pop_size'])
    scores = []

    # get bounds
    upper_bound, lower_bound = agent.get_bounds()

    # Initialize the sample within bounds
    best_sample_keypoints = np.array(
        [np.random.uniform(lower_bound[i], upper_bound[i], cfg['num_keypoints']) for i in range(agent.action_dim)]
        )
    
    # dim = num_keypoints x action_dim
    best_sample_keypoints = np.dstack(best_sample_keypoints)[0]

    # Return variables
    # create variable to store best reward and best action keypoints as return
    best_reward = 0
    best_actions_keypoints = np.zeros_like(best_sample_keypoints)
    
    for i_iteration in range(1, cfg['n_iterations']+1):
        # generate the population
        # population_keypoints shape => num_keypoints * pop_size * action_dim 
        N = np.random.normal(size=(cfg['num_keypoints'], cfg['pop_size'], agent.action_dim))
        population_keypoints  = np.array(
            # Add fluctuation to best keypoints and generate population
            # clamp function -> make sure the angle in the right range
            [
                [h.clamp(best_sample_keypoints[j] + N[j][i], lower_bound, upper_bound) for i in range(cfg['pop_size'])]
                                                                                    for j in range(cfg['num_keypoints'])]
            )

        # interpolation => shape => [(num_keypoints-1)*interval] * pop_size * action_dim
        population = h.interpolation(population_keypoints, cfg['interpolate_interval'])

        # evaluate population
        rewards = agent.evaluate_parallel(population)
        
        # Select n_elite best candidates from collected rewards
        elite_idxs = rewards.argsort()[-n_elite:]
        
        # Select n_elite best elite keypoints according to elite index; 
        elite_samples_keypoints = population_keypoints[:,elite_idxs,:] # elite_idxs is in the ascending order

        # Here we have many ways to decide the best sample keypoints
        ### 1. calculate the mean
        # cal elite keypoints mean to get the best keypoints
        # best_sample_keypoints = np.array(elite_samples_keypoints).mean(axis=1)
        ### 2. select max reward sample
        best_sample_keypoints = elite_samples_keypoints[:,-1,:]

        # evluate the mean keypoints
        reward = agent.evaluate(h.interpolation(best_sample_keypoints, cfg['interpolate_interval']))
        
        scores_deque.append(reward)
        scores.append(reward)     

        # get best action
        if max(rewards) > best_reward:
            best_reward = max(rewards)
            # save the largest reward keypoints
            best_actions_keypoints = elite_samples_keypoints[:,-1,:]

            h.save_file(os.path.join(cfg['result_file_path'], "best_actions_keypoints"), best_actions_keypoints)
            print('*Episode {}\t Best score: {:.2f}, file saved!'.format(i_iteration, best_reward))
        
        if cfg["wandb"]:
            wandb.log({
                "best reward": best_reward, 
                "mean reward": np.mean(scores_deque)
                })

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

    return best_reward
