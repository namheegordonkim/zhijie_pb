import numpy as np
from cmaes import CMA
import common.helper as h
import os, inspect
import wandb
from collections import deque
# Get current dir
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def cma_es(agent, cfg, print_every=10):
    # get keypoints shape
    keypoints_shape = (cfg['num_keypoints'], agent.action_dim)
    # get action dim
    action_bounds = [*zip(agent.env.action_space.low, agent.env.action_space.high)]

    optimizer = CMA(
                    mean = np.array([0.0] * np.prod(keypoints_shape)), 
                    bounds = np.array(action_bounds * cfg['num_keypoints']), 
                    sigma = 0.5, 
                    n_max_resampling = 1,
                    population_size = cfg['pop_size']
                )

    scores_deque = deque(maxlen = optimizer.population_size)
    best_reward = 0
    for i_iteration in range(cfg['n_iterations']):
        solutions = []
        for _ in range(optimizer.population_size):
            keypoints = optimizer.ask()
            actions =  h.interpolation(keypoints.reshape(keypoints_shape), cfg['interpolate_interval'])
            score = -agent.evaluate(actions)

            if score < best_reward:
                best_reward = score
                best_actions_keypoints = keypoints.reshape(keypoints_shape)
                h.save_file(os.path.join(cfg['result_file_path'], "best_actions_keypoints"), best_actions_keypoints)
                print('*Episode {}\t Best score: {:.2f}, file saved!'.format(i_iteration, -best_reward))

            solutions.append((keypoints, score))
            scores_deque.append(-score)
        optimizer.tell(solutions)

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
        
        if cfg["wandb"]:
            wandb.log({
                "best reward": -best_reward, 
                "mean reward": np.mean(scores_deque)
                })
        
    return -best_reward

