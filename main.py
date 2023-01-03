import inspect
import os
import gym
import pybullet_envs
import numpy as np
import time
from collections import deque
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from omegaconf import OmegaConf

import common.helper as h
from record_animation import show_video_of_model

# import mocca_envs
from envs.envs import Walker2DBulletEnv


class Agent():
    def __init__(self, env, envs, cfg):
        self.env = env
        self.envs = envs
        self.cfg = cfg

    def evaluate(self, actions):
        _ = self.env.reset()
        
        episode_return = 0.0
        for action in actions:
            _, reward, done, _ = self.env.step(action)
            episode_return += reward 
            if done:
                break

        return episode_return

    def evaluate_parallel(self, actions):
        _ = self.envs.reset()
        
        episode_return = np.zeros(self.cfg['pop_size'])
        dones_flag = np.zeros(self.cfg['pop_size'], dtype=bool)
        for action in actions:
            _, rewards, dones, _ = self.envs.step(action)

            # if all env failed, then break
            if sum(~dones_flag) == 0:
                break

            dones_flag += dones
            # calculate reward
            episode_return += np.multiply(rewards, ~dones_flag)

        return episode_return


# cross entropy method
def cem(agent,
        action_dim, 
        num_keypoints,
        upper_bound,
        lower_bound,
        interval,
        target_reward = 500,
        scores_deque_length = 100,
        n_iterations=500, 
        print_every=10, 
        pop_size=50, 
        elite_frac=0.2, 
        sigma=0.1):
    """The Implementation of the cross-entropy method.
        
    Params
    ======
        Agent (object): agent instance
        action_dim (int): the number of action dimension
        num_keypoints (int): the number of keypoints
        upper_bound (array): the upper bound of joints radius
        lower_bound (array): the lower bound of joints radius
        interval: interpolation interval
        target_reward : the target reward
        scores_deque_length : 100 the number of scores length
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """

    # set the number of elite
    n_elite = int(pop_size * elite_frac)
     
    # set variable store scores
    scores_deque = deque(maxlen=scores_deque_length)
    scores = []

    # Initialize the sample within bounds
    best_sample_keypoints = np.array(
        [np.random.uniform(lower_bound[i], upper_bound[i], num_keypoints) for i in range(action_dim)]
        )

    # dim = num_keypoints x action_dim
    best_sample_keypoints = np.dstack(best_sample_keypoints)[0]

    # Return variables
    # create variable to store best reward and best action keypoints as return
    best_reward = 0
    best_actions_keypoints = np.zeros_like(best_sample_keypoints)
    
    for i_iteration in range(1, n_iterations+1):
        # generate the population
        # population_keypoints shape => num_keypoints * pop_size * action_dim 
        population_keypoints  = np.array(
            # Add fluctuation to best keypoints and generate population
            # clamp function -> make sure the angle in the right range
            [
                [h.clamp(best_sample_keypoints[j] + sigma * np.random.randn(action_dim), lower_bound, upper_bound) for _ in range(pop_size)]
                                                                                    for j in range(num_keypoints)]
            )

        # interpolation => shape => [(num_keypoints-1)*interval] * pop_size * action_dim
        population = h.interpolation(population_keypoints, interval)

        # evaluate population
        rewards = agent.evaluate_parallel(population)
        
        # Select n_elite best candidates from collected rewards
        elite_idxs = rewards.argsort()[-n_elite:]
        
        # Select n_elite best elite keypoints according to elite index; 
        elite_samples_keypoints = population_keypoints[:,elite_idxs,:] # elite_idxs is in the ascending order

        # cal elite keypoints mean to get the best keypoints
        best_sample_keypoints = np.array(elite_samples_keypoints).mean(axis=1)
        
        # evluate the mean keypoints
        reward = agent.evaluate(h.interpolation(best_sample_keypoints, interval))
        
        scores_deque.append(reward)
        scores.append(reward)     

        # get best action
        if max(rewards) > best_reward:
            best_reward = max(rewards)
            # save the largest reward keypoints
            best_actions_keypoints = elite_samples_keypoints[:,-1,:]
            h.save_file(result_file_path + "/best_actions_keypoints", best_actions_keypoints)

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
        
        if np.mean(scores_deque) >= target_reward:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            break

    return best_actions_keypoints, best_reward, scores

#------------------------------- Get configuration -----------------------------------------------#

# Get current dir
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Get config path
config_file_path = currentdir + "/cfg/env_cfg.yaml"
# set result path
result_file_path = currentdir + "/result"

# Load config file
cfg = h.load_config(config_file_path)

def main():
    # running id
    # run_id = int(time.time())
    h.make_dir(result_file_path) # create result folder

    # need to modify later
    env = Walker2DBulletEnv(render=False)
    env.reset()

    # initialize multiple(pop_size) environments
    envs = gym.vector.AsyncVectorEnv(
        [lambda: Walker2DBulletEnv(render=False)] * cfg['pop_size'],
        shared_memory=True
        )
    envs.reset()
    
    # print cfg
    print("#"*19 + " Config " + "#"*19)
    print(OmegaConf.to_yaml(cfg))
    # pre-setting
    action_dim = envs.single_action_space.shape[0] # get action dimension
    # initialize agent
    agent = Agent(env, envs, cfg)

    # get the bound of joints radius
    hi = np.array([j.upperLimit for j in env.ordered_joints], dtype=np.float32).flatten()
    lo = np.array([j.lowerLimit for j in env.ordered_joints], dtype=np.float32).flatten()
    
    # train the model using cem 
    best_actions_keypoints, best_reward, scores = cem(
                        agent = agent, 
                        action_dim = action_dim, 
                        num_keypoints = cfg['num_keypoints'], 
                        upper_bound = hi, 
                        lower_bound = lo, 
                        interval = cfg['interpolate_interval'], 
                        n_iterations = cfg['n_iterations'],
                        target_reward = cfg['target_reward'],
                        scores_deque_length = cfg['scores_deque_length'],
                        pop_size = cfg['pop_size'],
                        elite_frac = cfg['elite_frac']
                        )

    print("\nBest reward: \t", best_reward)

    # save file
    h.save_file(result_file_path + "/best_actions_keypoints", best_actions_keypoints)

    h.plot(scores, result_file_path)
    # close the environment
    env.close()
    envs.close()
    
    return 0

def test():
    env = Walker2DBulletEnv(render=False)
    env.reset()
    # TODO: find action_pieces such that sum_reward is large. Use something like CEM or CMA-ES.
    # load best actions keypoints
    best_actions_keypoints = h.load_file("result/best_actions_keypoints")
    # get best action lists
    best_actions = h.interpolation(best_actions_keypoints, interval = cfg['interpolate_interval'])

    episode_return = 0.0
    for action in best_actions:
        _, reward, done, _ = env.step(action)
        episode_return += reward 
        env.render()
        time.sleep(0.066)

        if done:
            break

    print("Episode return:", episode_return)
    env.close()
        
if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()
    print("\nThe time of execution of main multiple process program is :",
      (end-start) * 10**3, "ms\n")

    # save video
    show_video_of_model()
    
    #print("Testing: \n")
    # test()
