import gym
import pybullet_envs
import numpy as np
import time
from collections import deque
import random

# import mocca_envs
from envs.envs import Walker2DBulletEnv

def interpolation(keypoints, interval):

    actions = []
    # interpolate 
    actions = [np.linspace(keypoints[index], keypoints[index + 1], interval)  for index in range(len(keypoints)) if index != len(keypoints)-1]
    # concatenate
    # actions size => [(num_keypoints-1) * interval] * action_dim
    actions = np.concatenate(actions, axis=0)
    
    return actions

class Agent():
    def __init__(self, env):
        self.env = env

    def evaluate(self, actions):
        _ = self.env.reset()
        
        episode_return = 0.0
        for action in actions:
            _, reward, done, _ = self.env.step(action)
            episode_return += reward 
            if done:
                break

        return episode_return


# cross entropy method
def cem(agent,
        action_dim, 
        num_keypoints,
        upper_bound,
        lower_bound,
        interval,
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
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """
    # set the number of elite
    n_elite=int(pop_size * elite_frac)
    # scores deque
    scores_deque = deque(maxlen=100)
    scores = []

    # Initialize the sample within limitations
    best_sample_keypoints = np.array([np.random.uniform(lower_bound[i], upper_bound[i], num_keypoints) for i in range(action_dim)])
    best_sample_keypoints = np.dstack([i for i in best_sample_keypoints])[0]
    
    # best action list reward
    best_reward = 0
    for i_iteration in range(1, n_iterations+1):
        # get the population
        # samples shape => pop_size * action_dim * num_keypoints
        # population_keypoints = np.array([np.array([np.random.uniform(lower_bound[i], upper_bound[i], num_keypoints) for i in range(action_dim)]) for j in range(pop_size)])
        population_keypoints = np.array([np.array([np.random.randn(num_keypoints) for i in range(action_dim)]) for j in range(pop_size)])

        # population shape => pop_size * num_keypoints * action_dim 
        population_keypoints = np.stack([best_sample_keypoints + sigma * np.dstack([i for i in sample_keypoint])[0] for sample_keypoint in population_keypoints])
        
        
        # interpolate and population shape => pop_size * [(num_keypoints-1) * interval)] * action_dim
        population = np.stack([interpolation(keypoints, interval) for keypoints in population_keypoints])
        
        # evaluate population
        rewards = np.array([agent.evaluate(sample) for sample in population])
        
        # Select best candidates from collected rewards
        elite_idxs = rewards.argsort()[-n_elite:]
        elite_samples_keypoints = [population_keypoints[i] for i in elite_idxs]
        elite_samples = [population[i] for i in elite_idxs]
        best_sample_keypoints = np.array(elite_samples_keypoints).mean(axis=0)
        best_sample = np.array(elite_samples).mean(axis=0)

        # evluate the best sample which got according to mean method
        reward = agent.evaluate(best_sample)
        scores_deque.append(reward)
        scores.append(reward)       

        # got best action
        if reward > best_reward:
            best_reward = reward
            best_actions_keypoints = best_sample_keypoints

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
        
        if np.mean(scores_deque)>=300.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            break

    return best_actions_keypoints, scores

def load_arr(file_path):
    # open the file in read binary mode
    file = open(file_path, "rb")
    #read the file to numpy array
    best_actions_keypoints = np.load(file)
    return best_actions_keypoints


def main():
    # initialize environment
    env = Walker2DBulletEnv(render=False)
    env.reset()

    # pre=setting
    action_dim = env.action_space.shape[0] # get action dimension
    num_keypoints = 7 # the number of key points
    interpolate_interval = 50 # set the interpolate interval

    # agent
    agent = Agent(env)

    # get the bound of joints radius
    hi = np.array([j.upperLimit for j in env.ordered_joints], dtype=np.float32).flatten()
    lo = np.array([j.lowerLimit for j in env.ordered_joints], dtype=np.float32).flatten()

    # number of iterations
    n_iterations = 200

    best_actions_keypoints, scores = cem(agent, 
                    action_dim, 
                    num_keypoints, 
                    hi, 
                    lo, 
                    interpolate_interval, 
                    n_iterations)
    print(scores)

    # open a binary file in write mode
    file = open("best_actions_keypoints", "wb")
    # save array to the file
    np.save(file, best_actions_keypoints)
    # close the file
    file.close

    return 0
   

def test():
    env = Walker2DBulletEnv(render=True)
    env.reset()
     # TODO: find action_pieces such that sum_reward is large. Use something like CEM or CMA-ES.
    sum_reward = 0
    # load best actions keypoints
    best_actions_keypoints = load_arr("best_actions_keypoints")
    # get best action lists
    best_actions = interpolation(best_actions_keypoints, interval=50)

    episode_return = 0.0
    for action in best_actions:
        _, reward, done, _ = env.step(action)
        episode_return += reward 
        env.render()
        time.sleep(0.066)

        if done:
            break

    print("episode_return:", episode_return)
        
if __name__ == '__main__':
    main()

    # print("Testing: \n")
    # test()
    


