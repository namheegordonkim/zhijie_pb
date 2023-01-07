import inspect
import os
import pybullet_envs, gym
import numpy as np
import time, datetime
from collections import deque
import argparse
from omegaconf import OmegaConf
import wandb
import common.helper as h
from record_animation import show_video_of_model
# import mocca_envs
from envs.envs import Walker2DBulletEnv

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Get current dir
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class Agent():
    def __init__(self, env, envs, cfg):
        self.env = env
        self.envs = envs
        self.cfg = cfg
        self.action_dim = envs.single_action_space.shape[0] # get action dimension

    def get_bounds(self):
        _ = self.env.reset()
        # get the bound of joints radius
        hi = np.array([j.upperLimit for j in self.env.ordered_joints], dtype=np.float32).flatten()
        lo = np.array([j.lowerLimit for j in self.env.ordered_joints], dtype=np.float32).flatten()
        return hi, lo

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
        cfg,
        print_every=10):
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
    n_elite = int(cfg['pop_size'] * cfg['elite_frac'])
     
    # set variable store scores
    scores_deque = deque(maxlen=cfg['scores_deque_length'])
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

            if i_iteration >= cfg['save_model']:
                h.save_file(os.path.join(cfg['result_file_path'], "best_actions_keypoints"), best_actions_keypoints)
                print('*Episode {}\t Best score: {:.2f}, model saved!'.format(i_iteration, best_reward))
        
        if cfg["wandb"]:
            wandb.log({
                "best reward": best_reward, 
                "mean reward": np.mean(scores_deque)
                })

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
        
        if np.mean(scores_deque) >= cfg['target_reward']:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            break

    return best_actions_keypoints, best_reward, scores


def main(cfg):
    h.make_dir(cfg['result_file_path']) # create result folder
    env = Walker2DBulletEnv(render=False, direction = cfg['direction'])# need to modify later

    # initialize multiple(pop_size) environments
    envs = gym.vector.AsyncVectorEnv(
        [lambda: Walker2DBulletEnv(render=False, direction = cfg['direction'])] * cfg['pop_size'],
        shared_memory=True
        )

    # print cfg
    print("#"*19 + " Config " + "#"*19)
    print(OmegaConf.to_yaml(cfg))

    # initialize agent
    agent = Agent(env, envs, cfg)

    # train the model using cem 
    best_actions_keypoints, best_reward, scores = cem(agent, cfg)

    print("\nBest reward: \t", best_reward)
    
    # plot scores
    h.plot(scores, cfg['result_file_path'])
    # save file
    h.save_file(os.path.join(cfg['result_file_path'], "best_actions_keypoints"), best_actions_keypoints)

    # close the environment
    env.close()
    envs.close()

    return best_reward


        
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", required=False, help="Open wandb", action='store_true')
    ap.add_argument("-d", "--direction", type=str, required=False, choices=['forward', 'backward'], 
                        help="Decide direction")
    ap.add_argument("--iter", type=int, required=False,
                    help="The number of iterations")      
    ap.add_argument("--pop_size", type=int, required=False,
                    help="The size of population ")           
    ap.add_argument("--num_keypoints", type=int, required=False,
                    help="The number of keypoints ")   

    args = vars(ap.parse_args())

    #------------------------------- Get configuration -----------------------------------------------#
    # Get config path
    config_file_path = os.path.join(currentdir ,"cfg/env_cfg.yaml")
    # Load config file
    cfg = h.load_config(config_file_path)

    # update cfg
    if args['wandb']:
        cfg['wandb'] = True
    if args['direction'] is not None:
        cfg['direction'] = args['direction']
    if args['iter'] is not None:
        cfg['n_iterations'] = args['iter']
    if args['pop_size'] is not None:
        cfg['pop_size'] = args['pop_size']
    if args['num_keypoints'] is not None:
        cfg['num_keypoints'] = args['num_keypoints']

    # set result path
    cfg['result_file_path'] = os.path.join(currentdir, "result", cfg['env_name'], cfg['direction'])

    # set result path
    # get time
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # set run name
    run_name = f"{cfg['env_name']}_{cfg['direction']}_{time_str}"

    # create wandb
    if cfg['wandb']:
        # Add `sync_tensorboard=True` when you start a W&B run
        # W&B supports TensorBoard to automatically log all the metrics from your script into our dashboards 
        wandb.init(
            project="ResearchProject",
            group=cfg['env_name'] + cfg['direction'],
            name=run_name,
            # sync_tensorboard=True
        )

    start = time.time()

    best_reward = main(cfg)
    seconds = time.time() - start
    running_time = str(datetime.timedelta(seconds=seconds))

    print('\nThe time of execution of main multiple process program is: ', running_time)
    # print('\nThe time of execution of main multiple process program is {:.3f} seconds.'.format(running_time))
    
    # save video
    # show_video_of_model(False, cfg)

    if cfg['wandb']:
        # add wandb table
        columns = list(cfg.keys()) + ['best_reward', 'running_time']
        table = wandb.Table(columns=columns)
        record = list(cfg.values()) + [best_reward, running_time]

        table.add_data(*record)
        wandb.log({"result_table":table}, commit=False)

        # Log the video 
        video_name = cfg['env_name'] + "_" + cfg['direction']
        path_to_video = os.path.join(cfg['result_file_path'], "video", "{}.mp4".format(video_name))
        wandb.log({"video": wandb.Video(path_to_video, fps=4, format="mp4", caption = video_name)})

        wandb.finish()