import os, inspect
import pybullet_envs, gym
import numpy as np
import time, datetime

import argparse
from omegaconf import OmegaConf
import wandb

import common.helper as h
from common.record_animation import show_video_of_model
from common.algo.CMAES import cma_es
from common.algo.CEM import cem
from common.Agent import Agent

# import mocca_envs
from envs.envs import Walker2DBulletEnv
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Get current dir
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def main(cfg):
    
    h.make_dir(cfg['result_file_path']) # create result folder
    env = Walker2DBulletEnv(render=False, direction = cfg['direction'])# need to modify later
    # initialize multiple(pop_size) environments
    envs = gym.vector.AsyncVectorEnv(
        [lambda: Walker2DBulletEnv(render=False, direction = cfg['direction'])] * cfg['pop_size'],
        shared_memory=True
        )
    # initialize agent
    agent = Agent(env, envs, cfg)

    # print cfg
    print("#"*19 + " Config " + "#"*19)
    print(OmegaConf.to_yaml(cfg))

    if cfg['algo'] == "CEM":
        # train the model using cem 
        best_reward = cem(agent, cfg)
    else:
        best_reward = cma_es(agent, cfg)

    print("\nBest reward: \t", best_reward)
    
    # close the environment
    env.close()
    envs.close()

    return best_reward

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", required=False, help="Open wandb", action='store_true')
    ap.add_argument("--algo", type=str, required=False, choices=['CEM', 'CMA-ES'], 
                        help="Choose the algorithm")
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
    if args['algo'] is not None:
        cfg['algo'] = args['algo']

    # set result path
    cfg['result_file_path'] = os.path.join(currentdir, "result", cfg['env_name'], cfg['direction'])

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

    # run algo
    start = time.time()
    best_reward = main(cfg)
    seconds = time.time() - start

    running_time = str(datetime.timedelta(seconds=seconds))
    print('\nThe time of execution of main multiple process program is: ', running_time)

    # save video
    show_video_of_model(False, cfg)

    # keep data to wandb
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