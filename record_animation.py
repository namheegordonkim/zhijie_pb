import gym
import os
import inspect
import time
# For visualization
from gym.wrappers.monitoring import video_recorder
import common.helper as h

from envs.envs import Walker2DBulletEnv

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def show_video_of_model(render):
    # initialize environment
    env = Walker2DBulletEnv(render=render)

    # get config file
    config_file_path = currentdir + "/cfg/env_cfg.yaml"
    cfg = h.load_config(config_file_path)

    video_path = currentdir + "/result/video"
    h.make_dir(video_path)
    
    # create video
    vid = video_recorder.VideoRecorder(env, path= video_path + "/{}.mp4".format("Walker2DBulletEnv"), enabled=video_path is not None)

    # load best actions
    best_actions_keypoints = h.load_file("result/best_actions_keypoints")
    # get best action lists
    best_actions = h.interpolation(best_actions_keypoints, interval=cfg['interpolate_interval'])

    env.reset()
    episode_return = 0.0
    for action in best_actions:
        env.unwrapped.render()
        vid.capture_frame()
        _, reward, done, _ = env.step(action)
        episode_return += reward 
        env.render()
        time.sleep(0.066)

        if done:
            break
    print("Video has been saved to "+ video_path + "/{}.mp4".format("Walker2DBulletEnv"))
    print("Episode return:", episode_return)
    # close environment
    env.close()

if __name__ == '__main__':
    show_video_of_model(render = False)
