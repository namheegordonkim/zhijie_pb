import gym
import os
import inspect
import time
# For visualization
from gym.wrappers.monitoring import video_recorder
import common.helper as h
# import mocca_envs
from envs.envs import Walker2DBulletEnv

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def show_video_of_model():
    # initialize environment
    env = Walker2DBulletEnv(render=False)
    env.reset()

    video_path = currentdir + "/video"
    h.make_dir(video_path)

    # create video
    vid = video_recorder.VideoRecorder(env, path= video_path + "/{}.mp4".format("Walker2DBulletEnv"), enabled=video_path is not None)

    # load best actions
    best_actions_keypoints = h.load_file("result/best_actions_keypoints")
    # get best action lists
    best_actions = h.interpolation(best_actions_keypoints, interval=50)

    env.reset()
    done = False

    for action in best_actions:
        env.unwrapped.render()
        vid.capture_frame()
        _, _, done, _ = env.step(action)
        env.render()
        time.sleep(0.066)

        if done:
            break

    env.close()

if __name__ == '__main__':
    show_video_of_model()
