import os
import inspect
# For visualization
from gym.wrappers.monitoring import video_recorder
import common.helper as h

from envs.envs import Walker2DBulletEnv

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
config_file_path = currentdir + "/cfg/env_cfg.yaml"

def show_video_of_model(render = False, cfg = None):
    # get config file
    # Load config file
    if cfg is not None:
        cfg = cfg
    else:    
        cfg = h.load_config(config_file_path)
        # set result path
        cfg['result_file_path'] = os.path.join(currentdir, "result", cfg['env_name'], cfg['direction'])

    # initialize environment
    env = Walker2DBulletEnv(render = render, direction = cfg['direction'])
    video_dir = os.path.join(cfg['result_file_path'], "video")
    h.make_dir(video_dir)
    
    video_name = "{}.mp4".format(cfg['env_name'] + "_" + cfg['direction'])
    
    # create video
    vid = video_recorder.VideoRecorder(env, path = os.path.join(video_dir, video_name) , enabled=video_dir is not None)

    # load best actions
    best_actions_keypoints = h.load_file(os.path.join(cfg['result_file_path'], "best_actions_keypoints"))
    # get best action lists
    best_actions = h.interpolation(best_actions_keypoints, interval = cfg['interpolate_interval'])

    env.reset()

    episode_return = 0.0    

    for action in best_actions:
        # env.unwrapped.render()
        env.render()
        vid.capture_frame()
        _, reward, done, _ = env.step(action)
        episode_return += reward 
        # time.sleep(0.066)

        if done:
            break

    print("Video has been saved to "+ video_dir)
    print("Episode return:", episode_return)

    # close 
    vid.close() # do not forget this
    env.close()

if __name__ == '__main__':
    show_video_of_model(render = False)
