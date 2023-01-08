import numpy as np

class Agent():
    def __init__(self, env, envs, cfg):
        self.env = env
        self.envs = envs
        self.cfg = cfg
        self.env.reset()
        self.envs.reset()
        self.action_dim = envs.single_action_space.shape[0] # get action dimension

    def get_bounds(self):
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
