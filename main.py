# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gym
import pybullet_envs
import numpy as np
import time

# import mocca_envs
from envs.envs import Walker2DBulletEnv


def main():
    env = Walker2DBulletEnv(render=True)
    env.reset()

    actions = np.random.random([100, 6])

    for a in actions:
        # env.step(np.zeros(6))
        env.step(a)
        env.render()
        time.sleep(0.033)

    # import pybullet as p
    # import time
    # import pybullet_data
    # physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    # p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    # p.setGravity(0, 0, -10)
    # planeId = p.loadURDF("plane.urdf")
    # startPos = [0, 0, 1]
    # startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    # boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)
    # # set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    # for i in range(10000):
    #     p.stepSimulation()
    #     time.sleep(1. / 240.)
    # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos, cubeOrn)
    # p.disconnect()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
