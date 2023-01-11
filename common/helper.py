import os
import numpy as np
import yaml
import matplotlib.pyplot as plt

def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

def save_file(filename, arr):
    # open a binary file in write mode
    file = open(filename, "wb")
    # save array to the file
    np.save(file, arr)
    # close the file
    file.close

def load_file(filename):
    # open the file in read binary mode
    file = open(filename, "rb")
    #read the file to numpy array
    arr = np.load(file)
    return arr

def load_config(configFilePath):
    with open(configFilePath, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

def interpolation(keypoints, interval):

    actions = []
    # interpolate 
    actions = [np.linspace(keypoints[index], keypoints[index + 1], interval)  for index in range(len(keypoints)) if index != len(keypoints)-1]
    # concatenate
    # actions size => [(num_keypoints-1) * interval] * action_dim
    actions = np.concatenate(actions, axis=0)
    
    return actions

def plot(scores, path):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(path + '/scores.png')
    plt.close(fig)


def clamp(array, lower_bound, upper_bound):
    
    for index in range(len(array)):
        array[index] = max(min(array[index], upper_bound[index]), lower_bound[index])
    return array


def beautify_time(running_time):
    running_time = running_time.split(",")
    if len(running_time) == 2:
        day = running_time[0].split(" ")[0] + "d"
    
    clock = running_time[-1].split(":")
    hour = clock[0] + "h " if int(clock[0])!=0 else ""
    minute = clock[1] +"m " if int(clock[1])!=0 else ""
    second = clock[2].split(".")[0] + "s" if int(clock[2].split(".")[0])!=0 else ""

    if 'day' in locals():
        running_time = day + hour + minute + second
    else:
        running_time = hour + minute + second
    return running_time