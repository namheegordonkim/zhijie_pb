# zhijie_pb

### Installation

```bash
pip install -r requirements.txt.
```

### Running the project

usage: `main.py [-h] [--wandb] [-d {forward,backward}] [--iter ITER] [--pop_size POP_SIZE] [--num_keypoints NUM_KEYPOINTS]`

optional arguments to the `main` script are:
* -h, --help            show this help message and exit
* --wandb               Open wandb
* -d {forward,backward}, --direction {forward,backward} Decide direction
* --iter ITER           The number of iterations
* --pop_size POP_SIZE   The size of population
* --num_keypoints NUM_KEYPOINTS The number of keypoints

Example:

```bash
python main.py --wandb -d backward --iter 2000 --pop_size 50 --num_keypoints 6
```


### For windows install pybullet

pls check this method: https://deepakjogi.medium.com/how-to-install-pybullet-physics-simulation-in-windows-e1f16baa26f6

### CEM
Here I first implement cross entropy method and in order to avoid curse of dimensionality, reduce the dimensionality by selecting few number of keypoints, and then interpolate using spline-based methods.

