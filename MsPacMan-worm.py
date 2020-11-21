import retro
import time
import numpy as np
import scipy.io
import tifffile as tf
import random
import seaborn as sns
import matplotlib.pyplot as plt

up = [0, 0, 0, 3]
down = [0, 0, 0, 5]
right = [0, 0, 0,1,1,5]
left = [0, 0, 0,1,1,3]
action_list = [up, down, left, right]

# load data
fname = 'C:/Users/rldun/code/misc_scripts/retro/wbSimpleStruct-20200221-12-01-44.mat'
mat = scipy.io.loadmat(fname)
dff = mat['deltaFOverF']
ids = mat['ID1'][0]
AVAL_ndx = np.where(ids=='AVAL')[0][0]
numt = dff.shape[0]

# load tracestate
fname_mat = 'C:/Users/rldun/code/misc_scripts/retro/traceState-20200221-12-01-44.mat'
tracestate_mat = scipy.io.loadmat(fname_mat)
tracestate = tracestate_mat['traceState'][0][0][0]
AVAL_tracestate = tracestate[:,AVAL_ndx]
AVAL_trace = dff[:,AVAL_ndx]

# load rom
#game = ('MsPacMan-Genesis')
#env = retro.make(game=game)
#obs = env.reset()

# plot quick heatplot of activity
#fig, ax = plt.subplots()
sns.heatmap(dff.T)
plt.xticks([])
plt.yticks([])
plt.show()
plt.xlabel('Time (s)')
plt.ylabel('Neuron #')

def main():
    game = ('MsPacMan-Genesis')
    env = retro.make(game=game)
    obs = env.reset()

    t = 0
    while True:

        # env.action_space.sample() # useful function
        # obs, rew, done, info = env.step(random.choice(action_list))

        # with some probability do random action
        prandom = 0.05
        if random.random() < prandom:
            obs, rew, done, info = env.step(random.choice(action_list))
        else:
            # tracestate at time t
            act = action_list[AVAL_tracestate[t] - 1]
            obs, rew, done, info = env.step(act)

        # render image
        env.render()

        # increment counter
        t += 1  
        if t == numt:
            t = 0

        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()