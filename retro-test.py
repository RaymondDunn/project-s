import retro
import time
import numpy as np
import scipy.io
import tifffile as tf


def main():
    game = ('MsPacMan-Genesis')
    env = retro.make(game=game)
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        #time.sleep(0.01)
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()

# if you want to run the game interactively...
# python -m retro.examples.interactive --game SpaceInvaders-Atari2600

# Downloaded ROMs
# SpaceInvaders-Atari2600
# MsPacMan-Genesis
# Asteroids-Atari2600

# import 
# python -m retro.import .

# fname
fname = 'C:/Users/rldun/Desktop/wbSimpleStruct-20200221-12-01-44.mat'
mat = scipy.io.loadmat(fname)
dff = mat['deltaFOverF']

# action = np.ones((1000)
game = ('MsPacMan-Genesis')
env = retro.make(game=game)
obs = env.reset()
