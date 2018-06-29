
# coding: utf-8

import numpy as np
import sys
sys.path.append("/home/a_parida/MantaFlow/manta/tensorflow/tools/")
import uniio
import glob
import pickle

vel_files= glob.glob('./data/**/*.uni', recursive=True)
# load data
velocities = []

for uniPath in vel_files:
    header, content = uniio.readUni(uniPath)# returns [Z,Y,X,C] np array
    h = header['dimX']
    w  = header['dimY']
    arr = content[:, ::-1, :, :-1]  # reverse order of Y axis
    arr = np.reshape(arr, [w, h, 2])# discard Z from [Z,Y,X]
    velocities.append( arr )

loadNum = len(velocities)
if loadNum<200:
	print("Error - use at least two full sims, generate data by running 'manta ./manta_genSimSimple.py' a few times..."); exit(1)

velocities = np.reshape( velocities, (len(velocities), 64,64,2) )

with open('./data/velocity.pickle', 'wb') as handle:
    pickle.dump(velocities, handle)
