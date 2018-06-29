import tensorflow as tf
import numpy as np
import sys
sys.path.append("/home/a_parida/MantaFlow/manta/tensorflow/tools/")
import uniio
import glob
import random
import scipy.misc
import os
basePath = 'data/' # path where data is availabe;

trainingEpochs = 1500
batchSize      = 150
inSize         = 64 * 64 * 2
######################################################
# Ex 2.1 – Saving and Loading Training Data
#####################################################

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

print("Read uni files, total data " + format(velocities.shape) )
valiSize = max(100, int(loadNum * 0.1)) # at least 1 full sim...
valiData = velocities[loadNum-valiSize:loadNum,:]
velocities  = velocities[0:loadNum-valiSize,:]
print("Split into %d training and %d validation samples" % (velocities.shape[0], valiData.shape[0]) )
loadNum = velocities.shape[0]

############################################################

##############################################################
# Ex 2.2– First Network Architecture
##############################################################

def convolution2d(input, biases, weights, strides, padding_kind='SAME'):
    input = tf.nn.conv2d(input, weights, strides, padding=padding_kind)
    input = tf.nn.bias_add(input, biases)
    input = tf.nn.leaky_relu(input)
    return input

def velocityFieldToPng(frameArray):
    """ Returns an array that can be saved as png with scipy.misc.toimage
    from a velocityField with shape [height, width, 2]."""
    outputframeArray = np.zeros((frameArray.shape[0], frameArray.shape[1], 3))
    for x in range(frameArray.shape[0]):
        for y in range(frameArray.shape[1]):
            # values above/below 1/-1 will be truncated by scipy
            frameArray[y][x] = (frameArray[y][x] * 0.5) + 0.5
            outputframeArray[y][x][0] = frameArray[y][x][0]
            outputframeArray[y][x][1] = frameArray[y][x][1]
    return outputframeArray

# the network structure
xIn = tf.placeholder(tf.float32, shape=[None, 64,64, 2])

y = tf.placeholder(tf.float32, shape=[None, 64,64,2])
xOut=tf.reshape(y, shape=[-1, 64*64*2])

#layer1: Convolution
weights1=tf.Variable(tf.random_normal([12,12,2,2], stddev=0.01))#weights==filters
#[filter_height, filter_width, in_channels, out_channels]
#bias=out_channels
bias1=tf.Variable(tf.random_normal([2], stddev=0.01))
stride1=[1,4,4,1]
out1=convolution2d(xIn,bias1,weights1,stride1)

#layer2: Convolution
weights2=tf.Variable(tf.random_normal([6,6,2,4], stddev=0.01))#weights==filters
#[filter_height, filter_width, in_channels, out_channels]
#bias=out_channels
bias2=tf.Variable(tf.random_normal([4], stddev=0.01))
stride2=[1,4,4,1]
out2=convolution2d(out1,bias2,weights2,stride2)

#layer3: Fully Connected Network
out2 = tf.reshape(out2, shape=[-1, 64 ]) # flatten
fc_1weights = tf.Variable(tf.random_normal([64, 64*64*2], stddev=0.01))
fc_1bias   = tf.Variable(tf.random_normal([64*64*2], stddev=0.01))
fc1 = tf.add(tf.matmul(out2, fc_1weights), fc_1bias)
fc1 = tf.nn.tanh(fc1)

# cost and Optimistion
cost = tf.nn.l2_loss(fc1 - xOut)
opt  = tf.train.AdamOptimizer(0.0001).minimize(cost)

#creating Seesion starting training
print("Starting training...")
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# lets train for all epochs
for epoch in range(trainingEpochs):
	batch = []
	for currNo in range(0, batchSize):
		r = random.randint(0, loadNum-1)
		batch.append( velocities[r] )

	_ , currentCost = sess.run([opt, cost], feed_dict={xIn: batch, y: batch})
	if epoch%10==9 or epoch==trainingEpochs-1:
		[valiCost,vout] = sess.run([cost, fc1], feed_dict={xIn: valiData, y: valiData})
		print("Epoch %d/%d: cost %f , validation cost %f " % (epoch, trainingEpochs, currentCost, valiCost) )

		if epoch==trainingEpochs-1:
			outDir = "./test_simple/ConvConvFC/"
			if not os.path.exists(outDir): os.makedirs(outDir)
			print("\n Training done. Writing %d images from validation data to directory %s..." % (len(valiData),outDir) )
			for i in range(len(valiData)):
				val_in=velocityFieldToPng(valiData[i])
				val_out=velocityFieldToPng(vout[i].reshape((64, 64,2)))
				scipy.misc.toimage( np.reshape(val_in, [64, 64, 3]) , cmin=0.0, cmax=1.0).save("%s/in_%d.png" % (outDir,i))
				scipy.misc.toimage( np.reshape(val_out, [64, 64, 3]) , cmin=0.0, cmax=1.0).save("%s/out_%d.png" % (outDir,i))

print("Done...")

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    #print(variable_parameters)
    total_parameters += variable_parameters
print("Total No. Learnable Parameters Conv & Fully Connected:",total_parameters)
############################################################
