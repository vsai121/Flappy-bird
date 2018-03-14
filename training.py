import tensorflow as tf
import numpy as np

import flappy as game
import cv2 as cv

import random


ACTIONS = 2 #Up and down
FRAMESPERACTION=1
FINAL_EPSILON = 0.001
OBSERVE = 200000

def weight_variable(shape):

    w = tf.truncated_normal(shape=shape , stddev=0.1)
    return tf.Variable(w)

def bias_variable(shape):

    b = tf.zeros(shape=shape)
    return tf.variable(b)

def createNetwork():

    inp = tf.placeholder("float", [None, 80, 80, 4])


    W1= tf.weight_variable([8,8,4,32])
    b1 = tf.bias_variable([32])

    W2 = tf.weight_variable([4, 4, 32, 64])
    b2 = tf.bias_variable([64])

    W3 = tf.weight_variable[3, 3, 64, 64])
    b3 = tf.bias_variable([64])

    W4 = tf.weight_variable([1600, 512])
    b4 = tf.bias_variable([512])

    W5 = tf.weight_variable([512, ACTIONS])
    b5 = tf.bias_variable([ACTIONS])


    conv1 = tf.nn.relu(tf.nn.conv2d(inp, W1, strides = [1, 4, 4, 1], padding = "VALID") + b1)

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides = [1, 2, 2, 1], padding = "VALID") + b2)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W3, strides = [1, 1, 1, 1], padding = "VALID") + b3)

    conv3_flat = tf.reshape(conv3, [-1, 1600])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W4) + b4)

    fc5 = tf.matmul(fc4, W5) + b5

    return inp, fc5 , fc4


def trainNetwork(inp, out, sess):

    argmax = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None]) #True value

    action = tf.reduce_sum(tf.matmul(out, argmax), reduction_indices = 1)
    loss = tf.reduce_mean(tf.square(action - y)) #Squared error loss

    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

    state = game.GameState()

    prevObs = deque()

    no_action = np.zeros(ACTIONS)
    no_action[0]=1

    img , reward , terminal = state.frame_step(no_action)


    img = cv.cvtColor(cv.resize(img, (80, 80)), cv.COLOR_BGR2GRAY) #Processing image using OpenCV
    _ , img = cv.threshold(img , 1 , 255 , cv.THRESH_BINARY)

    imgStack = np.stack((img , img , img , img) , axis=2)

    init = tf.initialize_all_variables()
    sess.run(init)

    checkpoint = tf.train.get_checkpoint_state("saved_networks")

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded :", checkpoint.model_checkpoint_path)
    else:
        print("Unable to find network weights")

    epsilon = 0.1
    time = 0

    while True:

        out_t = out.eval(feed_dict = {inp : [imgStack]})[0])
        action_t = np.zeros([ACTIONS])

        action_index = 0

        if time % FRAMESPERACTION == 0:
            prob = random.random()

            if prob <=epsilon:
                print("Taking a random action")
                action_index = random.randrange(ACTIONS)
                action_t[action_index]=1

            else:
                action_index = np.argmax(out_t)
                action_t[action_index]=1

        else:
            action_t[0]=1


        if epsilon > FINAL_EPSILON and time > OBSERVE:
            epsilon = epsilon - (0.1 - FINAL_EPSILON) / EXPLORE
