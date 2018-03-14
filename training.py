import tensorflow as tf
import numpy as np
import sys
sys.path.append('game/')
import flappy as game
import cv2 as cv

import random
from collections import deque


ACTIONS = 2 #Up and down
FRAMESPERACTION=1
FINAL_EPSILON = 0.001
OBSERVE = 10000
EXPLORE = 200000
REPLAY_MEM = 5000
GAMMA = 0.99
GAME='bird'
EXPLORE

def weight_variable(shape):

    w = tf.truncated_normal(shape=shape , stddev=0.1)
    return tf.Variable(w)

def bias_variable(shape):

    b = tf.zeros(shape=shape)
    return tf.Variable(b)

def createNetwork():

    inp = tf.placeholder("float", [None, 80, 80, 4])


    W1= weight_variable([8,8,4,32])
    b1 = bias_variable([32])

    W2 = weight_variable([4, 4, 32, 64])
    b2 = bias_variable([64])

    W3 = weight_variable([3, 3, 64, 64])
    b3 = bias_variable([64])

    W4 = weight_variable([1600, 512])
    b4 = bias_variable([512])

    W5 = weight_variable([512, ACTIONS])
    b5 = bias_variable([ACTIONS])


    conv1 = tf.nn.relu(tf.nn.conv2d(inp, W1, strides = [1, 4, 4, 1], padding = "SAME") + b1)
    conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides = [1, 2, 2, 1], padding = "SAME") + b2)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W3, strides = [1, 1, 1, 1], padding = "SAME") + b3)

    conv3_flat = tf.reshape(conv3, [-1, 1600])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W4) + b4)

    fc5 = tf.matmul(fc4, W5) + b5

    return inp, fc5 , fc4


def trainNetwork(inp, out, sess):

    argmax = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None]) #True value

    action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices = 1)
    loss = tf.reduce_mean(tf.square(action - y)) #Squared error loss

    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

    state = game.GameState()

    prevObs = deque()

    no_action = np.zeros(ACTIONS)
    no_action[0]=1

    img , reward , terminal = state.frame_step(no_action)


    img = cv.cvtColor(cv.resize(img, (80, 80)), cv.COLOR_BGR2GRAY) #Processing image using OpenCV
    _ , img = cv.threshold(img , 1 , 255 , cv.THRESH_BINARY)

    imgStack = np.stack((img , img , img , img,) , axis=2)
    print(imgStack.shape)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
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

        out_t = out.eval(feed_dict = {inp : [imgStack]})[0]
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

        nextImg , nextReward , terminal  = state.frame_step(action_t)

        nextImg = cv.cvtColor(cv.resize(nextImg, (80, 80)), cv.COLOR_BGR2GRAY) #Processing image using OpenCV
        _ , nextImg = cv.threshold(nextImg , 1 , 255 , cv.THRESH_BINARY)

        nextImg = np.reshape(nextImg , (80 , 80 , 1))

        nextImgStack = np.append(nextImg , imgStack[:,:,:3] , axis=2)

        prevObs.append((imgStack , action_t , reward , nextImgStack , terminal))

        if len(prevObs)> REPLAY_MEM:
            prevObs.popleft()

        if time>OBSERVE:

            minibatch = random.sample(prevObs , 32)

            curobs = np.zeros((32 , imgStack.shape[0] , imgStack.shape[1] , imgStack.shape[2]))
            curaction=[]
            curreward=[]
            nextobs = np.zeros((32 , imgStack.shape[0] , imgStack.shape[1] , imgStack.shape[2]))

            ybatch =[]
            i = 0
            for obs in minibatch:

                curobs[i , : , : , :] = obs[0]
                curaction.append(obs[1])
                curreward.append(obs[2])
                nextobs[i , : , : , :] = obs[3]
                i = i+1

            outbatch = out.eval(feed_dict = {inp :  nextobs})

            for i in range(len(minibatch)):

                terminal = minibatch[i][4]

                if terminal:
                    ybatch.append(curreward[i])

                else:
                     ybatch.append(curreward[i] + GAMMA * np.max(outbatch[i]))



            #print(curaction)
            train_step.run(feed_dict = {
                y : ybatch,
                argmax: curaction,
                inp : curobs})

        img = nextImg
        time+=1

        if time % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = time)


        s = ""

        if time <= OBSERVE:
            s = "observe"
        elif time > OBSERVE and time <= OBSERVE + EXPLORE:
            s = "explore"
        else:
            s = "train"

        print("TIMESTEP", time, "/ STATE", s, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward, \
            "/ Q_MAX %e" % np.max(out_t))



def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
