import tensorflow as tf
import numpy as np      #for random number generation

s = tf.Variable(0, name='s')
avg = tf.Variable(0, name='avg')
num = np.random.randint(100, size=5)	#random number between 0 and 99 in each of five array spots

model = tf.global_variables_initializer() 	#maps relationships of variables

with tf.Session() as session:
    session.run(model)
    for i in range(5):
        s = s + num[i]		
        avg = s / (i+1)
    print(session.run(s))	#prints sum and average. Calling session.run on each variable yields their actual values to print
    print(session.run(avg))	
    #print(avg) 				a call to this would just display tensor information, not the value
