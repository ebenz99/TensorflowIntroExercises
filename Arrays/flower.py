import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
# First, load the image again
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"
image = mpimg.imread(filename)
height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
	x = tf.transpose(x, perm=[1, 0, 2])
	#swap axis 1 and 0, axis 2 stays where it is
	x = tf.reverse_sequence(x, seq_lengths = [height] * width, batch_axis=0, seq_axis=1)		
	#[height] value is 1, but necessary to give tensor rank 1 (a vector) as opposed to width's scalar rank 0
	#batch axis is parallel to what's getting sliced, so seq lengths are width of batch axis
	#seq_axis is perpendicular to what's getting slice, so to reverse whole thing want to go up to height
	#axis 0 is x, 1 is y
	session.run(model)
	result = session.run(x)

print(result.shape)	#result is the image
plt.imshow(result)
plt.show()
