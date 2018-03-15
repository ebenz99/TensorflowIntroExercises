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
print("Original", image.shape)
# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()
#lets x exist when this is run

mirror_mask = np.ones((height,)) * (width/2)

with tf.Session() as session:
    # Get the right half of the image
    right_side = tf.slice(x, [0,width//2,0], [-1, -1, -1])	#-1's mean just go til the end
    # Note swapped dims in the last two parameters
    mirrored = tf.reverse_sequence(right_side, mirror_mask, 1, batch_axis=0)
    # Get the left half of the image
    # Now stich them back up again
    stiched = tf.concat(axis=1, values=[mirrored, right_side])
    session.run(model)
    result = session.run(stiched)

print("Mirrored", result.shape)
plt.imshow(result)
plt.show()
