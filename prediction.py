from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import sys
import utils

MODEL_FOLDER = "jaundice_detection-model"


yellowIntensity = utils.getYellowIntensityWithFileName('recentClick.jpg')
test_X = numpy.asarray([yellowIntensity])

# tf Graph Input
X = tf.placeholder("float")

# Set model weights
W = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Start training
with tf.Session() as sess:
    saver.restore(sess, MODEL_FOLDER+"/model.ckpt")
    print("Model restored")
    print("Predicting...")
    classification = sess.run(pred, feed_dict={X: test_X})
    print(classification)





