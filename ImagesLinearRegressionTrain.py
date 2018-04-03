from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import sys
import utils

MODEL_FOLDER = "jaundice_detection-model"

learning_rate = 0.00001
training_epochs = 5000
display_step = 50

JaundiceFiles = utils.getJaundiceFilesArray()
NONJaundiceFiles = utils.getNONJaundiceFilesArray()
ColorArray = []
LabelsArray = []

print("Loading Jaundice Dataset in Memory")

for i in range(len(JaundiceFiles)):
    file = JaundiceFiles[i]
    print("Reading Yellow Intensity of Image {} out of {} images".format(i+1, len(JaundiceFiles)))
    yellowIntensity = utils.getJaundiceYellowIntensity(file)
    # print("yellowIntensity ", yellowIntensity)
    ColorArray.append(yellowIntensity)
    LabelsArray.append(1.0)

print("Succesfully Loaded Jaundice Dataset in Memory !!!")
print("Loading NON-Jaundice Dataset in Memory")

for i in range(len(NONJaundiceFiles)):
    file = NONJaundiceFiles[i]
    print("Reading Yellow Intensity of Image {} out of {} images".format(i+1, len(NONJaundiceFiles)))
    yellowIntensity = utils.getNONJaundiceYellowIntensity(file)
    # print("yellowIntensity ", yellowIntensity)
    ColorArray.append(yellowIntensity)
    LabelsArray.append(0.0)

print("Succesfully Loaded NON-Jaundice Dataset in Memory !!!")

train_X = numpy.asarray(ColorArray)
train_Y = numpy.asarray(LabelsArray)
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs in epoch step
        c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    save_path = saver.save(sess, MODEL_FOLDER+"/model.ckpt")
    print("Model saved in path: {}".format(save_path))





