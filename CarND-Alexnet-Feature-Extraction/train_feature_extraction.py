import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle

# TODO: Load traffic signs data.
# Load pickled data
import pickle
import numpy as np

training_file = '../CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train.p'
validation_file= '../CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/valid.p'
testing_file = '../CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/test.p'

with open(training_file, mode='br') as f:
    train = pickle.load(f)
with open(validation_file, mode='br') as f:
    valid = pickle.load(f)
with open(testing_file, mode='br') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# TODO: Split data into training and validation sets.

# TODO: Define placeholders and resize operation.

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, (227, 227))
one_hot_y = tf.one_hot(y, 43)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
mu = 0
sigma = 0.1
nb_classes = 43
x_dim = fc7.get_shape().as_list()[-1]
w = tf.Variable(tf.truncated_normal([x_dim, nb_classes], mu, sigma))
b = tf.Variable(tf.truncated_normal([nb_classes], mu, sigma))
logits = tf.add(tf.matmul(fc7, w), b)

EPOCHS = 10
BATCH_SIZE = 128

# TODO: Define loss, training, accuracy operations.
rate = 0.001
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
y_pred = tf.cast(tf.argmax(logits, 1), tf.int32)
correct_prediction = tf.equal(y_pred, y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# TODO: Train and evaluate the feature extraction model.

sample_ids = np.random.choice(X_train.shape[0], 200)
X_train_sample = X_train[sample_ids]
y_train_sample = y_train[sample_ids]

sample_ids = np.random.choice(X_valid.shape[0], 200)
X_valid_sample = X_valid[sample_ids]
y_valid_sample = y_valid[sample_ids]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_sample)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_sample, y_train_sample = shuffle(X_train_sample, y_train_sample)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_sample[offset:end], y_train_sample[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        validation_accuracy = evaluate(X_valid_sample, y_valid_sample)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './traffic_sign_sess')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
