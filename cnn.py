
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
train = pd.read_json("train.json")
#test = pd.read_json("test.json")


# In[2]:


X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])


# In[3]:


X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],
                          ((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
target_train=train['is_iceberg']


# In[4]:


from sklearn.model_selection import train_test_split
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, target_train, random_state=1, train_size=0.75)


# In[5]:


y_valid.shape


# In[6]:


x = tf.placeholder(tf.float32, shape=(None, 75, 75, 3), name = "x")
y_ = tf.placeholder(tf.int64, shape=(None), name = "y_")


# In[29]:


he_init = tf.contrib.layers.variance_scaling_initializer()
conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding = "valid")
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding = "valid")
#pool2_flat = tf.reshape(pool2, [-1, 18 * 18 * 64])
pool2_flat = tf.contrib.layers.flatten(pool2)
dense1 = tf.layers.dense(inputs=pool2_flat, units=300, activation=tf.nn.elu)
dense2 = tf.layers.dense(inputs=dense1, units=100, activation=tf.nn.elu)
dropout = tf.layers.dropout(inputs=dense2, rate=0.4)
y_conv = tf.layers.dense(inputs=dropout, units=2) #activation=tf.nn.sigmoid)


# In[30]:


cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#cross_entropy = tf.reduce_mean(tf.keras.backend.binary_crossentropy(y_, y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
correct_prediction = tf.nn.in_top_k(y_conv, y_, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[31]:


init = tf.global_variables_initializer()

with tf.Session() as sess: 
    init.run()
    for i in range(200):
        sess.run(train_step, feed_dict={x:X_train_cv, y_: y_train_cv})
        train_accuracy = accuracy.eval(feed_dict={x:X_train_cv, y_: y_train_cv})
        valid_accuracy = accuracy.eval(feed_dict={x:X_valid, y_: y_valid})
        print("step %d, training accuracy %g, valid accuracy %g"%(i, train_accuracy, valid_accuracy))




