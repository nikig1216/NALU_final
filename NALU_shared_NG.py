import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# Simple Neural Accumulator (NAC) for +/-
def nac_simple_single_layer(x_in, W_hat, M_hat):
    '''
    Define a Neural Accumulator (NAC) for addition/subtraction -> Useful to learn the addition/subtraction operation
    Attributes:
        x_in -> Input vector
        out_units -> number of output neurons
    Return:
        Output tensor of mentioned shsape and associated weights
    '''

    in_features = x_in.shape[1]

    # Get W
    W = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)

    y_out = tf.matmul(x_in,W)

    return y_out,W

# Complex NAC using Log Space for x,/,^

def nac_complex_single_layer(x_in, W_hat, M_hat, epsilon = 0.000001):
    '''
    :param x_in: input feature vector
    :param out_units: number of output units of the cell
    :param epsilon: small value to avoid log(0) in the output result
    :return: associated weight matrix and output tensor
    '''

    in_shape = x_in.shape[1]

    W = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)

    # Express Input feature in log space
    x_modified = tf.log(tf.abs(x_in) + epsilon)
    # x_modified = tf.asinh(x_in + epsilon)

    m = tf.exp( tf.matmul(x_modified, W) )

    return m, W

# NALU, combining simple NAC and complex NAC
def nalu(x_in, out_units, epsilon=0.000001, get_weights=False):
    '''
    :param x_in: input feature vector
    :param out_units: number of output units of the cell
    :param epsilon: small value to avoid log(0) in the output result
    :param get_weights: True if want to get the weights of the model
                        in return
    :return: output tensor
    :return: Gate weight matrix
    :return: NAC1 (simple NAC) weight matrix
    :return: NAC2 (complex NAC) weight matrix
    '''

    in_shape = x_in.shape[1]

    # define W_hat and M_hat
    W_hat = tf.get_variable(name="W_hat", initializer=tf.initializers.random_uniform(minval=-2, maxval=2),
                            shape=[in_shape, out_units], trainable=True)
    M_hat = tf.get_variable(name="M_hat", initializer=tf.initializers.random_uniform(minval=-2, maxval=2),
                            shape=[in_shape, out_units], trainable=True)

    # Get output of simple NAC
    a, W_simple = nac_simple_single_layer(x_in, W_hat, M_hat)

    # Get output of complex NAC
    m, W_complex = nac_complex_single_layer(x_in, W_hat, M_hat, epsilon= epsilon)

    # Gate signal layer
    G = tf.get_variable(initializer=tf.random_normal_initializer(stddev=1.0),
                        shape=[in_shape, out_units], name="Gate_weights", trainable=True)

    g = tf.nn.sigmoid( tf.matmul(x_in, G) )

    y_out = g * a + (1 - g) * m

    if(get_weights):
        return y_out, G, W_simple, W_complex
    else:
        return y_out


# Test the Network by learning the addition

# Generate a series of input number X1,X2 and X3 for training
# x1 =  np.arange(1000,11000, step=5, dtype= np.float32)
# x2 =  np.arange(500, 6500 , step=3, dtype= np.float32)
# x3 = np.arange(0, 2000, step = 1, dtype= np.float32)
x1 =  np.arange(500, 1500, step=2, dtype= np.float32)
x2 =  np.arange(1, 501 , step=1, dtype= np.float32)
x3 = np.arange(0, 500, step = 1, dtype= np.float32)

# Make any function of x1,x2 and x3 to try the network on
# y_train = (x1/4) + (x2/2) + x3**2
y_train = x1 * x2 #+ x3

x_train = np.column_stack((x1,x2))

# print(x_train.shape)
# print(y_train.shape)

# Generate a series of input number X1,X2 and X3 for testing
# x1 = np.random.randint(0,1000, size= 200).astype(np.float32)
# x2 = np.random.randint(1, 500, size=200).astype(np.float32)
# x3 = np.random.randint(50, 150 , size=200).astype(np.float32)
x1 = np.random.randint(1500,11000, size=200).astype(np.float32)
x2 = np.random.randint(500, 6500, size=200).astype(np.float32)
x3 = np.random.randint(0, 2000 , size=200).astype(np.float32)

x_test = np.column_stack((x1,x2))

# y_test = (x1/4) + (x2/2) + x3**2

y_test = x1 * x2 #+ x3

print()
# print(x_test.shape)
# print(y_test.shape)

# ===== Build Model =====

# Define the placeholder to feed the value at run time
X = tf.placeholder(dtype=tf.float32, shape =[None , 2])    # Number of samples x Number of features (number of inputs to be added)
Y = tf.placeholder(dtype=tf.float32, shape=[None,])

# define the network
# Here the network contains only one NAC cell (for testing)
y_pred, G, weight_simp, weight_comp = nalu(X, out_units=1, get_weights=True)
y_pred = tf.squeeze(y_pred)             # Remove extra dimensions if any

# Mean Square Error (MSE)
loss = tf.reduce_mean( (y_pred - Y) **2)
#loss= tf.losses.mean_squared_error(labels=y_train, predictions=y_pred)



# training parameters
alpha = 0.01 # learning rate
epochs = 30000


optimize = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # pre training evaluate
    print("Pre training MSE: ", sess.run (loss, feed_dict={X: x_test, Y:y_test}))
    print()
    cost_history = []
    G_history = []
    ws_history = []
    wc_history = []

    for i in range(epochs):
        _, cost, G_value, weight_simple, weight_complex = sess.run([optimize, loss, G, weight_simp, weight_comp], feed_dict={X: x_train, Y: y_train})
        print("epoch: {}, MSE: {}".format(i, cost))
        cost_history.append(cost)
        G_history.append(G_value)
        ws_history.append(weight_simple)
        wc_history.append(weight_complex)

    # plot the MSE over each iteration
    plt.plot(np.arange(epochs),np.log(cost_history))  # Plot MSE on log scale
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()

    print()
    #print(W.eval())
    #print()
    # post training loss
    print("Post training MSE: ", sess.run(loss, feed_dict={X: x_test, Y: y_test}))

    print("Actual sum: ", y_test[0:10])
    print()
    y_hat = sess.run(y_pred, feed_dict={X: x_test, Y: y_test})
    print("Predicted sum: ", y_hat[0:10] )


    print("Hello.")
