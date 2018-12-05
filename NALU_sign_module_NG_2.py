import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd

# Simple Neural Accumulator (NAC) for +/-
def nac_simple_single_layer(x_in, W_hat, M_hat):
    '''
    Define a Neural Accumulator (NAC) for addition/subtraction -> Useful to learn the addition/subtraction operation
    :param x_in -> Input vector
    :param W_hat -> Weight matrices
    :param M_hat -> "      "
    :return Output tensor of mentioned shape & associated weights
    '''

    # Get W
    W = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)

    y_out = tf.matmul(x_in, W)
    print("nac simple shape: ", tf.shape(y_out), " | ", y_out.shape)

    return y_out, W

# Complex NAC using Log Space for x,/,^
def nac_complex_single_layer(x_in, W_hat, M_hat, epsilon=0.000001):
    '''
    :param x_in: input feature vector
    :param W_hat -> Weight matrices
    :param M_hat -> "      "
    :param epsilon: small value to avoid log(0) in the output result
    :return: output tensor & associated weight matrix
    '''

    in_shape = x_in.shape[1]

    W = tf.nn.tanh(W_hat) * tf.nn.sigmoid(M_hat)

    # Express Input feature in log space
    x_modified = tf.log(tf.abs(x_in) + epsilon)
    # x_modified = tf.asinh(x_in + epsilon)

    m = tf.exp(tf.matmul(x_modified, W))

    print("nac complex shape: ", tf.shape(m), " | ", m.shape)

    return m, W

# Sign Module
def sign_module(x_in):
    '''
    :param x_in: input feature vector
    :return: output tensor, should just be a scalar, -1 or 1
    '''
    in_shape = x_in.shape[1]

    # Weights to learn how to extract sign features
    # S_hat = tf.get_variable(name="S_hat", initializer=tf.initializers.random_uniform(minval=-2, maxval=2),
    #                         shape=[in_shape, in_shape], trainable=True)
    S_hat = tf.get_variable(name="S_hat", initializer=tf.initializers.random_uniform(minval=0, maxval=2),
                            shape=[1, in_shape], trainable=True)

    # Theoretically grabs the sign
    temp = 2 * tf.nn.sigmoid(x_in * S_hat) - 1

    print("temp is: ", tf.shape(temp), " | ", temp.shape)

    sign = tf.reduce_prod(temp, axis=1)

    c = tf.get_variable(name="sign_multiplier", initializer=tf.initializers.constant(value=10), shape=[1],
                        trainable=True)

    sign = 2 * tf.sigmoid(c * sign) - 1

    print("sign is: ", sign.shape)

    return sign, S_hat, c

# NALU, combining simple NAC and complex NAC
def nalu(x_in, out_units, epsilon=0.000001, get_weights=True):
    '''
    :param x_in: input feature vector
    :param out_units: number of output units of the cell
    :param epsilon: small value to avoid log(0) in the output result
    :param get_weights: True if want to get the weights of the model
                        in return
    :return: output tensor
    :return: Gate weight matrix
    :return: NAC1 (simple NAC) weight matrix
    :return: NAC2 (complex NAC) weight matrix, Should be identical to NAC1 b/c they're shared
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
    m, W_complex = nac_complex_single_layer(x_in, W_hat, M_hat, epsilon=epsilon)

    s, s_weight, c_weight = sign_module(x_in)
    s = tf.expand_dims(s, 1)

    # Sign Gate
    # S_G = tf.get_variable(initializer=tf.random_normal_initializer(stddev=1.0),
    #                     shape=[in_shape, out_units], name="Sign_gate_weights", trainable=True)

    # Gate signal layer
    G = tf.get_variable(initializer=tf.random_normal_initializer(stddev=1.0),
                        shape=[in_shape, out_units], name="Gate_weights", trainable=True)

    # s_g = tf.nn.sigmoid( tf.matmul(x_in,S_G) )
    g = tf.nn.sigmoid(tf.matmul(x_in, G))

    print("a is:", tf.shape(a), a.shape)
    print("m is:", tf.shape(m), m.shape)
    print("s is:", tf.shape(s), s.shape)
    temporary = tf.multiply(m, s, 'factor_in_sign')
    print("the test: ", tf.shape(temporary), temporary.shape)
    y_out = g * a + (1 - g) * m * s
    print("y_out is:", tf.shape(y_out), y_out.shape)
    if (get_weights):
        return y_out, G, W_simple, W_complex, s_weight, c_weight, g
    else:
        return y_out


# Test the Network by learning the addition

# Generate a series of input number X1,X2 and X3 for training
x1 = np.arange(-250, 250, step=2, dtype=np.float32)
x2 = np.arange(1, 251, step=1, dtype=np.float32)
# x2 = -1*np.abs(x1)+125  # [-125 to 125 to -125]
# x1_more = np.arange(-250, 250, step=1, dtype=np.float32)
# x2_more = np.abs(x1_more)-125
# x1 = np.concatenate((x1,x1_more))
# x2 = np.concatenate((x2,x2_more))

x3 = np.arange(0, 500, step=1, dtype=np.float32)

# Make any function of x1,x2 and x3 to try the network on
# y_train = (x1/4) + (x2/2) + x3**2
y_train = x1 * x2  # + x3

x_train = np.column_stack((x1, x2))
x_train, y_train = shuffle(x_train, y_train, random_state=0)

# print(x_train.shape)
# print(y_train.shape)

# Generate a series of input number X1,X2 and X3 for testing
x1 = np.random.randint(-1000, 1000, size=200).astype(np.float32)
# x2 = np.random.randint(-1000, 1000, size=200).astype(np.float32)
x2 = np.random.randint(250, 1000, size=200).astype(np.float32)
x3 = np.random.randint(0, 2000, size=200).astype(np.float32)

x_test = np.column_stack((x1, x2))

# y_test = (x1/4) + (x2/2) + x3**2
y_test = x1 * x2  # + x3

print()
# print(x_test.shape)
# print(y_test.shape)

# ===== Build Model =====

# Define the placeholder to feed the value at run time
X = tf.placeholder(dtype=tf.float32,
                   shape=[None, 2])  # Number of samples x Number of features (number of inputs to be added)
Y = tf.placeholder(dtype=tf.float32, shape=[None, ])

# define the network
# Here the network contains only one NAC cell (for testing)
y_pred, G, weight_simp, weight_comp, s_w, c_w, g = nalu(X, out_units=1, get_weights=True)
# y_pred = nalu(X, out_units=1)  # Remove extra dimensions if any
y_pred = tf.squeeze(y_pred)

# Mean Square Error (MSE)
loss = tf.reduce_mean((y_pred - Y) ** 2)
# loss= tf.losses.mean_squared_error(labels=y_train, predictions=y_pred)

# training parameters
alpha = 0.1  # learning rate
epochs = 30000

optimize = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # pre training evaluate
    print("Pre training MSE: ", sess.run(loss, feed_dict={X: x_test, Y: y_test}))
    print()

    cost_history = []
    G_history = []
    g_history = []
    ws_history = []
    wc_history = []
    c_w_array = []
    s_w_history = []

    for i in range(epochs):
        _, cost, G_value, weight_simple, weight_complex, s_weight, c_weight, g_value = sess.run([optimize, loss, G, weight_simp, weight_comp, s_w, c_w, g],
                                                                   feed_dict={X: x_train, Y: y_train})
        print("epoch: {}, MSE: {}".format(i, cost))
        cost_history.append(cost)
        G_history.append(np.squeeze(G_value))
        g_history.append(g_value[0])
        ws_history.append(np.squeeze(weight_simple))
        wc_history.append(np.squeeze(weight_complex))
        s_w_history.append(np.squeeze(s_weight))
        c_w_array.append(c_weight[0])

    s_w_history = np.transpose(s_w_history)
    ws_history = np.transpose(ws_history)
    wc_history = np.transpose(wc_history)
    G_history = np.transpose(G_history)

    # plot the MSE over each iteration
    fig1 = plt.figure()
    plt.plot(np.arange(epochs), np.log(cost_history))  # Plot MSE on log scale
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Cost History")
    fig1.show()

    fig2 = plt.figure()
    plt.plot(np.arange(epochs), G_history[0], '--b', np.arange(epochs), G_history[1], '--g', np.arange(epochs), g_history, '-r')
    plt.xlabel("Epoch")
    plt.ylabel("Gate value")
    plt.title("Gate History (1 = Addition, 0 = Multiplication)")
    fig2.show()

    fig3 = plt.figure()
    plt.plot(np.arange(epochs), c_w_array)
    plt.xlabel("Epoch")
    plt.ylabel("Scaler value")
    plt.title("Scaler History")
    fig3.show()

    fig4 = plt.figure()
    plt.plot(np.arange(epochs), s_w_history[0],np.arange(epochs), s_w_history[1])
    plt.xlabel("Epoch")
    plt.ylabel("Sign weight value")
    plt.title("Sign weight History")
    fig4.show()

    fig5 = plt.figure()
    plt.plot(np.arange(epochs), ws_history[0], np.arange(epochs), ws_history[1])
    plt.xlabel("Epoch")
    plt.ylabel("Simple NAC Weight Value")
    plt.title("Simple NAC Weights - History")
    fig5.show()

    fig5a = plt.figure()
    plt.plot(np.arange(epochs), wc_history[0], np.arange(epochs), wc_history[1])
    plt.xlabel("Epoch")
    plt.ylabel("Complex NAC Weight Value")
    plt.title("Complex NAC Weights - History")
    fig5a.show()

    print()
    # print(W.eval())
    # print()
    # post training loss
    print("Post training MSE: ", sess.run(loss, feed_dict={X: x_test, Y: y_test}))

    # print("Actual sum: ", y_test[0:10])
    print()
    y_hat = sess.run(y_pred, feed_dict={X: x_test, Y: y_test})

    temp = np.transpose(np.vstack((y_test,y_hat)))
    temp = np.concatenate((x_test,temp),axis=1)
    compare_y = pd.DataFrame(temp)
    compare_y.columns= ['x1','x2','y_test','y_pred']
    compare_y.to_csv("debug.csv")

    # print("Predicted sum: ", y_hat[0:10])

    print("Hello.")
