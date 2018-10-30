import numpy as np
import pandas as pd
import tensorflow as tf

def get_train_val_test_split(X):
    print("Splitting data into training, validation and test sets")

    X = np.array([np.append(elm, elm[0][:4]) for elm in X])

    unique_writers = np.unique(X[:,-1])
    
    unique_writers = np.random.permutation(unique_writers)

    train_split_idx = int(0.8 * unique_writers.shape[0])
    train_writers = unique_writers[:train_split_idx]


    X_train = X[np.isin(X[:, -1], train_writers)]
    y_train = X_train[:, -2]
    X_train = X_train[:, 2:-2]
    

    val_split_idx = train_split_idx + int(0.1 * unique_writers.shape[0])
    val_writers = unique_writers[train_split_idx: val_split_idx]
    
    X_val = X[np.isin(X[:, -1], val_writers)]
    y_val = X_val[:, -2]
    X_val = X_val[:, 2:-2]

    test_writers = unique_writers[val_split_idx:]
    X_test = X[np.isin(X[:, -1], test_writers)]
    y_test = X_test[:, -2]
    X_test = X_test[:, 2:-2]
    
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_data_split(dataset_type, feature_extraction_type):

    print("Reading data from ./%s_%s_processed.csv" % (dataset_type, feature_extraction_type))
    input_data = pd.read_csv("./%s_%s_processed.csv" % (dataset_type, feature_extraction_type), index_col=False)

    input_data = np.array(input_data)

    return get_train_val_test_split(input_data)

def calculate_logistic_loss(X_train, y_train, W):
    loss = 0
    Z = np.array(np.dot(X_train, W), dtype=np.float32)
    h_x = 1.0 / (1.0 + np.exp(-1 * Z))
    a = np.dot(y_train.T, np.log(h_x))
    b = np.dot((1 - y_train).T, np.log(1 - h_x))
    loss = np.sum(a + b)
    
    loss =  (-1) * loss / (X_train.shape[0])
        
    return loss

def gradient_descent(W, X_train,y_train, learning_rate=0.1):
    iterations = 500
    for k in range(iterations):
        # if (k+1) % 10 == 0:
        #     print("Iteration %d of %d" % (k+1, iterations))        
        
        Z = np.array(np.dot(X_train, W), dtype=np.float32)
        h_x = 1.0 / (1.0 + np.exp(-1 * Z)) 

        y_train = np.array(y_train, dtype=np.float32).reshape((y_train.shape[0], 1))
        np.subtract(h_x, y_train, out=h_x)
        v = np.dot(X_train.T, h_x)
        gradient = np.sum(v)
        # print(gradient)
        W = W - (learning_rate * gradient / X_train.shape[0])
    return W

def get_accuracy(X, y, W):
    diff = 0

    Z = np.array(np.dot(X, W), dtype=np.float32)
    h_x = 1.0 / (1.0 + np.exp(-1 * Z)) 

    h_x[h_x >= 0.5] = 1
    h_x[h_x < 0.5] = 0


    y = np.array(y, dtype=np.float32).reshape((y.shape[0], 1))
    np.abs(np.subtract(h_x, y, out=h_x), out=h_x)

    diff = np.sum(h_x)

    accuracy = 1 - (diff / X.shape[0])
    return accuracy

def logistic_model(X_train, y_train, X_val, y_val, X_test, y_test):        


    ones = np.ones((X_train.shape[0], 1))
    # X_train = np.hstack([ones, X_train])
    # W = np.zeros((X_train.shape[1], 1))
    W = np.random.uniform(-0.2, 0.2, size=(X_train.shape[1], 1))
    
    print("Calculating loss...")
    loss = calculate_logistic_loss(X_train, y_train, W)
    print("Loss is:", loss)

    print("Performing Gradient Descent")
    W = gradient_descent(W, X_train, y_train)

    training_accuracy = get_accuracy(X_train, y_train, W)

    ones = np.ones((X_val.shape[0], 1))
    # X_val = np.hstack([ones, X_val])
    validation_accuracy = get_accuracy(X_val, y_val, W)
    
    ones = np.ones((X_test.shape[0], 1))
    # X_test = np.hstack([ones, X_test])
    testing_accuracy = get_accuracy(X_test, y_test, W)

    print("Accuracy on training set is: %.3f" % training_accuracy)
    print("Accuracy on validation set is: %.3f" % validation_accuracy)
    print("Accuracy on test set is: %.3f" % testing_accuracy)

def randomize(x, y):
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def weight_variable(name, shape):
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(name, shape):
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)
 
def fc_layer(x, num_units, name, use_relu):

    in_dim = x.get_shape()[1]
    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])
    layer = tf.matmul(x, W)
    layer += b
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

def neural_network(X_train, y_train, X_val, y_val, X_test, y_test):
    
    n_classes = 1
    h1 = 10
    
    epochs = 10             # Total number of training epochs
    batch_size = 100        # Training batch size
    display_freq = 100      # Frequency of displaying the training results
    learning_rate = 0.001

    X = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='X')
    y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

    fc1 = fc_layer(X, h1, 'FC1', use_relu=True)
    output_logits = fc_layer(fc1, n_classes, 'OUT', use_relu=False)
    
    cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
    correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)
    global_step = 0
    # Number of training iterations in each epoch
    num_tr_iter = int(len(y_train) / batch_size)
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))
        X_train, y_train = randomize(X_train, y_train)
        y_train = y_train.reshape((y_train.shape[0], 1))
        for iteration in range(num_tr_iter):
            global_step += 1
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(X_train, y_train, start, end)

            # Run optimization op (backprop)
            feed_dict_batch = {X: x_batch, y: y_batch}
            sess.run(optimizer, feed_dict=feed_dict_batch)

            if iteration % display_freq == 0:
                # Calculate and display the batch loss and accuracy
                loss_batch, acc_batch = sess.run([loss, accuracy],
                                                feed_dict=feed_dict_batch)

                print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                    format(iteration, loss_batch, acc_batch))

        # Run validation after every epoch
        y_val = y_val.reshape((y_val.shape[0], 1))
        feed_dict_valid = {X: X_val[:1000], y: y_val[:1000]}
        loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
            format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')

    y_test = y_test.reshape((y_test.shape[0], 1))
    feed_dict_test = {X: X_test[:1000], y: y_test[:1000]}
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_valid)

    print("Testing accuracy is %d percent" % (acc_test*100))


if __name__ == "__main__":

    X_train, y_train, X_val, y_val, X_test, y_test = get_data_split("human", "addition")

    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape)


    logistic_model(X_train, y_train, X_val, y_val, X_test, y_test)

    # neural_network(X_train, y_train, X_val, y_val, X_test, y_test)