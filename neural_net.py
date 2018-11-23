import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
import pandas as pd
from keras.utils import np_utils
import helpers

# Initializing the weights to Normal Distribution
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01, seed=42))

def create_model(activation_function, optimizer):
    # Initializing the input to hidden layer weights
    input_hidden_weights  = init_weights([9, NUM_HIDDEN_NEURONS_LAYER_1])
    # Initializing the hidden to output layer weights
    hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 2])

    # Computing values at the hidden layer
    hidden_layer = None
    if activation_function == "relu":
        hidden_layer = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))
    elif activation_function == "tanh":
        hidden_layer = tf.nn.tanh(tf.matmul(inputTensor, input_hidden_weights))
    elif activation_function == "sigmoid":
        hidden_layer = tf.nn.sigmoid(tf.matmul(inputTensor, input_hidden_weights))
    # Computing values at the output layer
    output_layer = tf.matmul(hidden_layer, hidden_output_weights)

    # Defining Error Function
    error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

    # Defining Learning Algorithm and Training Parameters
    training = None
    if optimizer == "gradient_descent":
        training = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(error_function)
    elif optimizer == "gradient_descent_with_momentum":
        training = tf.train.MomentumOptimizer(learning_rate=learning_rate,
            momentum=0.9
        ).minimize(error_function)
    elif optimizer == "adam":
        training = tf.train.AdamOptimizer(learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
        ).minimize(error_function)
    elif optimizer == "rmsprop":
        training = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
            decay=0.9,
            epsilon=1e-10
        ).minimize(error_function)

    # Prediction Function
    prediction = tf.argmax(output_layer, 1)

    return training, prediction


# # Training the Model

def get_accuracy(predicted_test_labels, test_data_labels):
    counter = 0
    for i in range (0,len(predicted_test_labels)):
        predicted_test_labels[i] = 0 if int(predicted_test_labels[i]) < 0.5 else 1
        if(int(predicted_test_labels[i]) == test_data_labels[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(predicted_test_labels)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return accuracy

def run_model(learning_rates, processedTrainingData, processedTrainingLabel, processedValidationData, processedValidationLabel, processedTestingData, processedTestingLabel, num_of_epochs=5000, batch_size=128):
    
    training_accuracies = []
    predicted_test_labels = []
    predicted_val_labels = []
    with tf.Session() as sess:

        # Set Global Variables ?
        tf.global_variables_initializer().run()
        for lr in learning_rates:
            accuracy = []
            for epoch in tqdm_notebook(range(num_of_epochs)):
                if (epoch+1) % 100 == 0:
                    print("Epoch %d" % (epoch+1))
                #Shuffle the Training Dataset at each epoch
                p = np.random.permutation(range(len(processedTrainingData)))
                processedTrainingData  = processedTrainingData[p]
                processedTrainingLabel = processedTrainingLabel[p].reshape(processedTrainingLabel.shape[0], 1)

                # Start batch training
                for start in range(0, len(processedTrainingData), batch_size):
                    end = start + batch_size
                    sess.run(training, feed_dict={inputTensor: processedTrainingData[start:end], 
                                              outputTensor: processedTrainingLabel[start:end],
                                                 learning_rate: lr})
                    # Training accuracy for an epoch
                accuracy.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==
                                     sess.run(prediction, feed_dict={inputTensor: processedTrainingData,
                                                                         outputTensor: processedTrainingLabel})))
            training_accuracies.append(tuple(accuracy))
            # Validation
            processedValidationLabel = processedValidationLabel.reshape((processedValidationLabel.shape[0], 1))
            processedTestingLabel = processedTestingLabel.reshape((processedTestingLabel.shape[0], 1))

            val_accuracy = np.mean(np.argmax(processedValidationLabel, axis=1) ==
                                     sess.run(prediction, feed_dict={inputTensor: processedValidationData,
                                                                         outputTensor: processedValidationLabel}))
            # Testing
            test_accuracy = np.mean(np.argmax(processedTestingLabel, axis=1) ==
                                     sess.run(prediction, feed_dict={inputTensor: processedTestingData,
                                                                         outputTensor: processedTestingLabel}))

    return training_accuracies, predicted_test_labels, val_accuracy, test_accuracy


def neural_network(X_train, y_train, X_val, y_val, X_test, y_test):

    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape)    


    global NUM_HIDDEN_NEURONS_LAYER_1, learning_rates, num_of_epochs, batch_size, processedTrainingData, processedTrainingLabel, processedValidationData, processedValidationLabel, processedTestingData, processedTestingLabel, inputTensor, outputTensor, learning_rate, training, prediction

    NUM_HIDDEN_NEURONS_LAYER_1 = 100
    learning_rates = [0.001] #[0.001, 0.1]
    num_of_epochs = 5000
    batch_size = 128

    inputTensor  = tf.placeholder(tf.float32, [None, 9])
    outputTensor = tf.placeholder(tf.float32, [None, 1])
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    # Process Dataset
    processedTrainingData, processedTrainingLabel = X_train, y_train
    processedValidationData, processedValidationLabel = X_val, y_val
    processedTestingData, processedTestingLabel = X_test, y_test




    training, prediction = create_model(activation_function="relu", optimizer="rmsprop")
    training_accuracies, predicted_test_labels, val_accuracy, test_accuracy = run_model(learning_rates, processedTrainingData, processedTrainingLabel, processedValidationData, processedValidationLabel, processedTestingData, processedTestingLabel, num_of_epochs, batch_size)


    i = 0
    for training_accuracy, learning_rate in zip(training_accuracies, learning_rates):
        #print(len(training_accuracy))
        df = pd.DataFrame()
        df['acc'] = training_accuracy
        ax = df.plot(grid=True, title="learning rate %s" % learning_rate)
        fig = ax.get_figure()
        fig.savefig("Trained_with_%s_5000.png" % str(learning_rate))
        i += 1
        print("Training Accuracy for learning rate %s is: %s" % (learning_rate, training_accuracy[len(training_accuracy) - 1] * 100 ))
        print("Validation accuracy for learning rate %s is: %s" % (learning_rate, val_accuracy*100))
        print("Testing accuracy for learning rate %s is: %s" % (learning_rate,test_accuracy*100))

