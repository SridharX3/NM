import numpy as np
import pandas as pd
import helpers

import matplotlib.pyplot as plt


def calculate_logistic_loss(X_train, y_train, W):
    loss = 0
    Z = np.array(np.dot(X_train, W), dtype=np.float32)
    h_x = 1.0 / (1.0 + np.exp(-1 * Z))
    a = np.dot(y_train.T, np.log(h_x))
    b = np.dot((1 - y_train).T, np.log(1 - h_x))
    loss = np.sum(a + b)
    
    loss =  (-1) * loss / (X_train.shape[0])
        
    return loss

def gradient_descent(W, X_train,y_train, learning_rate=0.1, iterations = 100):
    losses = []
    accuracies = []
    for k in range(iterations):
        if (k+1) % 10 == 0:
            print("Iteration %d of %d" % (k+1, iterations))        
        
        Z = np.array(np.dot(X_train, W), dtype=np.float32)
        h_x = 1.0 / (1.0 + np.exp(-1 * Z)) 

        y_train = np.array(y_train, dtype=np.float32).reshape((y_train.shape[0], 1))
        np.subtract(h_x, y_train, out=h_x)
        v = np.dot(X_train.T, h_x)
        gradient = np.sum(v)
        # print(gradient)
        W = W - (learning_rate * gradient / X_train.shape[0])

        loss = calculate_logistic_loss(X_train, y_train, W)
        accuracy = get_accuracy(X_train, y_train, W)
        losses.append(loss)
        # print(accuracy)
        accuracies.append(accuracy)

    return W, losses, accuracies

def get_accuracy(X, y, W):

    Z = np.array(np.dot(X, W), dtype=np.float32)
    h_x = 1.0 / (1.0 + np.exp(-1 * Z)) 

    accuracy = 0.0
    counter = 0
    val = 0.0

    h_x[h_x > 0.5] = 1
    h_x[h_x <= 0.5] = 0

    for i in range (0,len(h_x)):
        if int(h_x[i]) == y[i]:
            counter+=1
    accuracy = (float((counter*100))/float(len(h_x)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return accuracy


def plot_loss(losses):
    plt.xlabel("Number of iterations")
    plt.ylabel("Error")
    plt.plot(losses)
    plt.show()


def logistic_model(X_train, y_train, X_val, y_val, X_test, y_test):        

    # ones = np.ones((X_train.shape[0], 1))
    # X_train = np.hstack([ones, X_train])
    # W = np.zeros((X_train.shape[1], 1))
    W = np.random.uniform(-0.2, 0.2, size=(X_train.shape[1], 1))
    
    print("Calculating loss...")
    loss = calculate_logistic_loss(X_train, y_train, W)
    training_accuracy = get_accuracy(X_train, y_train, W)

    losses = []
    accuracies = []


    losses.append(loss)
    accuracies.append(training_accuracy)

    print("Initial loss is:", loss)
    # print("Initial accuracy is:", training_accuracy)

    print("Performing Gradient Descent")
    W, gd_losses, gd_accuracies = gradient_descent(W, X_train, y_train, iterations=80, learning_rate=0.1)
    losses += gd_losses
    accuracies += gd_accuracies

    plot_loss(losses)

    training_accuracy = get_accuracy(X_train, y_train, W)

    # ones = np.ones((X_val.shape[0], 1))
    # X_val = np.hstack([ones, X_val])
    validation_accuracy = get_accuracy(X_val, y_val, W)
    
    # ones = np.ones((X_test.shape[0], 1))
    # X_test = np.hstack([ones, X_test])
    testing_accuracy = get_accuracy(X_test, y_test, W)

    print("Accuracy on training set is: %.3f" % training_accuracy)
    print("Accuracy on validation set is: %.3f" % validation_accuracy)
    print("Accuracy on test set is: %.3f" % testing_accuracy)


if __name__ == "__main__":
    np.random.seed(42)

    X_train, y_train, X_val, y_val, X_test, y_test = helpers.get_data_split("human", "subtraction")
    # 0.1, 80 iterations

    # X_train, y_train, X_val, y_val, X_test, y_test = helpers.get_data_split("human", "concatenation")
    # 0.01, 80 iterations


    # X_train, y_train, X_val, y_val, X_test, y_test = helpers.get_data_split("gsc", "subtraction")

    # X_train, y_train, X_val, y_val, X_test, y_test = helpers.get_data_split("gsc", "concatenation")

    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape)


    logistic_model(X_train, y_train, X_val, y_val, X_test, y_test)

    # neural_network(X_train, y_train, X_val, y_val, X_test, y_test)