import numpy as np
import linear_reg
import logistic_model
import neural_net
import helpers

if __name__ == "__main__":
    np.random.seed(42)

    X_train, y_train, X_val, y_val, X_test, y_test = helpers.get_data_split("human", "subtraction")
    # X_train, y_train, X_val, y_val, X_test, y_test = helpers.get_data_split("human", "concatenation")
    # X_train, y_train, X_val, y_val, X_test, y_test = helpers.get_data_split("gsc", "subtraction")
    # X_train, y_train, X_val, y_val, X_test, y_test = helpers.get_data_split("gsc", "concatenation")

    linear_reg.linear_model(X_train, y_train, X_val, y_val, X_test, y_test)

    logistic_model.logistic_model(X_train, y_train, X_val, y_val, X_test, y_test)

    neural_net.neural_network(X_train, y_train, X_val, y_val, X_test, y_test)