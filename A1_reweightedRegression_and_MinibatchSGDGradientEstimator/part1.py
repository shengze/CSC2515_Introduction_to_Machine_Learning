from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    data_num = y.shape[0]
    data_dim = X.shape[1]
    return X,y,features,data_num,data_dim


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        #TODO: Plot feature i against y
        ax = plt.subplot(3, 5, i + 1)
        ax.set_title(features[i])
        ax.set_xlabel("Feature " + str(i+1))
        ax.set_ylabel("Target")
        ax.scatter(X[:,i],y)
    
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    #raise NotImplementedError()
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
    w, residual, rank, s = np.linalg.lstsq(X, Y)
    return w
    
def MeanSquareError(X, Y, w):
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
    y_predict = np.matmul(X, w)
    mse = sum(np.square(Y - y_predict)) / (X.shape[0])
    return mse

def main():
    # Load the data
    X, y, features, num, dim = load_data()
    print("Features: {}".format(features))
    
    print(str(num) + "  " + str(dim))
    
    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    train_data = []
    train_target = []
    test_data = []
    test_target = []
    for i in range(int(num*0.8)):
        readyToRemove = np.random.choice(len(X))
        train_data.append(X[readyToRemove])
        X = np.delete(X, (readyToRemove), axis = 0)
        train_target.append(y[readyToRemove])
        y = np.delete(y, (readyToRemove), axis = 0)
    test_data = X
    test_target = y
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_target = np.array(train_target)
    test_target = np.array(test_target)

    # Fit regression model
    w = fit_regression(train_data, train_target)
    print("Weights for Features: \n",w)

    # Compute fitted values, MSE, etc.
    mse = MeanSquareError(test_data, test_target, w)
    print("Mean Square Error: ", mse)


if __name__ == "__main__":
    main()