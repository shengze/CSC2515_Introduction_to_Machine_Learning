from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        print("        ",j)
        predictions =  np.array([LRLS(x_test[i,:].reshape(1,d),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions-y_test)**2).mean()
    return losses


#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    x_mean = np.expand_dims(x_train.mean(axis = 0), 0)
    exp_sum = np.squeeze(l2(x_mean, x_train))
    exp_sum = exp_sum * (-1/(2*tau*tau))
    B_max = np.amax(exp_sum)
    exp_sum = np.exp(exp_sum - B_max)
    summation = sum(exp_sum)
    
    a = exp_sum / summation
    
    la = np.zeros((14, 14), float)
    np.fill_diagonal(la, lam)
    
    A = np.diag(a)
    One = np.matmul(np.matmul(np.transpose(x_train), A), x_train) + la
    Two = np.matmul(np.matmul(np.transpose(x_train), A), y_train)
    
    w, residual, rank, s = np.linalg.lstsq(One, Two)
    
    y_hat = np.matmul(w, np.transpose(test_datum))
    return y_hat

    
def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    losses = np.zeros(taus.shape[0])
    step = int(x.shape[0]/k)
    for subsample_i in range(k):
        print(subsample_i)
        if subsample_i < 4:
            x_test = x[subsample_i*step:(subsample_i+1)*step]
            x_train = np.concatenate([x[:subsample_i*step], x[(subsample_i+1)*step:]])
            y_test = y[subsample_i*step:(subsample_i+1)*step]
            y_train = np.concatenate([y[:subsample_i*step], y[(subsample_i+1)*step:]])
        else:
            x_test = x[subsample_i*step:]
            x_train = x[:subsample_i*step]
            y_test = y[subsample_i*step:]
            y_train = y[:subsample_i*step]
        lo = run_on_fold(x_test, y_test, x_train, y_train, taus)
        losses = losses + lo

    losses = losses/k
    return losses


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.plot(taus, losses)
    print("min loss = {}".format(losses.min()))
