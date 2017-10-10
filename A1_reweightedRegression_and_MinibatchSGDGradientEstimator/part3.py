import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

K = 500
BATCHES = 50
boston = load_boston()
X = boston['data']
X = np.concatenate((np.ones((506,1)),X),axis=1) #add constant one feature - no bias needed
y = boston['target']

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(X, indices, 0)
        y_batch = y[indices]
        return X_batch, y_batch


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    #X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w

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

def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    #print(dot)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    #raise NotImplementedError()
    grad = 2 * np.matmul(w, np.matmul(np.transpose(X), X)) - 2 * np.matmul(y, X)
    return grad
    
def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    #w_ = fit_regression(X, y)
    computed_grad = lin_reg_gradient(X, y, w)
    
    print("Computed Gradient of Loss Fuction: \n", computed_grad)
    
    #sigma = np.zeros(K, float)
    # Example usage
    batchGradSum = []
    for i in range(K):
        X_b, y_b = batch_sampler.get_batch()
        #w_l = fit_regression(X_b, y_b)
        batch_grad = lin_reg_gradient(X_b, y_b, w)
        batchGradSum.append(batch_grad)
    
    #sigma = np.var(batchGradSum, 0)
    batch_grad_sum = np.sum(batchGradSum, axis=0)/K
    print("Gradient of Loss Fuction Applying Mini-Batch: \n", batch_grad_sum)
    
    squared_dist = np.squeeze(l2(np.expand_dims(computed_grad, axis=0), np.expand_dims(batch_grad_sum, axis=0)))
    cosine_sim = cosine_similarity(computed_grad, batch_grad_sum)
    
    print("squared distance: ", squared_dist)
    print("cosine similarity: ", cosine_sim)
    
    sigma = []
    for m in np.arange(1, 401):
        batchGradSum_ = []
        for i in range(K):
            X_bb, y_bb = batch_sampler.get_batch(m)
            batch_grad_ = lin_reg_gradient(X_bb, y_bb, w)
            batchGradSum_.append(batch_grad_)
        batch_grad_sum_ = np.sum(batchGradSum_, axis = 0)/K
        sigma.append(batch_grad_sum_)
    sigma = np.log(np.array(sigma))
    
    mm = np.log(np.arange(1,401))
    plt.figure(figsize=(20, 5))
    # i: index
    for i in range(14):
        #TODO: Plot feature i against y
        ax = plt.subplot(3, 5, i + 1)
        ax.set_xlabel("log(m)")
        ax.set_ylabel("log(sigma_j)")
        ax.set_title("w"+str(i))
        ax.scatter(mm,sigma[:,i])
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()