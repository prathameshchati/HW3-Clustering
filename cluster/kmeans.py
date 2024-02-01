import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        self.k=k
        self.tol=tol
        self.max_iter=max_iter

        # check if k is trivial, 0 or 1
        if (self.k==0 or self.k==1):
            raise ValueError("The value of k must be greater than 1, please choose another value for k.")
        


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # initialize matrix
        self.mat=mat

        # if k is equal to or greater than the number of observations, raise an error
        if (self.k>=self.mat.shape[0]):
            raise ValueError("The value for k cannot be equal to the number of observations. Use a larger dataset or reduce the value of k.")

        # initialize centroids as random points chosen from the existing data (random choices without replacement)
        self.centroids=self.mat[np.random.choice(self.mat.shape[0], self.k, replace=False),:]

        # initialize storage for old centroids and iterations
        itr=0
        old_centroids=[]

        while itr<self.max_iter: # POTENTIALLY CHANGE TO <= TO HANDLE IF MAX_ITER=0 ################################

            # compute distance matrix
            cdist_mat=cdist(self.mat, self.centroids)
            predicted_labels=np.argmin(cdist_mat, axis=1)
            old_centroids=self.centroids

            updated_centroids=[]
            for c in set(predicted_labels):
                mat_cluster=self.mat[np.where(predicted_labels==c)]
                updated_centroids.append(np.mean(mat_cluster, axis=0))
            self.centroids=updated_centroids

            # check if the difference between the old and new centroids is small enough (if it reaches the tolerance level); if it is, stop the fitting loop
            error=np.mean(np.subtract(self.centroids, old_centroids)/old_centroids*100)
            if (self.tol!=None and error<=self.tol):
                break
            
            itr+=1
        
        self.itr=itr
        self.cluster_labels=predicted_labels


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # in order to prevent overwriting of the mat object from predict, label new mat object as _mat
        self._mat=mat

        # compute distance matrix and predict labels based on closest distance
        cdist_mat=cdist(self._mat, self.centroids)
        predicted_labels=np.argmin(cdist_mat, axis=1)

        # return the predicted labels
        return predicted_labels

        # pass

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        sse=0
        for c in set(self.cluster_labels):
            mat_cluster=self.mat[np.where(self.cluster_labels==c)]
            se=np.sum((mat_cluster-[self.centroids[c]])**2)
            sse+=se
        mse=sse/len(self.mat)

        return sse

        # pass

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        # return the centroids that were stored
        return self.centroids

        # pass
