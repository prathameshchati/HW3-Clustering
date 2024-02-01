import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        y_centers=[]
        # get cluster centroids
        for c in set(y):
            mat_cluster=X[np.where(y==c)]
            y_centers.append(np.mean(mat_cluster, axis=0))

        silhouette_scores=[]

        # compute silhouette score for each point in the matrix
        for obs_idx, c_idx in zip(range(len(X)), range(len(y))):

            # get point and center for that point
            obs=X[obs_idx]
            c=y[c_idx]

            # delete the point temporarily from the matrix
            mat_f=np.delete(X, [obs_idx], axis=0)
            y_f=np.delete(y, [c_idx], axis=0)

            # get all points in the cluster
            mat_f_cluster=mat_f[np.where(y_f==c)]

            # get next closest center
            arr=cdist([y_centers[c]], y_centers)
            arr_sorted=np.sort(arr)
            next_closest_center_idx=np.where(arr[0]==arr_sorted[0][1])[0][0]
            next_closest_center=y_centers[next_closest_center_idx]

            # get data from next closest center
            mat_f_cluster_next_closest=mat_f[np.where(y_f==next_closest_center_idx)]

            # average distance from point to all other points within its same cluster and next closest cluster
            avg_inter_cluster_distance=np.mean(cdist([obs], mat_f_cluster))
            avg_next_cluster_distance=np.mean(cdist([obs], mat_f_cluster_next_closest))
            
            # compute silhouette score
            ss=(avg_next_cluster_distance-avg_inter_cluster_distance)/max(avg_next_cluster_distance, avg_inter_cluster_distance)
            silhouette_scores.append(ss)

        return silhouette_scores


