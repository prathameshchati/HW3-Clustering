# Write your k-means unit tests here
from cluster.kmeans import KMeans
from cluster.utils import (
  make_clusters, 
  plot_clusters, 
  plot_multipanel)
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as sk_KMeans
import numpy as np
from collections import Counter
import pytest

def test_kmeans():


    # create sample data
    mat, observed_labels=make_clusters(n=100, m=2, k=3, scale=0.3) # set scale to get tight clusters
    # mat, observed_labels=make_blobs(n_samples=100, centers=3)

    # see if kmeans exits on iteration if tolerance is None (the number of iterations that it takes is stored)
    kmeans=KMeans(k=3, tol=None, max_iter=20)
    kmeans.fit(mat)
    assert kmeans.itr==20

    # for checking clustering, using the observed labels, our kmeans algorithm should cluster the same number of points as the observed labels
    # we can check that our two sets of labels have the same counts for each label without the actual label names
    predicted_labels_freq=dict(Counter(kmeans.cluster_labels))
    observed_labels_freq=dict(Counter(observed_labels))
    assert set(predicted_labels_freq.values())==set(observed_labels_freq.values()) # this is label agnostic, it just checks whether the same number of items are in each cluster
    assert len(kmeans.centroids)==3 # checks if we have three resulting centroids

    # test that the tolerance function is working - the number of iteratios should be less than the max
    kmeans=KMeans(k=3, tol=0.00001, max_iter=20)
    kmeans.fit(mat)
    assert kmeans.itr<20

    # check that it can handle large numbers of clusters
    mat, observed_labels=make_clusters(n=100, m=2, k=99, scale=0.3) # set scale to get tight clusters
    kmeans=KMeans(k=99, tol=0.00001, max_iter=20)
    kmeans.fit(mat)
    assert len(kmeans.centroids)==99 # checks if we have three resulting centroids - can't check the way it clustered because that could be a bit random. 

    # check that it works on a high dimensional dataset
    mat, observed_labels=make_clusters(n=1000, m=30, k=5, scale=0.3)
    kmeans=KMeans(k=5, tol=0.0000001, max_iter=100)
    kmeans.fit(mat)
    predicted_labels_freq=dict(Counter(kmeans.cluster_labels))
    observed_labels_freq=dict(Counter(observed_labels))
    assert set(predicted_labels_freq.values())==set(observed_labels_freq.values()) # this is label agnostic, it just checks whether the same number of items are in each cluster
    assert len(kmeans.centroids)==5 # checks if we have three resulting centroids






def test_edge_cases():
    # test where k is 0 or 1
    with pytest.raises(ValueError) as kmeans:
        KMeans(k=0)
    assert kmeans.type==ValueError

    with pytest.raises(ValueError) as kmeans:
        KMeans(k=1)
    assert kmeans.type==ValueError

    # raise error when k is greater than the given data
    mat, observed_labels=make_blobs(n_samples=100, centers=3)
    kmeans=KMeans(k=200)

    with pytest.raises(ValueError) as kmeans_fit:
            kmeans.fit(mat)
    assert kmeans_fit.type==ValueError