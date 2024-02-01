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
    mat, observed_labels = make_clusters(k=3)

    # see if kmeans exits on iteration if tolerance is None (the number of iterations that it takes is stored)
    kmeans=KMeans(k=3, tol=None, max_iter=15)
    kmeans.fit(mat)
    assert kmeans.itr==15

    # for checking clustering, using the observed labels, our kmeans algorithm should cluster the same number of points as the observed labels
    # we can check that our two sets of labels have the same counts for each label without the actual label names
    predicted_labels_freq=dict(Counter(kmeans.cluster_labels))
    observed_labels_freq=dict(Counter(observed_labels))
    assert set(predicted_labels_freq.values())==set(observed_labels_freq.values()) # this is label agnostic, it just checks whether the same number of items are in each cluster
    assert len(kmeans.centroids)==3 # checks if we have three resulting centroids

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