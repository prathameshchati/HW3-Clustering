# Write your k-means unit tests here
from cluster.kmeans import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as sk_KMeans
import numpy as np
import pytest

def test_kmeans():

    # see if it can cluster sample data
    mat, observed_labels=make_blobs(n_samples=100, centers=3)

    kmeans=KMeans(3)
    pass

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