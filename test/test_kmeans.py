# Write your k-means unit tests here
from cluster.kmeans import KMeans
import numpy as np
import pytest

def test_():
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
    mat, observed_labels=make_blobs(n_samples=100, centers=k)
    kmeans=KMeans(k=200)

    with pytest.raises(ValueError) as kmeans_fit:
            kmeans.fit(mat)
    assert kmeans_fit.type==ValueError


    
    



    pass