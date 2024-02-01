# write your silhouette score unit tests here
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
from cluster.utils import (
  make_clusters, 
  plot_clusters, 
  plot_multipanel)
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as sk_KMeans
import numpy as np
from collections import Counter
from sklearn.metrics import silhouette_score, silhouette_samples
import pytest


def test_silhouette_score():
    # create sample data
    mat, observed_labels = make_clusters(k=3)

    # run kmeans fitting
    kmeans=KMeans(k=3, tol=None, max_iter=20)
    kmeans.fit(mat)

    # get outputs
    predicted_labels=kmeans.cluster_labels

    # get our silhouette score
    silhouette=Silhouette()
    silhouette_scores=silhouette.score(mat, predicted_labels)
    mean_ss=np.mean(silhouette_scores)

    # get sklearns silhouette scores
    sklearn_ss=silhouette_score(mat, predicted_labels)

    # check if they are roughly equal (less than a percent error)
    ss_error=((sklearn_ss-mean_ss)/sklearn_ss)*100
    assert ss_error<=0.005 # very low error rate

    # check that the silhouette score per sample is also below a certain error threshold




