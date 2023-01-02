import numpy as np
import scipy

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer


class Kmeans2c_RF():
    def __init__(self, n_estimators=3000, criterion='squared_error', n_jobs=-1, max_features=1.0, random_state=42):
        self.__kmeans = KMeans(n_clusters=3, random_state=random_state)
        self.__rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs,
                                        criterion=criterion, max_features=max_features)
        self.__norm = Normalizer()

    def fit(self, X, y):
        X_norm = self.__normalize(X)
        self.__kmeans.fit(X_norm)

        new_X = np.concatenate([X, self.__km_predict(X_norm)], axis=1)
        self.__rf.fit(new_X, y)
        return self

    def __km_predict(self, X):

        d1 = np.linalg.norm(self.__kmeans.cluster_centers_[0] - X, 2, axis=1)
        d2 = np.linalg.norm(self.__kmeans.cluster_centers_[1] - X, 2, axis=1)
        d3 = np.linalg.norm(self.__kmeans.cluster_centers_[2] - X, 2, axis=1)
        return np.concatenate([d1[:, np.newaxis], d2[:, np.newaxis], d3[:, np.newaxis]], axis=1)

    def predict(self, X):
        X_norm = self.__normalize(X)

        new_X = np.concatenate([X, self.__km_predict(X_norm)], axis=1)
        tst = self.__rf.predict(new_X)
        return tst

    def __normalize(self, X):
        ind = [(X[:, i] > 1).any() for i in range(X.shape[1])]
        X_norm = X.copy()
        X_norm[:, ind] = self.__norm.fit_transform(X_norm[:, ind])

        return X_norm


class GMM_RF():
    def __init__(self, n_estimators=3000, criterion='squared_error', n_jobs=-1, max_features=1.0, random_state=42):

        self.__gmm = GaussianMixture(n_components=2, covariance_type='spherical', random_state=random_state)
        self.__rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs,
                                          criterion=criterion, max_features=max_features)

    def fit(self, X, y):
        self.__gmm.fit(X)
        self.centroids = self.get_centroids(X)

        new_X = np.concatenate([X, self.__centroids_dist(X)], axis=1)
        self.__rf.fit(new_X, y)
        return self


    def get_centroids(self, X):
        centers = np.empty(shape=(self.__gmm.n_components, X.shape[1]))
        for i in range(self.__gmm.n_components):
            density = scipy.stats.multivariate_normal(cov=self.__gmm.covariances_[i], mean=self.__gmm.means_[i]).logpdf(X)
            centers[i, :] = X[np.argmax(density)]
        return centers


    def __centroids_dist(self, X):
        print(self.centroids[0])
        d1 = np.linalg.norm(self.centroids[0] - X, 2, axis=1)
        d2 = np.linalg.norm(self.centroids[1] - X, 2, axis=1)
        #print(np.concatenate([d1[:, np.newaxis], d2[:, np.newaxis]], axis=1))
        return np.concatenate([d1[:, np.newaxis], d2[:, np.newaxis]], axis=1)

    def predict(self, X):

        new_X = np.concatenate([X, self.__centroids_dist(X)], axis=1)
        tst = self.__rf.predict(new_X)
        return tst

