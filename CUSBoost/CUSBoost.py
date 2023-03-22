import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import math

import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble._weight_boosting import _samme_proba
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler



def cus_sampler(X_train, y_train, number_of_clusters=23, percentage_to_choose_from_each_cluster=0.50):

    selected_idx = []
    selected_idx = np.asarray(selected_idx)

    value, counts = np.unique(y_train, return_counts=True)
    minority_class = value[np.argmin(counts)]
    majority_class = value[np.argmax(counts)]

    idx_min = np.where(y_train == minority_class)[0]
    idx_maj = np.where(y_train == majority_class)[0]

    majority_class_instances = X_train[idx_maj]
    majority_class_labels = y_train[idx_maj]

    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(majority_class_instances)

    X_maj = []
    y_maj = []

    points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

    for key in points_under_each_cluster.keys():

        points_under_this_cluster = np.array(points_under_each_cluster[key])
        number_of_points_to_choose_from_this_cluster = math.ceil(
            len(points_under_this_cluster) * percentage_to_choose_from_each_cluster)




        selected_points = np.random.choice(points_under_this_cluster,
                                           size=number_of_points_to_choose_from_this_cluster, replace=False)
        X_maj.extend(majority_class_instances[selected_points])
        y_maj.extend(majority_class_labels[selected_points])

        selected_idx = np.append(selected_idx,selected_points)

        selected_idx = selected_idx.astype(int)


    X_sampled = X_train[selected_idx]
    y_sampled = y_train[selected_idx]

    return X_sampled, y_sampled, selected_idx


class CUSBoostClassifier:
    def __init__(self, n_estimators, depth):
        self.M = n_estimators
        self.depth = depth
        self.undersampler = RandomUnderSampler(replacement=False)

    def fit(self, X, Y):
        self.models = []
        self.alphas = []

        N, _ = X.shape
        W = np.ones(N) / N

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=self.depth, splitter='best')


            X_undersampled, y_undersampled, chosen_indices = cus_sampler(X,Y)


            tree.fit(X_undersampled, y_undersampled,
                     sample_weight=W[chosen_indices])

            P = tree.predict(X)

            err = np.sum(W[P != Y])

            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.0000001
            else:
                try:
                    if (np.log(1 - err) - np.log(err)) == 0 :
                        alpha = 0
                    else:
                        alpha = 0.5 * (np.log(1 - err) - np.log(err))
                    W = W * np.exp(-alpha * Y * P)  
                    W = W / W.sum()  
                except:
                    alpha = 0
 
                    W = W / W.sum()  

                self.models.append(tree)
                self.alphas.append(alpha)

    def predict(self, X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            FX += alpha * tree.predict(X)
        return np.sign(FX), FX

    def predict_proba(self, X):
        proba = sum(tree.predict_proba(X) * alpha for tree , alpha in zip(self.models,self.alphas) )


        proba = np.array(proba)


        proba = proba / sum(self.alphas)

        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0

        proba = proba /  normalizer

        return proba

    def predict_proba_samme(self, X):

        proba = sum(_samme_proba(est , 2 ,X) for est in self.models )

        proba = np.array(proba)

        proba = proba / sum(self.alphas)

        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba = proba / normalizer

        return proba.astype(float)


