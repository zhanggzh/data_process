from sklearn.neighbors import KDTree


class KDTreeSimilar(object):
    def __init__(self, features, features_path):

        self.tree = KDTree(features)
        self.features_path = features_path

    def predict(self, feature, k):
        _, indices = self.tree.query(feature, k=k)
        return self.features_path[indices]