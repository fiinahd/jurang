from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle

class TfidfKNN:
    def __init__(self, k=5):
        self.vectorizer = TfidfVectorizer()
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.knn.fit(X, labels)

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.knn.predict(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.vectorizer, self.knn), f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            vect, knn = pickle.load(f)
        model = TfidfKNN()
        model.vectorizer, model.knn = vect, knn
        return model