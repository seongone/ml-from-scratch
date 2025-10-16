import numpy as np
from srcs.utils import L1_distance, L2_distance

class KNN_Classifier:
    def __init__(self, n_neighbors: int=3, metric: str='l2'):
        self.k = n_neighbors
        self.dist_mode = metric
        self.dist_func = {'l1': L1_distance, 'l2': L2_distance}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        
    def _vote(self, neighbor_lables: np.ndarray) -> np.int64:
        # lable들을 정수로 변환 후 가장 많은 빈도의 arg를 구합니다.
        counts = np.bincount(neighbor_lables.astype(np.int64))
        return counts.argmax()
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # X_test 길이 만큼의 np.array를 생성하고 초기화합니다.
        y_pred = np.zeros(X_test.shape[0], dtype=np.int64)
        for i, sample in enumerate(X_test):
            # sample과 x 사이의 거리를 측정하여 가장 가까운 idx들을 k개 추출합니다.
            idx_sorted_list = np.argsort([self.dist_func[self.dist_mode](sample, x) for x in self.X_train])[:self.k]
            # 가장 가까운 k개의 idx를 lable로 변환합니다.
            neighbor_lables = np.array([self.y_train[idx] for idx in idx_sorted_list])
            # k개의 lable 중에서 가장 빈도 수가 높은 것을 택해, 해당 lable을 y_pred[i]로 지정합니다.
            y_pred[i] = self._vote(neighbor_lables)

        return y_pred

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> np.float64:
        # X_test에 대해서 예측합니다.
        y_pred = self.predict(X_test)
        # 정확도를 구해주고 반환합니다.
        accuracy = ((y_pred == y_test).astype(np.int64)).mean()
        return accuracy

class KNN_Regressor:
    pass

if __name__ == '__main__':
    import mglearn
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    X, y = mglearn.datasets.make_forge()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    N_NEIGHBORS = 5
    METRIC = 'l2'

    print("Classifier Result")
    sklearn_knn_clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric=METRIC)
    sklearn_knn_clf.fit(X_train, y_train)
    print(f"sklearn_knn_clf Score: {sklearn_knn_clf.score(X_test, y_test):.3f}")

    my_knn_clf = KNN_Classifier(n_neighbors=N_NEIGHBORS, metric=METRIC)
    my_knn_clf.fit(X_train, y_train)
    print(f"my_knn_clf Score: {my_knn_clf.score(X_test, y_test):.3f}")