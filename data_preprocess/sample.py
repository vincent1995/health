if __name__ == '__main__':

    from sklearn.datasets import make_classification

    from imblearn.over_sampling import SMOTE

    # Generate the dataset
    X, y = make_classification(n_classes=2, class_sep=2, weights=[0.997, 0.003],
                               n_informative=3, n_redundant=1, flip_y=0,
                               n_features=20, n_clusters_per_class=1,
                               n_samples=4000, random_state=10)
    print(y)

    # Apply SMOTE SVM
    sm = SMOTE(kind='svm')
    X_resampled, y_resampled = sm.fit_sample(X, y)
    import numpy as np
    print(np.all(y == y_resampled[0:5000]))
    print(y_resampled[5000:])
    print(np.all(0 == y_resampled[5000:]))

