from sklearn.neighbors import KNeighborsClassifier

def KNN_Classifier(X_train, y_train, X_test):

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(y_pred)