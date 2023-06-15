import os
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

def get_data(data_path_feat, data_path_class):
    if os.path.exists(data_path_feat) and os.path.exists(data_path_class):
        list_feat = np.load(data_path_feat)
        list_class = np.load(data_path_class)
        if len(list_feat[0]) == len(list_class):
            list_feat_diff = np.zeros(np.shape(list_feat))
            x, y = np.shape(list_feat)
            for j in range(x):
                for i in range(0, y, 3):
                    list_feat_diff[j, i] = 0
                    list_feat_diff[j, i+1] = list_feat[j, i+1]/list_feat[j, i]
                    list_feat_diff[j, i+2] = list_feat[j, i+2]/list_feat[j, i]
            print(list_feat_diff)
            return list_feat_diff, list_class
        else:
            try:
                raise Exception("Länge der Liste der Features stimmt nicht mit der Länge der Liste der Klassen zusammen")
            except Exception as e:
                print(str(e))
    else:
        try:
            raise Exception("Datenpfade wurden nicht gefunden")
        except Exception as e:
            print(str(e))

def visualization_feature(list_features, list_class):
    plt.scatter(list_class, list_features[0])
    plt.xlabel('Klasse')
    plt.ylabel('Feature')
    plt.title('"EAR Eyes Open" mit Klassenzugehörigkeit')

    plt.show()


def KNN_Classifier(X_train, y_train, X_test):

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(y_pred)
    
data_path_feat = "Datasets/Perclos_EARopen/ear_perclos.npy"
data_path_class = "Datasets/Perclos_EARopen/ear_perclos_class.npy"

list_feat_diff, list_class = get_data(data_path_feat, data_path_class)

visualization_feature(list_feat_diff, list_class)