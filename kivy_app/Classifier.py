import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import joblib

def get_data(data_path_feat, data_path_class):
    if os.path.exists(data_path_feat) and os.path.exists(data_path_class):
        list_feat = np.load(data_path_feat)
        list_class = np.load(data_path_class)
        if len(list_feat[0]) == len(list_class):
            list_feat_diff = np.zeros(np.shape(list_feat))
            x, y = np.shape(list_feat)
            for i in range(0, y, 3):
                # PERCLOS Difference
                list_feat_diff[1, i] = 1.0
                list_feat_diff[1, i+1] = list_feat[1, i+1]/list_feat[1, i]
                list_feat_diff[1, i+2] = list_feat[1, i+2]/list_feat[1, i]
                # EAR Eyes Open Difference
                # PERCLOS Difference
                list_feat_diff[0, i] = 1.0
                list_feat_diff[0, i+1] = list_feat[0, i+1]/list_feat[0, i]
                list_feat_diff[0, i+2] = list_feat[0, i+2]/list_feat[0, i]
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
    class_color = {0: 'green', 1: 'blue', 2: 'red'}
    colors = [class_color[class_] for class_ in list_class]
    plt.scatter(list_features[1], list_features[0], c=colors)
    plt.xlabel('PERCLOS')
    plt.ylabel('EAR Eyes Open')
    plt.title('"EAR Eyes Open" zu PERCLOS')

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=class_)
                   for class_, color in class_color.items()]
    plt.legend(handles=legend_elements)

    plt.show()

def classification(list_features, list_class):
    X_train, X_test, y_train, y_test = train_test_split(list_features, list_class, 
                                                        test_size=0.2, random_state=42)
    classifiers = []

    # Logistic Regression
    print("\nLogistic Regression")
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test)
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    print("Genauigkeit Logistic Regression:", accuracy_logreg)
    # Cross Validation Logistic Regression
    logreg_scores = cross_val_score(logreg, list_features, list_class, cv=5)
    print("Kreuzvalidierung Logistic Regression:", logreg_scores)
    print("Durchschnittliche Genauigkeit Logistic Regression:", logreg_scores.mean())
    classifiers.append(("Logistic Regression", logreg, accuracy_logreg))

    # KNN Classificator with k=3
    print("\nKNN Classifier with k=3")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print("Genauigkeit KNN Classifier:", accuracy_knn)
    # Cross Validation KNN Classifier
    knn_scores = cross_val_score(knn, list_features, list_class, cv=5)
    print("Kreuzvalidierung KNN Classifier:", knn_scores)
    print("Durchschnittliche Genauigkeit KNN Classifier:", knn_scores.mean())
    classifiers.append(("KNN Classifier with k=3", knn, accuracy_knn))

    # KNN Classificator with k=1, NN
    print("\nKNN Classifier with k=1")
    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    print("Genauigkeit NN Classifier:", accuracy_nn)
    # Cross Validation KNN Classifier
    nn_scores = cross_val_score(nn, list_features, list_class, cv=5)
    print("Kreuzvalidierung NN Classifier:", nn_scores)
    print("Durchschnittliche Genauigkeit NN Classifier:", nn_scores.mean())
    classifiers.append(("KNN Classifier with k=1", nn, accuracy_nn))

    # Support Vector Machine
    print("\nSupport Vector Machine")
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("Genauigkeit Support Vector Machine:", accuracy_svm)
    # Cross Validation Support Vector Machine
    svm_scores = cross_val_score(svm, list_features, list_class, cv=5)
    print("Kreuzvalidierung Support Vector Machine:", svm_scores)
    print("Durchschnittliche Genauigkeit Support Vector Machine:", svm_scores.mean())
    classifiers.append(("Support Vector Machine", svm, accuracy_svm))

    classifiers.sort(key=lambda x: x[2], reverse=True)
    best_classifier_name, best_classifier, best_accuracy = classifiers[0]

    joblib.dump(best_classifier, "best_classifier.pkl")

def regression(list_features, list_class):
    X_train, X_test, y_train, y_test = train_test_split(list_features, list_class, 
                                                        test_size=0.2, random_state=42)
    regressors = []

    # Linear Regression
    print("\nLinear Regression")
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred_linreg = linreg.predict(X_test)
    accuracy_linreg = r2_score(y_test, y_pred_linreg)
    print("R2 Score Linear Regression:", accuracy_linreg)
    regressors.append(("Linear Regression", linreg, accuracy_linreg))

    # Polynomial Regression
    print("\nPolynomial Regression")
    polyreg = PolynomialFeatures(degree=2)
    X_poly = polyreg.fit_transform(X_train)
    linreg_poly = LinearRegression()
    linreg_poly.fit(X_poly, y_train)
    y_pred_poly = linreg_poly.predict(polyreg.transform(X_test))
    accuracy_polyreg = r2_score(y_test, y_pred_poly)
    print("R2 Score Polynomial Regression:", accuracy_polyreg)
    regressors.append(("Polynomial Regression", linreg_poly, accuracy_polyreg))

    # Ridge Regression
    print("\nRidge Regression")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    accuracy_ridge = r2_score(y_test, y_pred_ridge)
    print("R2 Score Ridge Regression:", accuracy_ridge)
    regressors.append(("Ridge Regression", ridge, accuracy_ridge))

    # Lasso Regression
    print("\nLasso Regression")
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    accuracy_lasso = r2_score(y_test, y_pred_lasso)
    print("R2 Score Lasso Regression:", accuracy_lasso)
    regressors.append(("Lasso Regression", lasso, accuracy_lasso))

    regressors.sort(key=lambda x: x[2], reverse=True)
    best_regressor_name, best_regressor, best_accuracy = regressors[2]

    joblib.dump(best_regressor, "best_regressor.pkl")
    
data_path_feat = "Datasets/Perclos_EARopen/ear_perclos.npy"
data_path_class = "Datasets/Perclos_EARopen/ear_perclos_class.npy"

list_feat_diff, list_class = get_data(data_path_feat, data_path_class)

Perclos_list = np.array(list_feat_diff[1]).reshape(-1, 1)

EAR_Eyes_open_list = np.array(list_feat_diff[0]).reshape(-1, 1)

EAR_and_PERCLOS = list(map(list, zip(*list_feat_diff)))

regression(Perclos_list, list_class)

# Lade den besten Regressor aus der Datei
best_regressor = joblib.load("best_regressor.pkl")

# Erstelle eine Reihe von Eingabewerten (Features)
min_feature = np.min(Perclos_list)
max_feature = np.max(Perclos_list)
new_features = np.linspace(min_feature, max_feature, num=100).reshape(-1, 1)

# Mache Vorhersagen für die erstellten Eingabewerte
predicted_scores = best_regressor.predict(new_features)

# Plot der ursprünglichen Merkmalsdaten und der Vorhersagen
plt.scatter(Perclos_list, list_class, color='blue', label='Original')
plt.plot(new_features, predicted_scores, color='red', label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Score')
plt.title('Regression')
plt.legend()
plt.show()