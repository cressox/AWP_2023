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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from itertools import combinations
import joblib
import pandas as pd

def get_data(data_path_feat, data_path_class):
    """
    Load data from feature and class files and calculate the difference values for PERCLOS and EAR Eyes Open.

    Args:
        data_path_feat (str): Path to the feature data file.
        data_path_class (str): Path to the class data file.

    Returns:
        tuple: A tuple containing the feature data with difference values and the class data.

    Raises:
        Exception: If the length of the feature list does not match the length of the class list.
        Exception: If the data paths were not found.

    """

    # Checking if the data paths exists
    if os.path.exists(data_path_feat) and os.path.exists(data_path_class):
        
        # Loading the data
        list_feat = np.load(data_path_feat)
        list_class = np.load(data_path_class)
        
        # Adjust the format of the list

        sublist1 = []
        sublist2 = []
        sublist3 = []
        sublist4 = []
        for i in range(0, len(list_feat), 4):
            sublist1.append(list_feat[i])
            sublist2.append(list_feat[i+1])
            sublist3.append(list_feat[i+2])
            sublist4.append(list_feat[i+3])

        list_features = [sublist1, sublist2, sublist3, sublist4]

        list_feat = []

        for i in range(len(list_features[0])):
            sublist = [row[i] for row in list_features]
            list_feat.append(sublist)

        list_fps = np.load("Datasets/fps.npy")
        for i in range(len(list_fps)):
            list_feat[i][0] = list_feat[i][0]/list_fps[i]
            list_feat[i][1] = (list_feat[i][1]/list_fps[i])*1000

        if len(list_feat) == len(list_class):
            # Calculating the difference of the awake status to the tired and the half tired status
            list_feat_diff = np.zeros(np.shape(list_feat))
            x, y = np.shape(list_feat)
            for i in range(0, x, 3):
                # PERCLOS Difference
                list_feat_diff[i][0] = 1.0
                list_feat_diff[i+1][0] = list_feat[i+1][0]/list_feat[i][0]
                list_feat_diff[i+2][0] = list_feat[i+2][0]/list_feat[i][0]
                # PERCLOS Difference
                list_feat_diff[i][1] = 1.0
                list_feat_diff[i+1][1] = list_feat[i+1][1]/list_feat[i][1]
                list_feat_diff[i+2][1] = list_feat[i+2][1]/list_feat[i][1]
                # PERCLOS Difference
                list_feat_diff[i][2] = 1.0
                list_feat_diff[i+1][2] = list_feat[i+1][2]/list_feat[i][2]
                list_feat_diff[i+2][2] = list_feat[i+2][2]/list_feat[i][2]
                # PERCLOS Difference
                list_feat_diff[i][3] = 1.0
                list_feat_diff[i+1][3] = list_feat[i+1][3]/list_feat[i][3]
                list_feat_diff[i+2][3] = list_feat[i+2][3]/list_feat[i][3]

            return list_feat_diff, list_class
        else:
            # Exception if the length of the lists are not the same
            try:
                raise Exception("Länge der Liste der Features stimmt nicht mit der Länge der Liste der Klassen zusammen")
            except Exception as e:
                print(str(e))
    # Exception if the Data Paths dont exist
    else:
        try:
            raise Exception("Datenpfade wurden nicht gefunden")
        except Exception as e:
            print(str(e))

def get_data_two_sets(data_path_feat, data_path_class):
    """
    Load data from feature and class files and calculate the difference values for PERCLOS and EAR Eyes Open.

    Args:
        data_path_feat (str): Path to the feature data file.
        data_path_class (str): Path to the class data file.

    Returns:
        tuple: A tuple containing the feature data with difference values and the class data.

    Raises:
        Exception: If the length of the feature list does not match the length of the class list.
        Exception: If the data paths were not found.

    """

    # Checking if the data paths exists
    if os.path.exists(data_path_feat) and os.path.exists(data_path_class):
        
        # Loading the data
        list_feat = np.load(data_path_feat)
        list_class = np.load(data_path_class)
        
        if len(list_feat[0]) == len(list_class):
            # Calculating the difference of the awake status to the tired and the half tired status
            x, y = np.shape(list_feat)
            list_feat_0_1 = np.zeros((x, y-int(y/3)))
            list_feat_1_2 = np.zeros((x, y-int(y/3)))
            for i in range(0, y-int(y/3), 2):
                j = int(i*1.5)
                per_0_1 = list_feat[1, j+1]/list_feat[1, j]
                ear_0_1 = list_feat[0, j+1]/list_feat[0, j]
                per_1_2 = list_feat[1, j+2]/list_feat[1, j]
                ear_1_2 = list_feat[0, j+2]/list_feat[0, j]
                # PERCLOS Difference
                list_feat_0_1[1, i] = 1.0
                list_feat_0_1[1, i+1] = per_0_1
                # EAR Eyes Open Difference
                list_feat_0_1[0, i] = 1.0
                list_feat_0_1[0, i+1] = ear_0_1
                # PERCLOS Difference
                list_feat_1_2[1, i] = list_feat[1, j+1]/list_feat[1, j]
                list_feat_1_2[1, i+1] = per_1_2
                # EAR Eyes Open Difference
                list_feat_1_2[0, i] = list_feat[0, j+1]/list_feat[0, j]
                list_feat_1_2[0, i+1] = ear_1_2
            print(list_feat_0_1)
            print(list_feat_1_2)
            return list_feat_0_1, list_feat_1_2, list_class
        else:
            # Exception if the length of the lists are not the same
            try:
                raise Exception("Länge der Liste der Features stimmt nicht mit der Länge der Liste der Klassen zusammen")
            except Exception as e:
                print(str(e))
    # Exception if the Data Paths dont exist
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

def classification(list_features, list_class, classes):
    """
    Perform classification using different classifiers and select the best performing classifier based on accuracy.

    Args:
        list_features (numpy.ndarray): Array containing the feature data
        list_class (numpy.ndarray): Array containing the class labels
        classes (int): Number of classes in the data

    Returns:
        str: Name of the best performing classifier
        object: Best performing classifier object
        float: Accuracy of the best performing classifier

    """
    X_train, X_test, y_train, y_test = train_test_split(list_features, list_class,
                                                        test_size=0.2, random_state=42)
    classifiers = []

    if classes == 2:
        logreg = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000)
    else:
        logreg = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)

    # Logistic Regression
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test)
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

    if classes == 2:
        result = logreg.predict_proba(X_test)
        print(result)

    # Confusion Matrix
    conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)

    # Precision
    precision_logreg = precision_score(y_test, y_pred_logreg, average='weighted', zero_division=0)

    # Recall
    recall_logreg = recall_score(y_test, y_pred_logreg, average='weighted', zero_division=0)
    # F1-Score
    f1_logreg = f1_score(y_test, y_pred_logreg, average='weighted', zero_division=0)

    # Cross Validation Logistic Regression
    logreg_scores = cross_val_score(logreg, list_features, list_class, cv=5)
    classifiers.append(("Logistic Regression", logreg, np.mean(logreg_scores), 
                        conf_matrix_logreg, precision_logreg, recall_logreg, f1_logreg))

    # KNN Classifier with k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)

    # Confusion Matrix
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

    # Precision
    precision_knn = precision_score(y_test, y_pred_knn, average='weighted', zero_division=0)

    # Recall
    recall_knn = recall_score(y_test, y_pred_knn, average='weighted', zero_division=0)

    # F1-Score
    f1_knn = f1_score(y_test, y_pred_knn, average='weighted', zero_division=0)

    # Cross Validation KNN Classifier
    knn_scores = cross_val_score(knn, list_features, list_class, cv=5)
    classifiers.append(("KNN Classifier with k=3", knn, np.mean(knn_scores), 
                        conf_matrix_knn, precision_knn, recall_knn, f1_knn))

    # Support Vector Machine
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)

    # Confusion Matrix
    conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

    # Precision
    precision_svm = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)

    # Recall
    recall_svm = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)

    # F1-Score
    f1_svm = f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)

    # Cross Validation Support Vector Machine
    svm_scores = cross_val_score(svm, list_features, list_class, cv=5)
    classifiers.append(("Support Vector Machine", svm, np.mean(svm_scores), 
                        conf_matrix_svm, precision_svm, recall_svm, f1_svm))

    # Sorting the classifiers by performance
    classifiers.sort(key=lambda x: x[2], reverse=True)
    best_classifier = classifiers[0][1]
    
    joblib.dump(best_classifier, "best_classifier.pkl")

    for classifier in classifiers:
        classifier_name = classifier[0]
        classifier_scores = classifier[2]
        print(f"Classifier: {classifier_name}, Mean Score: {classifier_scores}")

    return classifiers

def create_markdown_file(classifiers, name):
    """
    Create a Markdown file summarizing the classification results.

    Args:
        classifiers (list): List of classifiers and their results

    """
    df = pd.DataFrame(classifiers, columns=["Classifier", "Object", "Accuracy", "Confusion Matrix", "Precision", "Recall", "F1-Score"])
    
    # Generate Markdown table for F1-Score, Recall, Precision, and Accuracy
    table = df[["Classifier", "F1-Score", "Recall", "Precision", "Accuracy"]].to_markdown(index=False)
    
    # Create Markdown file and write the table
    with open(name, "w") as file:
        file.write("# Classification Results\n\n")
        file.write("## Performance Metrics\n\n")
        file.write(table)
        
        for index, row in df.iterrows():
            classifier_name = row["Classifier"]
            confusion_matrix = row["Confusion Matrix"]
            classes = confusion_matrix.shape[0]

            # Check if the confusion matrix is 2x2 or 3x3
            if classes == 2:
                # Convert confusion matrix to DataFrame
                df_confusion = pd.DataFrame(confusion_matrix, index=["True 0", "True 1"],
                                            columns=["Predicted 0", "Predicted 1"])

                file.write(f"\n\n## Confusion Matrix - {classifier_name}\n\n")
                file.write(df_confusion.to_markdown())
            elif classes == 3:
                # Convert confusion matrix to DataFrame
                df_confusion = pd.DataFrame(confusion_matrix, index=["True 0", "True 1", "True 2"],
                                            columns=["Predicted 0", "Predicted 1", "Predicted 2"])

                file.write(f"\n\n## Confusion Matrix - {classifier_name}\n\n")
                file.write(df_confusion.to_markdown())
    
    print("Markdown file 'classification_results.md' has been created.")

def three_to_two_classes(list_features, list_class):
    mask = np.isin(list_class, [0, 2])
    filtered_features = list_features[mask]
    filtered_class = list_class[mask]

    filtered_class_0_1 = np.where(filtered_class == 2, 1, filtered_class)

    return filtered_features, filtered_class_0_1

data_path_feat = "Datasets/PERCLOS_EARopen_EAR_BLINKduration_list.npy"
data_path_class = "Datasets/PERCLOS_EARopen_EAR_BLINKduration_class.npy"

list_feat_diff, list_class = get_data(data_path_feat, data_path_class)

classifiers_three = classification(list_feat_diff, list_class, 3)

# After the classification is performed, call the create_markdown_file function
create_markdown_file(classifiers_three, "classification_results_three_classes.md")

two_class_features, two_class_class = three_to_two_classes(list_feat_diff, list_class)

#classifiers_two = classification(two_class_features, two_class_class, 2)

# After the classification is performed, call the create_markdown_file function
#create_markdown_file(classifiers_two, "classification_results_two_classes.md")

def regression(list_features, list_class):

    """
    Perform regression using different regressors and select the best performing regressor based on R2 score.

    Args:
        list_features (numpy.ndarray): Array containing the feature data.
        list_class (numpy.ndarray): Array containing the target values.

    Returns:
        str: Name of the best performing regressor.
        object: Best performing regressor object.
        float: R2 score of the best performing regressor.

    """
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

regression(list_feat_diff, list_class)