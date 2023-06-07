import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

training_data = pd.read_csv("TrainingDataMulti.csv", header=None)
testing_data = pd.read_csv("TestingDataMulti.csv", header=None)

X_training_data = training_data.iloc[:, :-1]
y_training_data = training_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_training_data, y_training_data, train_size=0.9, random_state=1, stratify=y_training_data)

knn = KNeighborsClassifier(n_neighbors=1)

cv_score = cross_val_score(knn, X_train, y_train, cv=10, scoring='f1_macro')
print("cv_score: ", cv_score)
print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_score.mean(), cv_score.std()))

print(
    "KNN score: %f"
    % knn.fit(X_train, y_train).score(X_test, y_test))
    
predictions = knn.predict(X_test)
output_label = knn.predict(testing_data)
print(output_label)

#print (predictions)
#print (y_test)
#get f1 score

print (f1_score(y_test, predictions, average='macro'))

#get confusion matrix
cm = confusion_matrix(y_test, predictions, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,                           display_labels=knn.classes_)
disp.plot()
plt.show()

final_results = pd.DataFrame(output_label)
print(final_results.value_counts())
final_results.to_csv("TestingResultsMulti.csv", index=False, header=None)


# #######################################################################################

# # Define a pipeline to search for the best combination of PCA truncation
# # and classifier regularization.
# pca = PCA()
# # Define a Standard Scaler to normalize inputs
# scaler = MinMaxScaler()

# # set the tolerance to a large value to make the example faster
# knn = KNeighborsClassifier();
# pipe = Pipeline(steps=[("scaler", scaler), ("knn", knn)])

# # Parameters of pipelines can be set using '__' separated parameter names:
# param_grid = {
    # "knn__n_neighbors": np.arange(1,101).tolist(),
# }
# search = GridSearchCV(pipe, param_grid, n_jobs=-1,cv=10, scoring='accuracy')
# search.fit(X_train, y_train)
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)


# # For each number of components, find the best classifier results
# results = pd.DataFrame(search.cv_results_)
# print (results)
# components_col = "param_knn__n_neighbors"
# best_clfs = results.groupby(components_col).apply(
    # lambda g: g.nlargest(1, "mean_test_score")
# )

# ################################################################################################