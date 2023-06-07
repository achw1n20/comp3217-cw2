import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

#Reads the training and testing dataset from the csv file given
training_data = pd.read_csv("TrainingDataBinary.csv", header=None)
testing_data = pd.read_csv("TestingDataBinary.csv", header=None)

#Slice the dataset into features (X) and output (y)
X_training_data = training_data.iloc[:, :-1]
y_training_data = training_data.iloc[:, -1]

#Splits the training data given into 90% training data and 10% testing data, as well as shuffling the data for randomness and fairness
X_train, X_test, y_train, y_test = train_test_split(X_training_data, y_training_data, train_size=0.9, random_state=1)

#Performs grid search to find best 'C' parameter
# ###############################################################################################

# # Define a pipeline to search for the best combination of PCA truncation
# # and classifier regularization.
# pca = PCA()
# # Define a Standard Scaler to normalize inputs
# scaler = StandardScaler()

# # set the tolerance to a large value to make the example faster
# logistic = LogisticRegression(max_iter=10000, tol=0.1)
# pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])

# # Parameters of pipelines can be set using '__' separated parameter names:
# param_grid = {
    # "pca__n_components": [5, 15, 30, 45, 60],
    # # "logistic__C": np.logspace(-1, 1, 1),
    # "logistic__C": [0.1, 1, 10, 100, 1000, 1e5],
# }
# search = GridSearchCV(pipe, param_grid, n_jobs=-1,cv=5)
# search.fit(X_training_data, y_training_data)
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)

# # Plot the PCA spectrum
# pca.fit(X_training_data)

# fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
# ax0.plot(
    # np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
# )
# ax0.set_ylabel("PCA explained variance ratio")

# ax0.axvline(
    # search.best_estimator_.named_steps["pca"].n_components,
    # linestyle=":",
    # label="n_components chosen",
# )
# ax0.legend(prop=dict(size=12))

# # For each number of components, find the best classifier results
# results = pd.DataFrame(search.cv_results_)
# print (results)
# components_col = "param_pca__n_components"
# best_clfs = results.groupby(components_col).apply(
    # lambda g: g.nlargest(1, "mean_test_score")
# )

# best_clfs.plot(
    # x=components_col, y="mean_test_score", yerr="std_test_score", legend=False, ax=ax1
# )
# ax1.set_ylabel("Classification accuracy (val)")
# ax1.set_xlabel("n_components")

# plt.xlim(-1, 70)

# plt.tight_layout()
# plt.show()

# ###############################################################################################

#Initialises the logistic regression model with the hyperparameters specified
logistic = LogisticRegression(C=1, random_state=1, max_iter=10000)

#Perform a 10-fold cross-validation and calculates its score
cv_score = cross_val_score(logistic, X_train, y_train, cv=10, scoring='f1_macro')
print("cv_score: ", cv_score)
print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_score.mean(), cv_score.std()))

#Fits the model with the 90% training data and calculates its training score
print(
    "LogisticRegression score: %f"
    % logistic.fit(X_train, y_train).score(X_test, y_test))

#Get results on 10% testing data as well as external unseen testing data
predictions = logistic.predict(X_test)
output_label = logistic.predict(testing_data)
print(output_label)

#get f1 score
print ("f1_score: ", f1_score(y_test, predictions, average='macro'))

#get confusion matrix
cm = confusion_matrix(y_test, predictions, labels=logistic.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,                           display_labels=logistic.classes_)
disp.plot()
plt.show()

#Export computed labels of the external testing data into a csv file
final_results = pd.DataFrame(output_label)
print(final_results.value_counts())
final_results.to_csv("TestingResultsBinary.csv", index=False, header=None)