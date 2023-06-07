import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#Reads the training and testing dataset from the csv file given
training_data = pd.read_csv("TrainingDataMulti.csv", header=None)
testing_data = pd.read_csv("TestingDataMulti.csv", header=None)

#Slice the dataset into features (X) and output (y)
X_training_data = training_data.iloc[:, :-1]
y_training_data = training_data.iloc[:, -1]

#Splits the training data given into 90% training data and 10% testing data, shuffles the data for randomness and fairness, as well as stratifying the data
X_train, X_test, y_train, y_test = train_test_split(X_training_data, y_training_data, train_size=0.9, random_state=1, stratify=y_training_data)

# Performs grid search to find best 'n_estimators' parameter
# ################################################################################################

# # Define a pipeline to search for the best combination of PCA truncation
# # and classifier regularization.
# pca = PCA()
# # Define a Standard Scaler to normalize inputs
# scaler = MinMaxScaler()

# # set the tolerance to a large value to make the example faster
# rf_model = RandomForestClassifier();

# # Parameters of pipelines can be set using '__' separated parameter names:
# param_grid = {
    # # "max_depth": np.arange(1,101).tolist(),
    # # "n_estimators": np.arange(1,1001).tolist(),
    # "n_estimators": [100, 200, 300, 400, 500],
# }
# search = GridSearchCV(rf_model, param_grid, n_jobs=-1,cv=10, scoring='accuracy')
# search.fit(X_train, y_train)
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)

# ################################################################################################

#Initialises the random forest model with the hyperparameters specified
rf_model = RandomForestClassifier(n_estimators=300, random_state=1, class_weight="balanced_subsample")

#Perform a 10-fold cross-validation and calculates its score
cv_score = cross_val_score(rf_model, X_train, y_train, cv=10, scoring='f1_macro')
print("cv_score: ", cv_score)
print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_score.mean(), cv_score.std()))

#Fits the model with the 90% training data and calculates its training score
print(
    "RandomForest score: %f"
    % rf_model.fit(X_train, y_train).score(X_test, y_test))
    
#Get results on 10% testing data as well as external unseen testing data
predictions = rf_model.predict(X_test)
output_label = rf_model.predict(testing_data)
print(output_label)

#get f1 score
print ("f1_score: ", f1_score(y_test, predictions, average='macro'))

#get confusion matrix
cm = confusion_matrix(y_test, predictions, labels=rf_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,                           display_labels=rf_model.classes_)
disp.plot()
plt.show()

#Export computed labels of the external testing data into a csv file
final_results = pd.DataFrame(output_label)
print(final_results.value_counts())
final_results.to_csv("TestingResultsMulti.csv", index=False, header=None)