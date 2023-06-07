import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

training_data = pd.read_csv("TrainingDataBinary.csv", header=None)
testing_data = pd.read_csv("TestingDataBinary.csv", header=None)

X_training_data = training_data.iloc[:, :-1]
y_training_data = training_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_training_data, y_training_data, train_size=0.95, random_state=1)

rf_model = RandomForestClassifier()

cv_score = cross_val_score(rf_model, X_train, y_train, cv=10, scoring='f1_macro')
print("cv_score: ", cv_score)
print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_score.mean(), cv_score.std()))

print(
    "RandomForest score: %f"
    % rf_model.fit(X_train, y_train).score(X_test, y_test))
    
predictions = rf_model.predict(X_test)
output_label = rf_model.predict(testing_data)
print(output_label)

#print (predictions)
#print (y_test)
#get f1 score

print (f1_score(y_test, predictions, average='macro'))

#get confusion matrix
cm = confusion_matrix(y_test, predictions, labels=rf_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,                           display_labels=rf_model.classes_)
disp.plot()
plt.show()

final_results = pd.DataFrame(output_label)
print(final_results.value_counts())
final_results.to_csv("TestingResultsBinary.csv", index=False, header=None)