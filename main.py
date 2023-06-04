# USING SMOTEENN
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import keras
from sklearn.metrics import classification_report

print("----------------------------------LOAD AND FILTER----------------------------------------")
# Load the dataset from an Excel file
df = pd.read_excel('data.xlsx')

df = df[~df.apply(lambda row: any('#' in str(val) for val in row), axis=1)]
df = df[~df.apply(lambda row: any('@' in str(val) for val in row), axis=1)]
df = df[~df.apply(lambda row: any('$' in str(val) for val in row), axis=1)]
df = df[~df.apply(lambda row: any('*' in str(val) for val in row), axis=1)]
df = df[~df.apply(lambda row: any('nan' in str(val) for val in row), axis=1)]
# df = df[~df.apply(lambda row: any('' in str(val) for val in row), axis=1)]

# relace
df['Gender'] = df['Gender'].replace({'F': 'Female'})
df['Gender'] = df['Gender'].replace({'M': 'Male'})
df['account_segment'] = df['account_segment'].replace({'Regular +': 'Regular Plus'})

# Drop the AccountID column
df = df.drop(['AccountID'], axis=1)

# Drop rows with missing values
df = df.dropna()



print(df.shape)


print("----------------------------------ONE HOT ENCODER----------------------------------------")
# Select categorical columns
ohe = OneHotEncoder(sparse=False)
cat_cols = ['Payment', 'Gender', 'account_segment', 'Marital_Status', 'Login_device']
encoded_cols = ohe.fit_transform(df[cat_cols])

# Replace original categorical columns with encoded columns
df = df.drop(cat_cols, axis=1)
for i, col in enumerate(ohe.get_feature_names(cat_cols)):
    df[col] = encoded_cols[:, i]
print(df.info())

# Separate the target variable
y = df['Churn'].to_numpy()
X = df.drop(['Churn'], axis=1).to_numpy()

print("----------------------------------FEATURES POLYNOMIAL----------------------------------------")
# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(X)
print(X.shape)

# print("----------------------------------SMOTE----------------------------------------")
# # Apply SMOTEENN to balance the dataset
# smote_enn = SMOTEENN()
# X, y = smote_enn.fit_resample(X, y)
# print(X.shape)



print("----------------------------------FEATURES SELECTION----------------------------------------")
# Create a random forest classifier model
rf_model = RandomForestClassifier(n_estimators=2000, n_jobs=-1)

# Train the model on the resampled dataset
rf_model.fit(X, y)

# Perform feature selection
selector = SelectFromModel(rf_model, prefit=True, threshold='mean')

# Transform the feature data
X = selector.transform(X)
print(X.shape)
print(y.shape)


print("----------------------------------SCALER----------------------------------------")
# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X.shape)

# ///////////////////////////////////
print("----------------------------------LogisticRegression----------------------------------------")
print(X.shape)
# Import logistic regression
from sklearn.linear_model import LogisticRegression

# Create a logistic regression model
model = LogisticRegression(n_jobs=-1, max_iter=2000)

# Evaluate the model using cross-validation
y_pred = cross_val_predict(model, X, y, cv=5, n_jobs=-1)

# Calculate evaluation metrics using true labels and predicted labels
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Print the mean and standard deviation of the scores
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {f1:.4f}")


# ///////////////////////////////////
print("----------------------------------SVM----------------------------------------")
print(X.shape)
# Import logistic regression

# Create a SVM classifier object
model = SVC(kernel='rbf')

# Evaluate the model using cross-validation
y_pred = cross_val_predict(model, X, y, cv=5, n_jobs=-1)

# Calculate evaluation metrics using true labels and predicted labels
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Print the mean and standard deviation of the scores
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {f1:.4f}")

# ///////////////////////////////////
print("----------------------------------DecisionTreeClassifier----------------------------------------")
print(X.shape)
# Create a DecisionTreeClassifier classifier object
model = DecisionTreeClassifier()

# Evaluate the model using cross-validation
y_pred = cross_val_predict(model, X, y, cv=5, n_jobs=-1)

# Calculate evaluation metrics using true labels and predicted labels
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Print the mean and standard deviation of the scores
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {f1:.4f}")

# ///////////////////////////////////
print("----------------------------------RandomForestClassifier----------------------------------------")
print(X.shape)

# Create a random forest classifier model
model = RandomForestClassifier(n_estimators=2000, n_jobs=-1)

# Evaluate the model using cross-validation
y_pred = cross_val_predict(model, X, y, cv=5, n_jobs=-1)

# Calculate evaluation metrics using true labels and predicted labels
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Print the mean and standard deviation of the scores
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {f1:.4f}")



# ///////////////////////////////////
print("----------------------------------DEEP NEURAL NETWORK----------------------------------------")

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model

# Define the k-fold cross-validation splitter
kf = KFold(n_splits=5, shuffle=True)

# Initialize the list to store the accuracy scores for each fold
accuracy_scores = []

reports = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(X):
    print(X.shape)
    print(y.shape)
    print(train_index.shape)
    print(test_index.shape)

    # Define the model
    model = keras.Sequential([
        keras.layers.Dense(64, input_shape=(X.shape[1],), activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model_checkpoint_callback = ModelCheckpoint(
        filepath='best_model.h5',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
        )

    callbacks = [model_checkpoint_callback]

    # Compile the model
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])


    # Split the data into training and test sets for the current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model on the training set for the current fold

    model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, verbose=1, callbacks=callbacks)

    model.load_weights('best_model.h5')

    # Evaluate the model on the test set for the current fold
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    # Calculate accuracy and append to accuracy_scores
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    # Calculate classification report and append to reports
    report = classification_report(y_test, y_pred, digits=4)
    reports.append(report)

    print(f'ACC : {accuracy_scores}')

# Print the mean and standard deviation of the accuracy scores across all folds
print(f"Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")

# Print the classification report for the last fold
print('Classification Report for the last fold:')
print(reports[-1])



















