import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
student = pd.read_csv("C:\\Users\\Dell\\OneDrive\\Documents\\student's dropout dataset.csv")

# Mapping Target column to numerical values
student['Target'] = student['Target'].map({
    'Dropout': 0,
    'Enrolled': 1,
    'Graduate': 2
})

# Create a new DataFrame with relevant columns
student_df = student.iloc[:, [1, 11, 13, 14, 15, 16, 17, 20, 22, 23, 26, 28, 29, 34]]

# Splitting data into features and target
X = student_df.iloc[:, 0:13]
y = student_df.iloc[:, -1]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to evaluate models
# def evaluate_model(clf, X_train, X_test, y_train, y_test):
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # scores = cross_val_score(clf, X_train, y_train, cv=10)
    # return accuracy, scores.mean()

# List of classifiers to evaluate
classifiers = [
    LogisticRegression(),
    SGDClassifier(max_iter=1000, tol=1e-3),
    Perceptron(tol=1e-3, random_state=0),
    LogisticRegressionCV(cv=5, random_state=0),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(max_depth=10, random_state=0),
    NuSVC(),
    GaussianNB(),
    MultinomialNB(),
    BernoulliNB(),
    CategoricalNB(),
    KNeighborsClassifier(n_neighbors=3)
]

# Evaluate each classifier
# evaluation_results = []
# for clf in classifiers:
#     accuracy, cv_mean = evaluate_model(clf, X_train, X_test, y_train, y_test)
#     evaluation_results.append((clf.class.name, accuracy, cv_mean))

# Streamlit Dashboard
st.title('Student Dropout Analysis Dashboard')

# Dataset Overview
st.header('Dataset Overview')
st.write(student.head())

# Correlation Matrix
st.header('Correlation Matrix')
fig = px.imshow(student.corr())
st.plotly_chart(fig)

# Target Distribution
st.header('Target Distribution')
x = student_df['Target'].value_counts().index
y = student_df['Target'].value_counts().values
df = pd.DataFrame({'Target': x, 'Count_T': y})
fig = px.pie(df, names='Target', values='Count_T', title='How many dropouts, enrolled & graduates are there in Target column')
fig.update_traces(labels=['Graduate', 'Dropout', 'Enrolled'], hole=0.4, textinfo='value+label', pull=[0, 0.2, 0.1])
st.plotly_chart(fig)

# Scatter Plots
st.header('Scatter Plots')
fig1 = px.scatter(student_df, x='Curricular units 1st sem (approved)', y='Curricular units 2nd sem (approved)', color='Target')
st.plotly_chart(fig1)

fig2 = px.scatter(student_df, x='Curricular units 1st sem (grade)', y='Curricular units 2nd sem (grade)', color='Target')
st.plotly_chart(fig2)

# Box Plot for Age at Enrollment
st.header('Box Plot for Age at Enrollment')
fig = px.box(student_df, y='Age at enrollment')
st.plotly_chart(fig)

# Distribution of Age at Enrollment
st.header('Distribution of Age at Enrollment')
fig, ax = plt.subplots()
sns.histplot(student_df['Age at enrollment'], kde=True, ax=ax)
st.pyplot(fig)

# Model Evaluation

# RandomForestClassifier Hyperparameter Tuning
st.header('RandomForestClassifier Hyperparameter Tuning')
st.write("Tuning hyperparameters... This may take a while.")

param_grid = {
    'bootstrap': [False, True],
    'max_depth': [5, 8, 10, 20],
    'max_features': [3, 4, 5, None],
    'min_samples_split': [2, 10, 12],
    'n_estimators': [100, 200, 300]
}

rfc = RandomForestClassifier()

# Progress bar
progress_bar = st.progress(0)
total_fits = 288 * 5
current_fits = 0

def update_progress(current_fits, total_fits):
    progress_bar.progress(current_fits / total_fits)

clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

best_rf = RandomForestClassifier(**clf.best_params_)
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

# Final Model Evaluation
st.header('Final Model Evaluation')
st.write("Final Model - Accuracy: ", accuracy_score(y_test, y_pred))
st.write("Cross-Validation Mean: ", cross_val_score(best_rf, X_train, y_train, cv=10).mean())
st.write("Precision Score: ", precision_score(y_test, y_pred, average='micro'))
st.write("Recall Score: ", recall_score(y_test, y_pred, average='micro'))
st.write("F1 Score: ", f1_score(y_test, y_pred, average='micro'))