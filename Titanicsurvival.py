#Task 1:-Use the Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not. This is a classic beginner project with readily available data.The dataset typically used for this project contains information about individual passengers, such as their age, gender, ticket class, fare, cabin, and whether or not they survived.
#Author - Ayush.Lokre

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv("C:\\Users\\AYUSH\\Desktop\\Codsoft\\Titanicsurvival\\Titanic-Dataset.csv")

#To get insights of the datasets
print(data.head())
print(data.tail())
print(data.info())
print(data.describe())

#Countplot of survived vs not survived
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=data)
plt.title('Count of Survived vs Not Survived')
plt.show()

#Countplot survived by sex
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Survived by Sex')
plt.show()

#Countplot survived by passenger class
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Pclass', data=data)
plt.title('Survived by Passenger Class')
plt.show()

#Histplot of age distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'].dropna(), kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

#boxplot by Passenger Class
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Fare', data=data)
plt.title('Fare by Passenger Class')
plt.show()

# Handle missing value
for column in ['Age', 'Embarked', 'Fare']:
    data[column].fillna(data[column].mode()[0], inplace=True)
# Droping columns
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#categorical values into numerical values
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
#Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))
#Print confusion matrix 
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))