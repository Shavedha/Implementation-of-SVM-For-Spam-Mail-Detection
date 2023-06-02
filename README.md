# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages
2. Read the file "spam.csv"
3. Split the data into training and testing sets, create and train the model, predict and calculate accuracy
4. Create an instance of 'SVC' and assign it to variable 'svc'
5. Predict the labels for the testing data using 'svc.predict'
6. End the program

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Y Shavedha
RegisterNumber: 212221230095 
```
```
import chardet
file = 'spam.csv'
with open (file, 'rb') as rawdata: # with open automatically closes the file.
    result = chardet.detect(rawdata.read(10000))
result

import pandas as pd
data = pd.read_csv("spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:


### data.head()
### data.info()
### data.isnull().sum()
### Y_prediction value
### Accuracy value


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
