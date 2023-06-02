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
<img width="452" alt="image" src="https://github.com/Shavedha/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427376/866645a5-b65c-4dad-bda3-43e2b383a51b">

### data.info()
<img width="261" alt="image" src="https://github.com/Shavedha/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427376/5197f76b-87ff-4e84-bf55-dfbe25468ffc">

### data.isnull().sum()
<img width="174" alt="image" src="https://github.com/Shavedha/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427376/590d4cea-ce2b-45a0-86b3-56d150b597d7">

### Y_prediction value
<img width="468" alt="image" src="https://github.com/Shavedha/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427376/069e51c1-7f1c-4144-b73a-31190be65eae">

### Accuracy value
<img width="293" alt="image" src="https://github.com/Shavedha/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427376/399e65f0-75f4-491f-9bcb-96e08fe9654d">


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
