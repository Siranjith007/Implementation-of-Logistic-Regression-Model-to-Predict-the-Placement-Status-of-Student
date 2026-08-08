# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, drop unnecessary columns, and encode categorical variables.
2. Define the features (X) and target variable (y).
3. Split the data into training and testing sets.
4. Train the logistic regression model, make predictions, and evaluate using accuracy and other.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SIRANJITH E
RegisterNumber:  212224060252
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)

```

## Output:
HEAD:
<img width="1279" height="330" alt="image" src="https://github.com/user-attachments/assets/622cf910-ebfd-461a-81e2-6020647ef653" />
COPY:
<img width="1141" height="345" alt="image" src="https://github.com/user-attachments/assets/86b6ea73-384f-4f51-b3c7-69fd244da549" />
FIT TRANSFORM:
<img width="1122" height="707" alt="image" src="https://github.com/user-attachments/assets/dba92154-a8c5-45c8-98b9-d2a1cf2da97a" />
LOGISTIC REGRESSION:
<img width="1231" height="309" alt="image" src="https://github.com/user-attachments/assets/ed54b50c-e9ef-42b3-a6e9-f2552a190abb" />
ACCURACY_SCORE:
<img width="1225" height="169" alt="image" src="https://github.com/user-attachments/assets/4860a70f-ae18-466e-a415-e7301613a8ea" />
CONFUSION MATRIX:
<img width="1229" height="203" alt="image" src="https://github.com/user-attachments/assets/48058f25-5153-4263-b001-65e11b69a691" />
CLASSIFICATION_REPORT AND PREDICTION:
<img width="1217" height="524" alt="image" src="https://github.com/user-attachments/assets/8bc96cfe-9ea0-4aa5-aece-33a0063b8b52" />









## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
