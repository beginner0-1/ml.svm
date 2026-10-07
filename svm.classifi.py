from sklearn.svm import SVC
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
df = datasets.load_iris(as_frame=True).frame 
print(df.isnull().sum())

x = df.drop("target",axis =1)
y = df["target"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

svc = SVC()
svc.fit(x_train_scaled,y_train)

y_pred =svc.predict(x_test_scaled)

print("accuracy ;",accuracy_score(y_test,y_pred))
print("classification report :\n",classification_report(y_test,y_pred))