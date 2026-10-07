from sklearn.svm import SVR
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import datasets

df = datasets.load_diabetes(as_frame=True).frame 

x = df.drop("target",axis=1)
y = df["target"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1,1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()

model = SVR()
model.fit(x_train,y_train_scaled)

y_pred_scaled = model.predict(x_test)

print("r2 :", r2_score(y_test_scaled,y_pred_scaled))


