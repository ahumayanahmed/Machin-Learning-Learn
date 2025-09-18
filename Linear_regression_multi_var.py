import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_excel('Linear_regression_multi_var.xlsx')
print(df)

print('\n Missing Null Value:\n', df.isnull().sum())
print(df.experience)
# Fill null values
df['experience'] = df['experience'].fillna(df['experience'].mean())
print(df['experience'])
print("\n After filling missing values:\n", df)
print("\n Missing Null Value:\n", df.isnull().sum())
reg=linear_model.LinearRegression()
f=reg.fit(df[['speed','car_age','experience']],df.risk)
print(f)
re=reg.predict([[160,10,5]])
print('\n Risk =',re)
