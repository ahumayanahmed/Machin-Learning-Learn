import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ডেটা লোড
df = pd.read_excel("start.xlsx", header=2)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(df)
print('\n \nMissing values:\n', df.isnull().sum())

# Scatter plot
plt.scatter(df['Starting'], df['Ending'])
plt.xlabel('Starting')
plt.ylabel('Ending')
plt.title('This is data analysis table')
plt.show()

# X, y split
x = df[['Starting']]   # ensure 2D
y = df[['Ending']]

print("X:\n", x.head())
print("Y:\n", y.head())


x_mean = x['Starting'].mean()
y_mean = y['Ending'].mean()

print("Mean of X (Starting):", x_mean)
print("Mean of Y (Ending):", y_mean)

# Scatter + Mean point
plt.scatter(df['Starting'], df['Ending'], label='Actual Data')
plt.scatter(x_mean, y_mean, color='red', marker='x', s=100, label='Mean Point')

# Linear Regression
reg = LinearRegression()
reg.fit(x, y)

# Slope & intercept
m = reg.coef_[0][0]
c = reg.intercept_[0]
print("Slope (m):", m)
print("Intercept (c):", c)

# Predict for 1000
y_pred_1000 = reg.predict([[1000]])
print("Prediction for Starting=1000:", y_pred_1000)

# Add predictions to DataFrame
df['predict_y'] = reg.predict(x)
print(df)

# Plot regression line
plt.scatter(df['Starting'], df['Ending'], label='Actual Data')
plt.plot(df['Starting'], df['predict_y'], color='red', label='Regression Line')
plt.xlabel('Starting')
plt.ylabel('Ending')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()
#loss and cost function mse & mae
from sklearn.metrics import mean_squared_error,mean_absolute_error
df['loss']=df['Ending']-df['predict_y']
print(df)
mse=mean_squared_error(df['Ending'],df['predict_y'])
print('\n MSE:',mse)
mae=mean_absolute_error(df['Ending'],df['predict_y'])
print('\n MAE:',mae)
#performance
print(reg.score(x,y))