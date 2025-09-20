import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# Linear data
x_linear = np.linspace(0, 10, 50)
y_linear = 2 * x_linear + 5 + np.random.randn(50) * 2  # add noise

# Non-linear data
x_nonlinear = np.linspace(0, 10, 50)
y_nonlinear = 2 * np.sin(x_nonlinear) + np.random.randn(50)  # add noise

#Plot Linear vs Non-linear data

plt.figure(figsize=(12, 5))

# Linear plot
plt.subplot(1, 2, 1)
plt.scatter(x_linear, y_linear, color='blue', label='Data (Linear)')
plt.plot(x_linear, 2 * x_linear + 5, color='red', label='True Linear')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Relationship')
plt.legend()

# Non-linear plot
plt.subplot(1, 2, 2)
plt.scatter(x_nonlinear, y_nonlinear, color='green', label='Data (Non-linear)')
plt.plot(x_nonlinear, 2 * np.sin(x_nonlinear), color='orange', label='True Non-linear')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Non-linear Relationship')
plt.legend()

plt.tight_layout()
plt.show()

# Correlation Coefficients

corr_linear = np.corrcoef(x_linear, y_linear)[0, 1]
corr_nonlinear = np.corrcoef(x_nonlinear, y_nonlinear)[0, 1]
print("Correlation (Linear Data):", corr_linear)
print("Correlation (Non-linear Data):", corr_nonlinear)

# Linear Regression on Linear Data

reg1 = LinearRegression()
reg1.fit(x_linear.reshape(-1, 1), y_linear)

print("R² Score (Linear Data):", reg1.score(x_linear.reshape(-1, 1), y_linear))

plt.scatter(x_linear, y_linear, color='blue', label='Data (Linear)')
plt.plot(x_linear, reg1.predict(x_linear.reshape(-1, 1)), color='green', label='Best Fit Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression (Linear Data)')
plt.legend()
plt.show()

#Linear Regression on Non-linear Data
poly = PolynomialFeatures(degree=4)  
X_poly = poly.fit_transform(x_nonlinear.reshape(-1, 1))

reg_poly = LinearRegression()
reg_poly.fit(X_poly, y_nonlinear)

# R² score
print("R² Score (Polynomial Regression):", reg_poly.score(X_poly, y_nonlinear))

# Prediction for plotting
x_curve = np.linspace(0, 10, 200).reshape(-1, 1)  
y_curve = reg_poly.predict(poly.transform(x_curve))

#Plot
plt.scatter(x_nonlinear, y_nonlinear, color='green', label='Data (Non-linear)')
plt.plot(x_nonlinear, 2 * np.sin(x_nonlinear), color='orange', label='True Non-linear')
plt.plot(x_curve, y_curve, color='red', label='Polynomial Fit (deg=4)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression (Non-linear Data)')
plt.legend()
plt.show()
