import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
x,y=make_classification(n_samples=100, n_features=10, n_classes=2,random_state=42)
print(pd.DataFrame(x))
#split dataset training ang testing
x_test,y_test,x_train,y_train=train_test_split(x,y,test_size=.20,random_state=42)
print(x_test.shape)