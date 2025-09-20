import pandas as pd

data={
    'x':[1,2,3,4,5,7,10],
    'y':[2,3,5,4,6,6,8]
}
df=pd.DataFrame(data)
print(df)
person_corr=df['x'].corr(df['y'])
print('\n person correlation=',person_corr)
3