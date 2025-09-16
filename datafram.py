import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
info=['name','roll','dept.']
info1=['humayan','09','CSE']
df=pd.DataFrame(zip(info,info1),columns=['student_info:','s_info1'], index=['s_name','s_roll','s_dept'])
print(df)
row,col=df.shape
print('\nrow=',row)
print('col=',col)
df1=pd.read_excel('Royal Knigths-17.xlsx')
print('\n \n',df1.head(1))
print('\n \n',df1.tail(1))
print('\n \n \n',df1.values)
print('\n \n \n',df1.columns)
print('\n \n \n',df1.index)
print('\n \n \n',df1.Name)
print('\n \n \n',df1['SL NO'].head(5))
print('\n \n \n',df1.Name.tail(5))
#subneting
df3 = df1[df1['Size'].isin(['M', 'XL'])]
print(df3)
#statistical analysis
print('\n \n \n',df1.describe())
# Numeric columns correlation
numeric_df = df1.select_dtypes(include='number')
print('\n\n\n', numeric_df.corr())

# Heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.show()