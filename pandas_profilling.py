import pandas as pd
from ydata_profiling import ProfileReport
df=pd.read_excel('August 25.xlsx')
profil=ProfileReport(df,title="pandas profiling report",explorative=True)
profil.to_file("August.html")