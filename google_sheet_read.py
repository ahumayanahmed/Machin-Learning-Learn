import pandas as pd
import gspread

# Credentials ফাইল দিয়ে authorize
gc = gspread.service_account(
    filename=r"C:\Users\LENOVO\Downloads\humayan420-a1fc097993cf.json"
)

# Spreadsheet URL দিয়ে ওপেন
sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1KkoM5lHDKrXMzF4qiwr5OL5rBz8SFAkvdmlaBmiuKiI/edit?usp=sharing")

# নির্দিষ্ট worksheet (sheet tab) সিলেক্ট করা
worksheet = sh.worksheet("hu")

# Data pandas DataFrame এ আনা
df = pd.DataFrame(worksheet.get_all_records())
print(df)
