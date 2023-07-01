"""
Purpose: DATS 6401 Visualization of Complex Data / Data pre-processing
Name: Ei Tanaka
Datasets: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
Kaggle Project: Store Sales time series forecasting
"""

# ============================================================
# 1. Pre-processing Datasets
# ============================================================
# ============================================================
# 1.1 Loading Data
# ============================================================
import numpy as np
import pandas as pd

url1 = 'https://media.githubusercontent.com/media/eitanaka/StoreSalesVisualization/main/datasets/train.csv'
url2 = 'https://raw.githubusercontent.com/eitanaka/StoreSalesVisualization/main/datasets/stores.csv'
url3 = 'https://raw.githubusercontent.com/eitanaka/StoreSalesVisualization/main/datasets/oil.csv'
url4 = 'https://raw.githubusercontent.com/eitanaka/StoreSalesVisualization/main/datasets/holidays_events%20(1).csv'

train_df = pd.read_csv(url1)
stores = pd.read_csv(url2)
oil = pd.read_csv(url3)
holidays = pd.read_csv(url4)

print(train_df.head(5).to_string())
print(stores.head(5).to_string())
print(oil.head(5).to_string())
print(holidays.head(5).to_string())

# ============================================================
# 1.2 Handling Identifiers
# ============================================================
train_df = train_df.drop('id', axis=1)

# ============================================================
# 1.3 Handling Missing values
# ============================================================
print(train_df.isnull().sum()) # non-null
print(stores.isnull().sum()) # non-null
print(oil.isnull().sum()) # 43 null
print(holidays.isnull().sum()) # non-null

# Impouting oil using forward and backward filling
oil = oil.fillna(method='ffill')
oil = oil.fillna(method='bfill')

print(f"Total number of missing values in the 'oil' DataFrame: {oil.isnull().sum().sum()}")
# ============================================================
# 1.4 Creating Lists of family and store number
# ============================================================
family_list = train_df['family'].unique()
store_list = stores['store_nbr'].unique()
city_list = stores['city'].unique()
state_list = stores['state'].unique()
type_list = stores['type'].unique()

print(f'The families are following:\n{family_list}')
print(f'The store numbers are following: \n{store_list}')
print(f'The cities are following: \n{city_list}')
print(f'The states are following: \n{state_list}')
print(f'The types are following: \n{type_list}')
# ============================================================
# 1.5 Merging train_df and stores on store number ('store_nbr)
# Merging holidays into train_merged df
# ============================================================
train_merged = pd.merge(train_df, stores, on='store_nbr')
print(train_merged.head(5).to_string())

print(holidays['locale'].unique())
holidays.rename(columns={'type': 'day_nature'}, inplace=True, errors='raise')
holidays_loc = holidays[holidays['locale'] == 'Local'].copy()
holidays_reg = holidays[holidays['locale'] == 'Regional'].copy()
holidays_nat = holidays[holidays['locale'] == 'National'].copy()

holidays_loc.rename(columns={'locale_name': 'city'}, inplace=True)
holidays_reg.rename(columns={'locale_name': 'state'}, inplace=True)

unique_cities = holidays_loc['city'].unique()
mismatched_cities = set(unique_cities) - set(city_list)
print(f"Number of cities that don't match in the list: {len(mismatched_cities)}")

unique_states = holidays_reg['state'].unique()
mismatched_states = set(unique_states) - set(state_list)
print(f"Number of states that don't match in the list: {len(mismatched_states)}")

train_merged['holiday'] = 0

# Check if date and city match in holidays_loc, set 'holiday' to 1
train_merged.loc[
    (train_merged['date'].isin(holidays_loc['date'])) & (train_merged['city'].isin(holidays_loc['city'])),
    'holiday'] = 1

# Check if date and state match in holidays_reg, set 'holiday' to 1
train_merged.loc[
    (train_merged['date'].isin(holidays_reg['date'])) & (train_merged['state'].isin(holidays_reg['state'])),
    'holiday'] = 1

# Check if date matches in holidays_nat, set 'holiday' to 1
train_merged.loc[
    train_merged['date'].isin(holidays_nat['date']),
    'holiday'] = 1

# ============================================================
# 1.6 Handling Date time objcets
# ============================================================
train_merged['date'] = pd.to_datetime(train_merged['date'])
train_merged['month'] = pd.to_datetime(train_merged['date']).dt.month
train_merged['day'] = pd.to_datetime(train_merged['date']).dt.day
train_merged['day_name'] = pd.to_datetime(train_merged['date']).dt.day_name()
train_merged['year'] = pd.to_datetime(train_merged['date']).dt.year
print(train_merged.head(1).to_string())

# ============================================================
# 1.7 Data Shapes
# ============================================================
print(f'Number of train data set samples: {train_merged.shape}')
print(train_merged.info())

# ===============================================================
# 1.8 Remove store sales less than equal to 0 to get more insight
# ===============================================================
sales_df = train_merged[train_merged['sales'] > 0]
print(f'The shape removing sales less than equal to 0: {sales_df.shape}')

# ============================================================
# 2. Outlier Detection & removal for numerical data
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='store_nbr', y='sales', data=sales_df)
plt.title('Boxplot of Sales by Store Number (before removing outliers)')
plt.xlabel('Store Number')
plt.ylabel('Sales')
plt.xticks(rotation=90)
plt.show()

# Removal of rows with outliers using IQR method
Q1 = sales_df['sales'].quantile(0.25)
Q3 = sales_df['sales'].quantile(0.75)
IQR = Q3 - Q1
filtered_entries = ((sales_df['sales'] >= Q1 - 1.5 * IQR) & (sales_df['sales'] <= Q3 + 1.5 * IQR))
sales_df = sales_df.loc[filtered_entries]

print(f'The shape after removing outliers: {sales_df.shape}')

# Boxplot after removing outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='store_nbr', y='sales', data=sales_df)
plt.title('Boxplot of Sales by Store Number (after removing outliers)')
plt.xlabel('Store Number')
plt.ylabel('Sales')
plt.xticks(rotation=90)
plt.show()

# Export latest version of data frame as 'sales.csv'
# I'll use the following data frames for data viz. section
sales_df.to_csv('datasets/sales.csv', index=False)