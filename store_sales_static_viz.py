"""
Purpose: DATS 6401 Visualization of Complex Data
Name: Ei Tanaka
Datasets: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
Kaggle Project: Store Sales time series forecasting
"""

# Loading data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# need edited
path = 'datasets/sales.csv'
sales_df = pd.read_csv(path)

# ============================================================
# 3. Principal Components Analysis (PCA) for numerical value
# ============================================================
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler

# Since there are only two numerical value, I chose features just sales, and onpromotion
X = sales_df.drop(columns=sales_df.columns.difference(['sales', 'onpromotion'])).values
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
print(f'explain variance ratio {pca.explained_variance_ratio_}')

# Data Viz. PCA
plt.plot(np.arange(1,3), 100*np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('# of labels')
plt.ylabel('PCA explained variance ratio')
var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
k = np.argmax(var_cumu > 95)
print('# of components explaining 95% variance ' + str(k+1))
plt.axvline(x=k+1, color='k', linestyle='--')
plt.axhline(y=95, color='r', linestyle='--')
plt.grid()
plt.show()

H = X.T @ X
a, b = np.linalg.eig(H)
print("D_Original = ", np.sqrt(a))
print("The condition number for X is =", LA.cond(X))

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_pca = pca.transform(X)
print(f'explain variance ratio {pca.explained_variance_ratio_}')

plt.plot(np.arange(1,2), 100*np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('# of labels')
plt.ylabel('PCA explained variance ratio')

var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
k = np.argmax(var_cumu > 95)
print('# of components explaining 95% variance ' + str(k+1))
plt.axvline(x=k+1, color='k', linestyle='--')
plt.axhline(y=95, color='r', linestyle='--')

plt.grid()
plt.show()

H_pca = X_pca.T @ X_pca
a, b = np.linalg.eig(H_pca)
print("D_reduced = ", np.sqrt(a))
print("The condition number for X is =", LA.cond(X_pca))

# ============================================================
# 4. Normality Test
# ============================================================
from scipy.stats import shapiro
# perform a Shapiro-Wilk test
stat, p = shapiro(sales_df['sales'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

# Histogram
plt.figure(figsize=(8, 6))
sns.histplot(data=sales_df, x='sales')
plt.title('Histogram of Sales')
plt.xlabel('Range')
plt.ylabel('Frequency')
plt.show()

# QQplot
import statsmodels.api as sm
plt.figure()
sm.qqplot(sales_df['sales'], line='s')
plt.title("QQ Plot for sales")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()

# ============================================================
# 5. Data Transformation
# ============================================================
sales_df['log_sales'] = np.log1p(sales_df['sales'])

plt.figure(figsize=(8, 6))
sns.histplot(data=sales_df, x='log_sales')
plt.title('Histogram of log Sales')
plt.xlabel('Range')
plt.ylabel('Frequency')
plt.show()

# ============================================================
# 6. Heatmap & Pearson correlation coefficient matrix
# ============================================================
corr = sales_df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
plt.figure(figsize=(10,8))
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Pearson Correlation Heatmap')
plt.show()

# ============================================================
# 7. Statistics (Descriptive Statistic)
# ============================================================
# Descriptive statistics for all data
print(sales_df.describe().to_string())

median_sales = round(sales_df['sales'].median(), 2)
print("Median Sales: ", median_sales)

std_sales = round(sales_df['sales'].std(), 2)
print("Standard Deviation of Sales: ", std_sales)

# ============================================================
# 8. Data Visualization (Static)
# ============================================================
# ============================================================
# 8.1 Line Plot
# ============================================================
# Group by date for sum of sales
grouped_date = sales_df.groupby('date')['sales'].sum().reset_index()
# Line plot of Total Sales over Time
plt.figure(figsize=(15, 10))
sns.lineplot(data=grouped_date, x='date', y='sales')
plt.title("Total Sales Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

# Group by date and store_nbr for sum of sales
grouped_date_store = sales_df.groupby(['date', 'store_nbr'])['sales'].sum().reset_index()
grouped_date_store = grouped_date_store.sort_values(['store_nbr', 'date'])

# Define the number of subplots and the store number ranges
num_subplots = 6
store_ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, sales_df['store_nbr'].max())]

fig, axs = plt.subplots(num_subplots, 1, figsize=(15, 15), sharex=True)
for i, (start, end) in enumerate(store_ranges):
    subset_data = grouped_date_store[(grouped_date_store['store_nbr'] >= start) & (grouped_date_store['store_nbr'] <= end)]
    sns.lineplot(data=subset_data, x='date', y='sales', hue='store_nbr', ax=axs[i])
    axs[i].set_title(f"Store Numbers {start} to {end}")
    axs[i].set_xlabel("Date")
    axs[i].set_ylabel("Sales")
    axs[i].grid(True)
plt.tight_layout()
plt.show()

# Group by date and family for sum of sales (need edited)
grouped_date_family = sales_df.groupby(['date', 'family'])['sales'].sum().reset_index()
# Plot for 'grouped_date_family'
plt.figure(figsize=(15, 10))
sns.lineplot(data=grouped_date_family, x='date', y='sales', hue='family')
plt.title("Total Sales Volume by Product Family Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

# ============================================================
# 8.2 bar(stack) plot
# ============================================================
# Average Sales per Store Type
type_sales = sales_df.groupby('type')['sales'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=type_sales.index, y=type_sales.values)
plt.title('Average Sales by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Average Sales')
plt.show()
#
# Average Sales by City
city_sales = sales_df.groupby('city')['sales'].mean()
plt.figure(figsize=(15, 6))
sns.barplot(x=city_sales.index, y=city_sales.values)
plt.title('Average Sales by City')
plt.xlabel('City')
plt.ylabel('Average Sales')
plt.xticks(rotation=90)
plt.show()

# Total Sales by Product Family
family_sales = sales_df.groupby('family')['sales'].sum()
plt.figure(figsize=(15, 6))
sns.barplot(x=family_sales.index, y=family_sales.values)
plt.title('Total Sales by Product Family')
plt.xlabel('Product Family')
plt.ylabel('Total Sales')
plt.xticks(rotation=90)
plt.show()

# Average Sales by Holiday Status
holiday_sales = sales_df.groupby('holiday')['sales'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=holiday_sales.index, y=holiday_sales.values)
plt.title('Average Sales by Holiday Status')
plt.xlabel('Holiday')
plt.ylabel('Average Sales')
plt.show()

# ============================================================
# 8.3 count plot
# ============================================================
# Count Plot for Family
plt.figure(figsize=(15, 6))
sns.countplot(x='family', data=sales_df)
plt.title('Product Family Count')
plt.xlabel('Product Family')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Count plot for city
plt.figure(figsize=(15, 6))
sns.countplot(x='city', data=sales_df)
plt.title('Store Count by City')
plt.xlabel('City')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# ============================================================
# 8.4 Pie plot
# ============================================================
# Pie chart for type count
type_counts = sales_df['type'].value_counts()
plt.figure(figsize=(10, 8))
plt.pie(type_counts,
        labels=type_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        explode=[.01, .01, .01, .01, .01])
plt.axis('equal')
plt.title('Store Type Distribution')
plt.legend()
plt.show()

# Pie chart for count sales
grouped_type_sales = sales_df.groupby('type')['sales'].sum()
plt.figure(figsize=(10, 8))
plt.pie(grouped_type_sales,
        labels=grouped_type_sales.index,
        autopct='%1.1f%%',
        startangle=140,
        explode=[.01, .01, .01, .01, .01])
plt.axis('equal')
plt.title('Sales by Type')
plt.legend()
plt.show()

# ============================================================
# 8.5 Displot
# ============================================================
# Displot of sales by type
plt.figure(figsize=(10, 6))
sns.displot(data=sales_df,
            x='sales',
            hue='type',
            bins=30,
            common_bins=True)
plt.title('Distribution of Sales by Type')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 8.6 Pair plot
# ============================================================
sample_df = sales_df.sample(frac=0.01, random_state=42)
sns.pairplot(sample_df[['sales',
                        'onpromotion',
                        'store_nbr',
                        'cluster',
                        'holiday']])
plt.show()

# ============================================================
# 8.7  Kernel Density
# ============================================================
plt.figure(figsize=(10,6))
sns.kdeplot(data=sales_df, x='sales', hue='type')
plt.title('Kernel Density Estimation of Sales')
plt.xlabel('Sales')
plt.ylabel('Density')
plt.show()

# ============================================================
# 8.8 Scatter plot
# ============================================================
sample_df = sales_df.sample(frac=0.05, random_state=42)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sample_df,
                x='onpromotion',
                y='sales',
                hue='holiday')
plt.title('Scatter plot of Sales vs On-Promotion Quantity')
plt.xlabel('On-Promotion Quantity')
plt.ylabel('Sales')
plt.show()

# ============================================================
# 8.9 Area Plot
# ============================================================
sales_over_time = sales_df.groupby('date')['sales'].sum().reset_index()
sales_over_time = sales_over_time.sort_values('date')
plt.figure(figsize=(15,10))
plt.fill_between(sales_over_time['date'], sales_over_time['sales'], color="skyblue", alpha=0.4)
plt.plot(sales_over_time['date'], sales_over_time['sales'], color="Slateblue", alpha=0.6)
plt.title('Total Sales over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

# ============================================================
# 8.10 Violin Plot
# ============================================================
plt.figure(figsize=(10, 6))
sns.violinplot(data=sales_df, x='type', y='sales')
plt.title('Violin Plot of Sales by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()

# ============================================================
# 8.11 Heatmap
# ============================================================
table_month = pd.pivot_table(sales_df, values='sales', index='store_nbr', columns='month', aggfunc=np.sum)
fig, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(table_month, annot=False, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.title("Sales by Store and Month")
plt.tight_layout()
plt.show()

table_family = pd.pivot_table(sales_df, values='sales', index='family', aggfunc=np.sum)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(table_family, annot=True, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.title("Sales by Family")
plt.tight_layout()
plt.show()

total_sum = table_family.sales.sum()
table_family/total_sum

table_day = pd.pivot_table(sales_df, values='sales', index='day_name', aggfunc=np.sum)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(table_day, annot=True, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.title("Sales by Day")
plt.tight_layout()
plt.show()

table_year = pd.pivot_table(sales_df, values='sales', index='store_nbr', columns='year', aggfunc=np.sum)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(table_year, annot=False, linewidths=.5, ax=ax, cmap="YlGnBu")
plt.title("Sales by Store Number and Year")
plt.tight_layout()
plt.show()

# ============================================================
# 9. Subplots
# ============================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Line Plot (grouped by date and store number)
grouped_sales_date_store = sales_df.groupby('date')['sales'].sum().reset_index()
axes[0, 0].plot(grouped_sales_date_store['date'], grouped_sales_date_store['sales'])
axes[0, 0].set_title('Line Plot')

# Bar Plot (grouped by store number)
grouped_sales_store = sales_df.groupby('store_nbr')['sales'].sum().reset_index()
axes[0, 1].bar(grouped_sales_store['store_nbr'], grouped_sales_store['sales'])
axes[0, 1].set_title('Bar Plot')

# Count Plot (store number)
sns.countplot(data=sales_df, x='store_nbr', ax=axes[1, 0])
axes[1, 0].set_title('Count Plot')

# Scatter Plot (store number vs. sales)
sample_df = sales_df.sample(frac=0.01, random_state=42)
axes[1, 1].scatter(sample_df['store_nbr'], sample_df['sales'])
axes[1, 1].set_title('Scatter Plot')

plt.tight_layout()
plt.show()
