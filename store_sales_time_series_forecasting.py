"""
Purpose: DATS 6401 Visualization of Complex Data
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

train_df = pd.read_csv(url1)
stores = pd.read_csv(url2)

print(train_df.head(5).to_string())
print(stores.head(5).to_string())
# ============================================================
# 1.2 Handling Identifiers
# ============================================================
train_df = train_df.drop('id', axis=1)

# ============================================================
# 1.3 Handling Missing values
# ============================================================
print(train_df.isnull().sum()) # non-null
print(stores.isnull().sum()) # non-null

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
# 1.5 Mergin train_df and stores on store number ('store_nbr)
# ============================================================
train_merged = pd.merge(train_df, stores, on='store_nbr')
print(train_merged.head(5).to_string())

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
plt.figure(figsize=(10,5))
sns.boxplot(x = 'sales', data = sales_df)
plt.title('Boxplot of Sales (before removing outliers)')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Removal using IQR method
Q1 = sales_df['sales'].quantile(0.25)
Q3 = sales_df['sales'].quantile(0.75)
IQR = Q3 - Q1
filtered_entries = ((sales_df['sales'] >= Q1 - 1.5 * IQR) & (sales_df['sales'] <= Q3 + 1.5 * IQR))
sales_df = sales_df[filtered_entries]

print(f'The shape after removing outliers: {sales_df.shape}')

# Boxplot after removing outliers
plt.figure(figsize=(10,5))
sns.boxplot(x = 'sales', data = sales_df)
plt.title('Boxplot of Sales (after removing outliers)')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# ============================================================
# 3. Principal Components Analysis (PCA) for numerical value
# ============================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = sales_df.drop(['date', 'store_nbr', 'family', 'city', 'state', 'type', 'cluster'], axis=1)
X = StandardScaler().fit_transform(X)

# Create a PCA that will retain 99% of the variance
pca = PCA(n_components=0.95, whiten=True)

# Conduct PCA
X_pca = pca.fit_transform(X)

# Show results
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_pca.shape[1])

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

# Group by date and store_nbr for sum of sales (need edited)
grouped_date_store = sales_df.groupby(['date', 'store_nbr'])['sales'].sum().reset_index()
grouped_date_store = grouped_date_store.sort_values(['store_nbr', 'date'])

plt.figure(figsize=(15, 10))
sns.lineplot(data=grouped_date_store, x='date', y='sales', hue='store_nbr')
plt.title("Total Sales Volume by Store Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True)
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
                        'cluster',]])
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
plt.figure(figsize=(10,6))
sns.scatterplot(data=sales_df,
                x='onpromotion',
                y='sales')
plt.title('Scatter plot of Sales vs Store Number')
plt.xlabel('On-promotion quantitiy')
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

# ============================================================
# 10. Dashboard
# ============================================================
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWwgP.css']
app = dash.Dash("Store Sale", external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Store Sales - Time Series', style={'textAlign': 'center'}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Tab one', children=[
            html.Div([
                html.Div('Select Store Number:', style={'marginTop': 10}),
                dcc.Dropdown(
                    id='store-dropdown',
                    options=[{'label': i, 'value': i} for i in train_merged['store_nbr'].unique()],
                    value=train_merged['store_nbr'].unique()[0]
                ),
                html.Div('Range Slider:', style={'marginTop': 10}),
                dcc.RangeSlider(id='range-slider'),
                html.Div('Button:', style={'marginTop': 10}),
                html.Button('Submit', id='submit-button', n_clicks=0),
                html.Div('Input Field:', style={'marginTop': 10}),
                dcc.Input(id='input'),
                html.Div('Output Field:', style={'marginTop': 10}),
                html.Div(id='output'),
                html.Div('Text Area:', style={'marginTop': 10}),
                dcc.Textarea(id='textarea'),
                html.Div('Check Box:', style={'marginTop': 10}),
                dcc.Checklist(id='checklist'),
                html.Div('Radio Items:', style={'marginTop': 10}),
                dcc.RadioItems(id='radioitems'),
                html.Div('DatePickerSingle:', style={'marginTop': 10}),
                dcc.DatePickerSingle(id='datepickersingle'),
                html.Div('DatePickerRange:', style={'marginTop': 10}),
                dcc.DatePickerRange(id='datepickerange'),
                html.Div('Upload Component:', style={'marginTop': 10}),
                dcc.Upload(id='upload'),
                html.Div('Download Component:', style={'marginTop': 10}),
                dcc.Download(id='download'),
                html.Div('Graph:', style={'marginTop': 10}),
                dcc.Graph(id='graph1'),
            ])
        ]),
        dcc.Tab(label='Tab two', children=[
            # Repeat the structure for Tab one and change the ids of the components
        ]),
        dcc.Tab(label='Tab three', children=[
            # Repeat the structure for Tab one and change the ids of the components
        ]),
    ]),
])
@app.callback(
    Output('graph1', 'figure'),
    Input('store-dropdown', 'value')
)
def update_graph(store_nbr):
    filtered_df = train_merged[train_merged['store_nbr'] == store_nbr]
    fig = px.line(filtered_df, x='date', y='sales', title=f'Store {store_nbr} Sales Over Time')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)

# %% [markdown]
# 17. the soft copy of your python programs

# %% [markdown]
# 18. readme.txt (explains how to run your python code)
