"""
Purpose: DATS 6401 Visualization of Complex Data
Name: Ei Tanaka
Datasets: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
"""
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Data Loading
train_df = pd.read_csv('/Users/eitanaka/Documents/GitHub/DATS6401_Final_Project/datasets/train.csv')
print(train_df.head().to_string())
print(train_df.shape)
train_df['data'] = pd.to_datetime(train_df['date'])
print(train_df.info())

# 1. Cover Page
# 2. Table of Contents
# 3. Table of figures and tables
# 4. Abstract
# 5. Introduction
# 6. Description of the Dataset
# 7. Pre-processing dataset
# 8. Outlier detection & removal
# 9. Principal Components Analysis (PCA)
# 10. Normality Test
# 11. Data transformation
# 12. Heatmap & Pearson correlation coefficient matrix
# 13. Statistics (Descriptive Statistic)
# 14. Data Visualizaiton
# 15. Subplots
# 16. Dashboard
# 17. Observations
# 18. Recommendation
# 19. A separate appendix
# 20. References
# 21. the soft copy of your python programs
# 22. readme.txt (explains how to run your python code)

# ==================================================================================================
# Phase 1 Statistic Graph
# line, bar(stack), count, cat, pie, displot, pair, heatmap, hist, qq, kernel density,
# scatter, multivariate box, area, violin plot
# ==============================================================        ====================================

# ==================================================================================================
# Phase 2 Interactive Graph
# Multiple Division, Multiple Tabs, Range Slider, Drop down menu, Button, Input field, output field
# Text area, check box, radio items, DatePickerSingle, DatePickerRange, Upload components,
# Download components, Graphs: refer to 14 for the list of plots
# ==================================================================================================