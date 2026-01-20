# ==========================================================================================================
# Sales Data Analysis Notebook 
# Purpose: Perform Exploratory Data Analysis (EDA) and preprocessing on Sales Data PDA 4052.
# Focus: Sales Person performance, Total Sales Value, and Priority correlation.
# The analysis is aimed to provide managerial insights and justify preprocessing steps.
# ==========================================================================================================

# ------ 1. Importing and Setting up the libraries ----------
# Pandas and Numpy are used for robust data manipulation and numerical operations.
# Matplotlib and Seaborn are employed for exploratory visualizations, helping identify trends and patterns.
# Seaborn is set with a 'whitegrid' style for clearer interpretation of plots.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')  # clean background to interpret plots accordingly

# ----- 2. Load raw dataset -----
# The dataset is imported to perform initial inspection prior to preprocessing.
# This step ensures we understand the structure, data types, and any inherent quality issues.
# Note: the assignment mentions an Excel file(.xls), here we use CSV for simplicity.

df = pd.read_csv('SalesData_4052.csv')

# ----- 3. Initial Data Inspection [Dataset Overview] -----
# Understanding the dataset's dimensions, types, and missing values is critical.
# This helps identify preprocessing requirements and informs subsequent analytical steps.


pd.set_option('display.expand_frame_repr' , False)  #Disable column wrapping for readability.
    
print('-' * 100, '\n')
print('First 5 rows of DataFrame \n', df.head(), '\n')
print('-' * 100, '\n')
print('Shape of dataset', df.shape, '\n')
print('-' * 100, '\n')
print('Data Types \n', df.dtypes, '\n')
print('-' * 100)
print('Missing values per column \n', df.isnull().sum(), '\n')

# Critical reasoning
# Ensuring correct data types allows for valid aggregation, correlation, and plotting.
# Missing values need to be addressed to avoid bias or errors in statistical measures.



# ----- 4. Data Processing & cleaning [Preparing for reliable analysis] -----
# Column names are standardized to avoid referencing errors and to improve readability.
# Common issues like special characters, spaces, and inconsistent casing are fixed.


df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace('£', '',regex=False)
    .str.replace('â', '',regex=False)
    .str.replace(' ', '_'))


# ----- 4.1 Rename value column & drop row missing info
# Rows missing a salesperson or total sales value are dropped because they cannot contribute meaningfully to analysis.
# Renaming the 'value_' column provides clarity in subsequent analysis. 

if 'value_' in df.columns:
    df = df.rename(columns={'value_': 'total_sales_value'})             
df = df.dropna(subset = ['sales_person', 'total_sales_value']) #drop unusable rows


# ----- 4.2 Handle missing numeric values -----
# Using the median is robust against skewed distribution.
# This ensures that imputed values do not distort central tendency measures like mean or standard deviation.


numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


# ----- 4.3 Handle missing categorical values -----
# Filling with the mode preserves the most common category and prevents loss of data.
# This approach maintains the integrity of categorical distributions.


categorical_cols = df.select_dtypes(include= 'object').columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# ----- 4.4 Data type corrections -----
# Convert 'date' column to datetime to enable temporal analysis.
# Coerce errors to NaT to avoid exceptions during plotting and aggregation.

df['date'] = pd.to_datetime(df['date'], errors = 'coerce')


# ----- 4.5 Priority mapping: keep categorical & numeric for correlation
# Priority is treated as ordinal data for correlation analysis with total sales value.
# Numeric mapping allows us to perform correlation and assess linear relationships.

priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}

df['priority'] = df['priority'].str.capitalize().str.strip()
df_valid = df[df['priority'].isin(priority_map.keys())].copy()
df_valid['priority_numeric'] = df_valid['priority'].map(priority_map)


# ----- 4.6 Remove duplicate rows -----
# Duplicates can artificially inflate counts and totals, affecting insights.
df_valid = df_valid.drop_duplicates()


# ----- 4.7 Outlier treatment using IQR method -----
# Outliers can disproportionately influence mean and correlation measures.
# Outlier detection is applied after imputation to ensure stable quantile estimation.
# Using IQR (Interquartile Range) ensures robust handling of extreme values.


Q1 = df_valid['total_sales_value'].quantile(0.25)
Q3 = df_valid['total_sales_value'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_valid = df_valid[
    (df_valid['total_sales_value'] >=lower_bound) & 
    (df_valid['total_sales_value'] <= upper_bound)]

print('-' * 100)
print('Cleaned columns and data types: \n', df.columns, '\n')
print('-' * 100)
print('Final dataset shape: \n', df_valid.shape, '\n')
print('-' * 100)


# ----- 5. Descriptive statistics -----
# Understanding central tendency, dispersion, and distribution after cleaning is crucial for interpreting results.
# Provides insight into sales spread, typical values, and variability.

print('Descriptive statistics: \n', df_valid.describe(), '\n')
print('-' * 100)
print('Sales by priority: \n', df_valid.groupby('priority')['total_sales_value'].describe(), '\n')
print('-' * 100)
print('Sales by salesperson: \n', df_valid.groupby('sales_person')['total_sales_value'].describe(), '\n')


# -----  6. Correlation Analysis -----
# Correlation assesses linear association between numeric priority and total sales value.
# Positive correlation indicates higher-priority deals tend to have higher sales values.

corr = df_valid['priority_numeric'].corr(df_valid['total_sales_value'])
print(f'Correlation between priority and total sales value: {corr:2f}')


# ----- 6.1 Average sales by sales_person
# Aggregating mean sales provides a performance benchmark for each salesperson.

avg_sales_by_person = (df_valid.groupby('sales_person')['total_sales_value'].mean()
                       .sort_values(ascending=False))

plt.figure()
avg_sales_by_person.plot(kind='bar')
plt.title('Average sales value by salesperson')
plt.xlabel('Sales person')
plt.ylabel('Average sales value')
plt.xticks(rotation=45, ha = 'right')
plt.tight_layout()
plt.show ()


# ------- 7. Visualization ------

# ----- 7.1 Total sales by sales person
sales_by_person = df_valid.groupby('sales_person')['total_sales_value'].sum().sort_values(ascending=False)
plt.figure(figsize= (10,6))

sales_by_person.plot(kind='bar', color= 'pink')
plt.title('Total sales value by Sales person')
plt.xlabel('Sales person')
plt.ylabel('Total sales value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ----- 7.2 Total vs Average sales comparison
# Provides insight into whether high total sales are due to volume of deals or high-value deals

sales_summary = df_valid.groupby('sales_person').agg(
    total_sales = ('total_sales_value', 'sum'),
    avg_sales = ('total_sales_value', 'mean'),
    deal_count = ('total_sales_value', 'count')). sort_values('total_sales', ascending = False)

print('salesperson summary: \n', sales_summary)

avg_sales_by_person.plot(kind='bar', color= 'green')
plt.title('Average sales value by salesperson')
plt.xlabel('Salesperson')
plt.ylabel('Average sales value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show ()


# ----- 7.3 Trend sales over the course of time
# Time-series aggregation helps identify overall sales trends.

if 'date' in df_valid.columns:
    sales_over_time = df_valid.groupby('date')['total_sales_value'].sum()

    plt.figure(figsize=(10,6))
    plt.plot(sales_over_time.index, sales_over_time.values, marker= 'o', color= 'purple')
    plt.title('Total sales value over time')
    plt.xlabel('Date')
    plt.ylabel('Total sales value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ----- 7.4 Total sales by priority 

totals = df_valid.groupby('priority')['total_sales_value'].sum().reindex(['Critical', 'High', 'Medium', 'Low'])
priority_colors = {'Critical':'#d62728','High':'#ff7f0e','Medium':'#2ca02c','Low':'#1f77b4'}

plt.figure(figsize=(8,6))
bars = plt.bar(totals.index, totals.values, color=[priority_colors[p] for p in totals.index])

plt.title('Total sales value by priority')
plt.xlabel('priority')
plt.ylabel('Total sales value')
plt.tight_layout()
plt.show()


# ----- 7.5 Correlation Heatmap (Priority vs Sales value)

plt.figure(figsize=(6,4))
sns.heatmap(df_valid[['priority_numeric','total_sales_value']].corr(),annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Priority and Total sales value')
plt.tight_layout()
plt.show()


# ----- 8. Business insights 
# Identify top and lowest performers to provide actionable insights for management.
top_salesperson = sales_by_person.idxmax()
top_sales_value = sales_by_person.max()

lowest_salesperson = sales_by_person.idxmin()
lowest_sales_value = sales_by_person.min()

print('Top performing salesperson:')
print(top_salesperson, '-', top_sales_value)

print('Lowest performing salesperson:')
print(lowest_salesperson, '-', lowest_sales_value)

# Analytical commentary:
# Top performers may warrant recognition or resource allocation
# Lowest performers could benefit from training or closer support
# Priority correlates positively with sales value, indicating higher-priority deals generate higher revenue.


# ----- 9. Final cleaned dataset

print('Cleaned dataset preview:')
print(df_valid.head())
