# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Display plots in notebook
%matplotlib inline

# Load dataset
df = pd.read_csv('train.csv')

# View first rows
df.head()

# Shape of dataset
print("Dataset Shape:", df.shape)

# Column names
print(df.columns)

# Dataset information
df.info()

df.describe()

# Our target is: SalePrice -- That’s what we want to predict.

#Visualize House Prices
plt.figure(figsize=(10,6))

sns.histplot(df['SalePrice'], kde=True)

plt.title('Distribution of Sale Prices')

plt.xlabel('Sale Price')

plt.ylabel('Frequency')

plt.show()

#Check Missing Values
missing_values = df.isnull().sum()

missing_values = missing_values[missing_values > 0]

missing_values.sort_values(ascending=False)

#Correlation Heatmap
correlation = df.corr(numeric_only=True)

plt.figure(figsize=(14,10))

sns.heatmap(correlation, cmap='coolwarm')

plt.title('Correlation Heatmap')

plt.show()

saleprice_corr = correlation['SalePrice'].sort_values(ascending=False)

print(saleprice_corr.head(10))

plt.figure(figsize=(10,6))

sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'])

plt.title('Ground Living Area vs Sale Price')

plt.xlabel('Ground Living Area')

plt.ylabel('Sale Price')

plt.show()

#Data Cleaning
missing_values = df.isnull().sum()

missing_values = missing_values[missing_values > 0]

missing_values.sort_values(ascending=False)

#missing PoolQC usually means: the house has no pool

#Calculate Percentage Missing
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_percent = missing_percent[missing_percent > 0]

missing_percent.sort_values(ascending=False)

#This helps decide:

# -which columns to drop
# -which to fill

#Drop Columns With Too Many Missing Values
columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']

df.drop(columns=columns_to_drop, inplace=True)

print("Updated Shape:", df.shape)

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())


#Why median?

#Housing prices and sizes are often skewed.
#Median handles outliers better than mean.

df.isnull().sum().sum()

remaining_missing = df.isnull().sum()

remaining_missing = remaining_missing[remaining_missing > 0]

remaining_missing.sort_values(ascending=False)

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = df[col].fillna('None')

df.isnull().sum().sum()

# Fill Numerical Missing Values
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())
    
# Detect Outliers
plt.figure(figsize=(10,6))

sns.boxplot(x=df['SalePrice'])

plt.title('Boxplot of Sale Prices')

plt.show()

# Remove Extreme Outliers
df = df.drop(df[(df['GrLivArea'] > 4000) &
                (df['SalePrice'] < 300000)].index)

print(df.shape)  

# Machine learning performs better with more normal distributions.
# apply logarithmic transformation
df['SalePrice'] = np.log1p(df['SalePrice'])

# New transformed Distribution
plt.figure(figsize=(10,6))

sns.histplot(df['SalePrice'], kde=True)

plt.title('Log-Transformed Sale Prices')

plt.show()

# Fill all categorical missing values with 'None'
categorical_cols = df.select_dtypes(include=['object']).columns

df[categorical_cols] = df[categorical_cols].fillna('None')

# Fill all numerical missing values with the median
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())
    
print("Remaining missing values:", df.isnull().sum().sum())    