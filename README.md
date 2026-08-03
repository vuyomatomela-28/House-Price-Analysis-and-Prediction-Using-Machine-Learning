# Home
Welcome to the House Price Analysis and Prediction project. This wiki documents the complete workflow, from data preprocessing to model evaluation and prediction using machine learning.

## Project Title
House Price Analysis and Prediction Using Machine Learning

## Project Overview
House prices are influenced by many factors, including the size of the property, construction quality, number of rooms, garage capacity, and location. Accurately estimating house prices is important for buyers, sellers, and real estate professionals.

This project applies machine learning techniques to analyze historical housing data and build predictive models capable of estimating house prices.

### - Problem statement
- Predicting house prices manually is difficult because many variables influence the final selling price.
- Goal: Build a model capable of predicting house prices using historical housing data.

### - Objectives
- Explore and understand the housing dataset.
- Clean and preprocess the data.
- Perform Exploratory Data Analysis (EDA).
- Build multiple machine learning models.
- Compare model performance.
- Select the best-performing model for prediction.

### Technologies used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset
The project uses the Ames Housing Dataset(from Kaggle), which contains detailed information about residential properties and their sale prices.

### Dataset Information:
- Number of observations: 1,460
- Number of features: 81
- Target variable: SalePrice

### Examples of Important Features:
- Overall Quality
- Ground Living Area
- Garage Capacity
- Basement Area
- Year Built
- Lot Size

The dataset contains both numerical and categorical variables, making it suitable for demonstrating data preprocessing and machine learning techniques.

## Introduction

- House prices are influenced by many factors such as location, house size, quality, and garage capacity.
- Traditional valuation methods can be subjective, so machine learning provides a data-driven approach to estimate house prices more consistently.

## Data Processing
Data preprocessing was performed to improve data quality before model training.

The following steps were completed:

- Identified and handled missing values.
- Filled missing numerical values using the median.
- Filled missing categorical values with appropriate labels.
- Converted categorical variables into numerical variables using One-Hot Encoding.
- Prepared the dataset for machine learning algorithms.

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) was conducted to better understand the dataset and identify relationships between variables.

Several visualisations were created, including:

- Distribution of house prices
![distribution of sales](https://github.com/vuyomatomela-28/House-Price-Analysis-and-Prediction-Using-Machine-Learning/blob/main/images/distribution%20of%20sales.png)

- Correlation heatmap
![correlation heatmap](https://github.com/vuyomatomela-28/House-Price-Analysis-and-Prediction-Using-Machine-Learning/blob/main/images/correlation%20heatmap.png)

The analysis revealed that variables such as Overall Quality, Ground Living Area, and Garage Capacity have a strong relationship with house prices.

EDA also helped identify missing values and potential outliers before model training.

## Feature Engineering
Feature engineering prepared the data for machine learning by transforming it into a suitable format.

The following techniques were applied:

- Separation of predictor variables and target variable.
- One-Hot Encoding for categorical variables.
- Train-test split using an 80:20 ratio.

These steps ensured that the models received numerical input and could be evaluated using unseen data.

## Model Training
Three regression algorithms were trained and compared:

### 1. Linear Regression

A simple regression model used as the baseline for comparison.

![linear regression - actual vs predicted](https://github.com/vuyomatomela-28/House-Price-Analysis-and-Prediction-Using-Machine-Learning/blob/main/images/linear%20regression%20-%20actual%20vs%20predicted.png)


### 2. Decision Tree Regressor

A non-linear model capable of learning complex relationships between features.

![Decision Tree - actual vs predicted](https://github.com/vuyomatomela-28/House-Price-Analysis-and-Prediction-Using-Machine-Learning/blob/main/images/Decision%20Tree%20-%20actual%20vs%20predicted.png)

### 3. Random Forest Regressor

An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

![Random Forest - actual vs predicted](https://github.com/vuyomatomela-28/House-Price-Analysis-and-Prediction-Using-Machine-Learning/blob/main/images/Random%20Forest%20-%20actual%20vs%20predicted.png)

Each model was trained using the same training dataset to ensure a fair comparison.

## Model Evaluation
The models were evaluated using three common regression metrics:

### 1. Mean Absolute Error (MAE)

Measures the average difference between predicted and actual house prices.

### 2.Root Mean Squared Error (RMSE)

Measures prediction error while giving greater importance to larger errors.

### 3.R² Score

Measures how well the model explains the variation in house prices.

After comparing all three models, the Random Forest Regressor achieved the best overall performance by producing the highest prediction accuracy and the lowest prediction errors.

## Results
The project successfully demonstrated the application of machine learning for house price prediction.

Key findings include:

- Random Forest Regressor produced the best predictive performance.
- Data preprocessing significantly improved model performance.
- Exploratory Data Analysis identified the most influential housing features.
- Machine learning effectively predicted house prices using historical housing data.

### Residual_distribution
![residual_distribution](https://github.com/vuyomatomela-28/House-Price-Analysis-and-Prediction-Using-Machine-Learning/blob/main/images/residual_distribution.png)

### feature_importance
![feature_importance](https://github.com/vuyomatomela-28/House-Price-Analysis-and-Prediction-Using-Machine-Learning/blob/main/images/feature_importance.png)

## Future Improvements
- Deploy the model as a web application using Streamlit or Flask.
- Use more recent housing datasets to improve generalisation.

## References
- Ames Housing Dataset
- Scikit-learn Documentation
- Pandas Documentation
- NumPy Documentation
- Matplotlib Documentation

## Conclusion
This project successfully demonstrated the application of machine learning techniques to analyze and predict residential house prices using the Ames Housing dataset. The project followed a complete data science workflow, beginning with data exploration and preprocessing, followed by feature engineering, model development, evaluation, and interpretation of the results.

During the data preparation phase, missing values were handled, categorical variables were encoded, and the dataset was transformed into a format suitable for machine learning. Exploratory Data Analysis (EDA) provided valuable insights into the relationships between housing characteristics and sale prices, revealing that variables such as overall quality, living area, garage capacity, and basement size were among the strongest predictors of house value.

Three machine learning algorithms—Linear Regression, Decision Tree Regressor, and Random Forest Regressor—were trained and evaluated using performance metrics including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the R² Score. Comparing multiple models made it possible to assess their strengths and limitations and identify the model that provided the most accurate predictions for the dataset.

The results demonstrate that machine learning can effectively model complex relationships within housing data and provide reliable estimates of property prices. Beyond prediction, the project highlights the importance of data preprocessing, feature engineering, and model evaluation in building accurate and dependable predictive models.

Overall, this project achieved its objective of developing a house price prediction system while providing practical experience in applying the complete machine learning lifecycle. The knowledge and skills gained throughout this project—including data cleaning, visualization, feature engineering, model training, evaluation, and interpretation—form a strong foundation for solving real-world predictive analytics problems and future work in the field of data science.
