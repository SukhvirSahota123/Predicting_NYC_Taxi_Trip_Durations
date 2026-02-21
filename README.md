# Predicting_NYC_Taxi_Trip_Durations
Built a scalable ML pipeline to predict NYC taxi trip duration (1.45M+ rides) using feature engineering (Manhattan distance, cyclical encoding), RFE, and 5-fold cross-validation. Improved R² from 0.22 to 0.69 with a tuned Decision Tree model.

## Tech Stack:
- Python
- Colab
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Overview
This project builds and evaluates machine learning models to predict NYC taxi trip duration using geospatial and temporal features. The full workflow includes exploratory data analysis (EDA), anomaly detection, feature engineering, model comparison, hyperparameter tuning, and final model selection.

Dataset sourced from the Kaggle NYC Taxi Trip Duration competition (training data only).

## Dataset
i. ~1.45M rides
ii. 11 Original Features
iii. Target: trip_duration (seconds)

Key Feature Groups:
1. Geospatial: pickup/dropoff latitude & longitude
2. Temporal: pickup datetime, day of week, month of year
3. Trip metadata: passenger count, vendorID

## Data Cleaning and Preprocessing 
We applied structured filtering to remove anomalies and unrealistic entries:
i. Removed trips < 90 seconds or > 4 hours
ii. Filtered passenger counts outside valid range (1–7)
iii. Restricted coordinates to Greater NYC bounding box
iv. Applied log transformation to target (log_duration) to reduce skewness

These steps improved distribution stability and model suitability for regression.

## Feature Engineering
Engineered features included:
1. Manhattan Distance (proxy for route distance)
2. Log-transformed Manhattan Distance
3. Cyclical encoding of pickup hour (sin/cos)
4. Cyclical encoding of day of week (sin/cos)
5. Rush-hour indicator
6. Temporal breakdown (hour, month, weekday)

Recursive Feature Elimination (RFE) was used to rank and select the most informative predictors.

## Models Evaluated 
We compared multiple regression approaches:
1. Linear Regression
2. Ridge Regression
3. Decision Tree Regressor
4. Gradient Boosting Regressor

Evaluation metrics:
1. R^2
2. RMSE
3. MAE

All models were evaluated using K-fold cross-validation.

## Hyperparameter Tuning 
For the Decision Tree model, we performed:
i. GridSearchCV
ii. 5-fold cross-validation
iii. Tuning of max_depth, min_samples_split, min_samples_leaf, max_features
iv. Validation curves to assess bias-variance tradeoff

## Results
1. Baseline Linear Regression: R² ≈ 0.22 (initial simple model)
2. Improved Linear Model: R² ≈ 0.63
3. Tuned Decision Tree: R² ≈ 0.69
         i. Lower RMSE and MAE than linear models
         ii. Better capture of nonlinear geographic & temporal interactions
   
The Decision Tree was selected as the final model due to superior predictive performance and stable cross-validation results.

##Key Insights 
i. Trip duration is strongly nonlinear
ii. Manhattan distance is one of the most predictive features
iii. Time-of-day effects significantly influence travel duration
iv. Linear regression struggles to capture geographic interaction effects

