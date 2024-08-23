# PRODIGY_DS_03
# Task 03: Decision Tree Classifier - Product Purchase Prediction

## Overview
This task involves building a decision tree classifier to predict whether a customer will purchase a product or service based on demographic and behavioral data. The objective is to analyze the dataset, preprocess the data, and train a model that can accurately predict customer behavior.

### Task Details
#### Objective:
To create a decision tree classifier that predicts whether a customer will purchase a product or service based on their demographic and behavioral data.

#### Sample Dataset:
The dataset used for this task is sourced from the UCI Machine Learning Repository: Bank Marketing Data.

## Dataset Information
### Bank Marketing Data
- Source: bank-full.csv.zip
- Description: This dataset contains information about a direct marketing campaign by a Portuguese banking institution. The goal is to predict whether the client will subscribe to a term deposit.
#### Key Columns:
- `age`: Age of the client.
- `job`: Type of job.
- `marital`: Marital status.
- `education`: Level of education.
- `default`: Has credit in default?
- `balance`: Account balance.
- `housing`: Has a housing loan?
- `loan`: Has a personal loan?
- `contact`: Type of communication contact.
- `day`: Last contact day of the month.
- `month`: Last contact month of the year.
- `duration`: Last contact duration, in seconds.
- `campaign`: Number of contacts performed during this campaign.
- `pdays`: Number of days since the client was last contacted from a previous campaign.
- `previous`: Number of contacts performed before this campaign.
- `poutcome`: Outcome of the previous marketing campaign.
- `y`: Target variable (whether the client subscribed to a term deposit).

## Model Development
### Steps:
1. **Data Preprocessing**:
    - Handling missing values.
    - Encoding categorical variables.
    - Splitting the dataset into training and testing sets.

2. **Model Training**:
    - Building the decision tree classifier using the training data.
    - Tuning hyperparameters to improve model accuracy.

3. **Model Evaluation**:
    - Evaluating the model's performance on the test set.
    - Generating a classification report, including precision, recall, and F1-score.

## Visualizations
- **Decision Tree Plot**: Visual representation of the decision tree model.
- **Confusion Matrix**: Displaying the accuracy of the predictions.
- **Feature Importance Plot**: Ranking the importance of each feature in making predictions.

## Libraries Used
- `pandas`: Used for data manipulation, including loading the dataset, handling missing values, encoding categorical variables, and splitting the dataset into training and testing sets.
- `numpy`: Utilized for numerical operations, including generating arrays, performing mathematical computations, and handling data structures that are passed to machine learning models.
- `scikit-learn`: Utilized for building the decision tree classifier, tuning hyperparameters, and evaluating the model's performance. It also provides functions for generating a confusion matrix and a classification report.
- `matplotlib`: Employed for visualizing the decision tree, plotting the confusion matrix, and creating various charts to understand the model's performance and feature importance.
- `seaborn`: Used alongside `matplotlib` for enhancing the visual appeal of the plots, particularly in creating the feature importance plot and the confusion matrix.

## Acknowledgments
- UCI Machine Learning Repository: For providing the Bank Marketing dataset.
