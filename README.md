# Breast Cancer Diagnosis Prediction

## Introduction

This project focuses on building machine learning models to predict breast cancer diagnosis (malignant or benign) based on characteristics of cell nuclei. The goal is to leverage machine learning techniques to assist in the classification of tumors.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, **obtained from Kaggle**. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image. The dataset includes 569 instances and 32 attributes, including the unique ID, diagnosis (M=malignant, B=benign), and 30 other features representing various measurements of the cell nuclei.

## Code Explanation

The accompanying notebook (`breast-cancer.ipynb`) contains the following steps:

1.  **Import Libraries**: Imports necessary libraries for data manipulation, visualization, and machine learning (pandas, numpy, matplotlib, seaborn, sklearn).
2.  **Load Data**: Loads the `breast-cancer.csv` dataset into a pandas DataFrame.
3.  **Data Exploration**: Performs initial data exploration, including checking the shape, column names, data types, missing values, and descriptive statistics.
4.  **Data Preprocessing**: Converts the categorical 'diagnosis' column into numerical values (M=1, B=0).
5.  **Data Visualization**: Visualizes the distribution of certain features (e.g., `area_worst`) for different diagnoses and creates a line plot of the diagnosis values.
6.  **Data Normalization**: Normalizes a subset of the features using `preprocessing.normalize`.
7.  **Data Splitting**: Splits the data into training and testing sets (70% train, 30% test) while maintaining the proportion of diagnoses in each set (stratification).
8.  **Logistic Regression Example**: Includes a simplified example of Logistic Regression to demonstrate the concept of binning and evaluation (Note: This is not the primary model used for prediction on the breast cancer data).
9.  **Linear Regression Model**: Trains a Linear Regression model on the prepared data and makes predictions on the test set.
10. **Model Evaluation**: Evaluates the Linear Regression model using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared. Accuracy for the binary classification is also calculated.

## Results

The Linear Regression model trained in this notebook achieved an accuracy of 1.0 on the test set. The regression metrics are as follows:
* Mean squared Error: 4.124405456972429e-05
* Root Mean squared Error: 4.124405456972429e-05
* Mean Absolute Error: 0.003636655128507961
* R-squared: 0.865755228107248

## How to Run the Notebook

1.  Clone the repository to your local machine.
2.  Make sure you have the necessary libraries installed (see the import section in the notebook).
3.  Download the `breast-cancer.csv` dataset and place it in the same directory as the notebook or update the file path in the code.
4.  Open the notebook in a Jupyter environment (e.g., Jupyter Notebook, JupyterLab, Google Colab).
5.  Run the cells sequentially to execute the code.
