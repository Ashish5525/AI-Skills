# Job Classification and Information Retrieval

This repository contains examples of how to classify job titles into categories and retrieve relevant information based on the predicted categories.

## Overview

The examples demonstrate the following tasks:

1. **Data Preprocessing**: The input data, which includes job postings and relevant information (job descriptions, skills, qualifications, salary ranges), is preprocessed using Pandas and SpaCy. Special characters are removed, then text is converted to lowercase, and finally job titles are tokenized for further processing.

2. **Machine Learning Model Training**: A logistic regression classifier is trained using scikit-learn. The model is trained on vectorized job titles to predict job categories based on the input data.

3. **Prediction and Information Retrieval**: The trained model is used to predict job categories for new job titles. Relevant information (job descriptions, skills, qualifications, salary ranges) is retrieved from a separate dataset based on the predicted categories.

## Tools Used

- **Pandas**: Pandas is a data manipulation library in Python that provides powerful data structures and functions for working with tabular data. It is used for reading, filtering, and preprocessing input data stored in CSV files and allows for robust data preprocessing that is ideal for ML projects.

- **SpaCy**: SpaCy is a natural language processing (NLP) library in Python that offers tools and functionalities for various NLP tasks, such as tokenization, part-of-speech tagging, and named entity recognition. It is used for tokenizing job titles and performing text vectorization for feature engineering. We used SpaCy in tandom with Pandas to tokenize the CSV values into a preprocessed dataset that the model will use.

- **scikit-learn (sklearn)**: scikit-learn is a machine learning library in Python that provides tools for data mining and data analysis tasks. It includes various algorithms and utilities for classification, regression, clustering, and more. It is used for splitting data, training a logistic regression classifier, and evaluating the model's performance. We used this tool to create a logistic classifier to be able to accuratly predict data values for the selected job catagory.

- **train_test_split**: `train_test_split` is a function provided by scikit-learn. It splits arrays or matrices into random train and test subsets. It is used for splitting the labeled dataset into training and testing sets for model evaluation.

- **Logistic Regression**: Logistic regression is a classification algorithm that is used to model the probability of a binary outcome. It is used as the classification algorithm to predict job categories based on the vectorized job titles.

- **CSV File Input/Output**: CSV (Comma Separated Values) files are a common format for storing tabular data. They are used for storing the input data (job postings) and relevant information (job descriptions, skills, qualifications, salary ranges) for various jobs based on their titles. We used CSVs as the data stored in our training file was quite extensive at over 1.5 millions rows long.

- **DataFrame Manipulation**: DataFrame manipulation refers to various operations performed on Pandas DataFrames, such as selecting columns, filtering rows, applying functions to columns, and merging/joining DataFrames. It is used for filtering the input data based on target job titles and retrieving relevant information based on predicted categories.

## Usage

To run the examples:

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the example scripts.