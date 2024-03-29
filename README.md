# Gold Price Prediction Model

## Overview

This project aims to predict gold prices using a Random Forest Regression algorithm, a supervised learning technique that employs ensemble learning for regression.

## Algorithm Definition

Random Forest Regression is a supervised learning algorithm that uses an ensemble learning method for regression. It leverages the power of multiple decision trees to make accurate predictions.

## Flowchart

<img width="500" height="auto" alt="Gold" src="https://github.com/m-rishab/Gold-Price-Prediction-Model/assets/113618652/e6bee21c-641f-415a-a431-721dbe881229">

## Steps

### 1. Data Collection

The first step involves collecting relevant data from various sources such as financial news websites, government reports, and social media.

### 2. Data Preparation and Cleaning

After data collection, the dataset undergoes cleaning, which includes eliminating duplicates, filling in blanks, and formatting the data appropriately for analysis.

### 3. Feature Selection and Engineering

Machine learning models require features for predictions. Relevant features for gold price prediction may include economic indicators (e.g., inflation rates, interest rates, GDP) and market-specific factors (e.g., gold production, gold demand, geopolitical events).

### 4. Model Training

The machine learning model is trained using historical data. This involves selecting the right algorithm and optimizing its settings for optimal performance.

### 5. Evaluation of the Model

After training, the model needs to be evaluated by comparing its predictions with real gold prices to assess its accuracy and performance.

## Demo

https://github.com/m-rishab/Gold-Price-Prediction-Model/assets/113618652/6d57c042-2a5e-43af-b442-b84cd297cf15

```python
# Sample code snippet
from gold_price_prediction_model import GoldPricePredictor

# Load the trained model
model = GoldPricePredictor.load_model('path/to/trained/model')

# Make predictions
predictions = model.predict(gold_features)

# Display results
print("Predicted Gold Price:", predictions)
