# Craigslist Used Vehicle Price Prediction

## Project Overview
This tool helps users estimate fair prices for used vehicles listed on Craigslist. Using machine learning, it analyzes key factors like mileage, age, and manufacturer to predict prices. The Random Forest model provides the most accurate estimates with 75% accuracy.

## Features
- Cleaned and processed 5,670 vehicle listings
- Visualizations of price distributions and key factors
- Compared multiple ML models (Linear Regression, Random Forest)
- Interactive price prediction with confidence intervals
- Comparison with similar vehicles

## How It Works
1. Input vehicle details (make, year, mileage, etc.)
2. System processes the data using our trained model
3. Returns price estimate with confidence range
4. Shows comparison with similar vehicles

## Technologies Used:
- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**:
  - Scikit-learn
  - Random Forest
- **Data Visualization**:
  - Matplotlib
  - Seaborn
- **Data Processing**: Pandas, NumPy

## Install dependencies:
- pip install -r requirements.txt

## Run the application:
- streamlit run app.py

## Installation
1. Clone repo:
```bash
git clone https://github.com/AyushmanShrestha2000/Used-Cars-Price-Prediction
cd craigslist-vehicle-prediction

project/
├── app.py                # Main application
├── vehicles.csv          # Dataset
├── README.md             # This file
└── requirements.txt      # Dependencies

[Kaggle Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data/data)
