# Craigslist Used Vehicle Price Prediction

# Project Overview
This project helps uses machine laerning to predict prices of different cars which are listed in Craiglist. Using key features (mileage, manufacture, age) prices are predicted. After using different models, Random Forest provided the most accurate prediction.

# Features
- Cleaned and processed 5,670 vehicle listings
- Visualizations of price distributions and key factors
- Compared multiple ML models (Linear Regression, Random Forest)
- Interactive price prediction with confidence intervals
- Comparison with similar vehicles

# Technologies Used: Streamlit, Python
- Machine Learning: Scikit-learn, Random Forest
- Data Visualization: Matplotlib, Seaborn
- Data Processing: Pandas, NumPy

# Install dependencies:
- pip install -r requirements.txt

# Run the application:
- streamlit run app.py

# Installation
1. Clone repo:
```bash
git clone https://github.com/AyushmanShrestha2000/Used-Cars-Price-Prediction
cd craigslist-vehicle-prediction

project/
├── app.py                # Main application
├── vehicles.csv          # Dataset
├── README.md             # This file
└── requirements.txt      # Dependencies

Data source: [Kaggle Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data/data)
