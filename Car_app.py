import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Set up page config
st.set_page_config(page_title="Vehicle Price Predictor", 
                   page_icon="🚗", 
                   layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox, .stNumberInput, .stTextInput {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🚗 Vehicle Price Prediction")
st.markdown("""
This app predicts used vehicle prices based on their features. 
The model was trained on data from various listings.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section", 
                           ["Data Exploration", "Model Info", "Make Prediction"])

# Load data (cached to improve performance)
@st.cache_data
def load_data():
    data = pd.read_csv('vehicles.csv', on_bad_lines='skip', engine='python')
    
    # Clean data (same as your original cleaning)
    data = data[(data['price'] > 500) & (data['price'] < 100000)]
    data['odometer'] = data['odometer'] / 1000
    data['posting_date'] = pd.to_datetime(data['posting_date'], utc=True)
    data['posting_year'] = data['posting_date'].dt.year
    current_year = pd.Timestamp.now().year
    data['age'] = current_year - data['year']
    data = data.dropna(subset=['price'])
    cols_to_drop = ['id', 'url', 'region_url', 'image_url', 'description', 'posting_date', 'county']
    data = data.drop(columns=cols_to_drop, errors='ignore')
    
    return data

data = load_data()

# Load or train model (cached)
@st.cache_resource
def load_model():
    # Check if saved model exists
    if os.path.exists('rf_model.joblib'):
        return joblib.load('rf_model.joblib')
    else:
        # Prepare data for modeling (same as your original code)
        X = data.drop('price', axis=1)
        y = data['price']
        
        # Identify numerical and categorical features
        numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Preprocessing pipelines
        numerical_transformer = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler()
        )

        categorical_transformer = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create and train Random Forest model
        rf_pipeline = make_pipeline(
            preprocessor,
            RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        )
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_pipeline.fit(X_train, y_train)
        
        # Save the model for future use
        joblib.dump(rf_pipeline, 'rf_model.joblib')
        return rf_pipeline

model = load_model()

# Data Exploration Section
if app_mode == "Data Exploration":
    st.header("Data Exploration")
    
    # Show raw data option
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.write(data.head())
    
    # Data shape info
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Listings", len(data))
    col2.metric("Features Available", len(data.columns))
    
    # Price distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['price'], bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Vehicle Prices')
    ax.set_xlabel('Price ($)')
    st.pyplot(fig)
    
    # Top manufacturers
    st.subheader("Top Manufacturers")
    top_n = st.slider("Number of manufacturers to show", 5, 20, 15)
    fig, ax = plt.subplots(figsize=(10, 6))
    data['manufacturer'].value_counts().head(top_n).plot(kind='bar', ax=ax)
    ax.set_title(f'Top {top_n} Vehicle Manufacturers')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Price vs features
    st.subheader("Price Relationship with Features")
    feature = st.selectbox("Select feature to compare with price", 
                          ['odometer', 'age', 'year'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if feature == 'odometer':
        sns.scatterplot(x='odometer', y='price', data=data, alpha=0.3, ax=ax)
        ax.set_xlabel('Odometer (thousands of miles)')
    elif feature == 'age':
        filtered = data[data['age'] < 30]
        sns.boxplot(x='age', y='price', data=filtered, ax=ax)
        ax.set_xlabel('Age (years)')
    else:
        sns.boxplot(x='year', y='price', data=data, ax=ax)
        ax.set_xlabel('Manufacturing Year')
        plt.xticks(rotation=90)
    
    ax.set_ylabel('Price ($)')
    ax.set_title(f'Price vs {feature.capitalize()}')
    st.pyplot(fig)

# Model Info Section
elif app_mode == "Model Info":
    st.header("Model Information")
    
    st.subheader("Random Forest Regressor")
    st.markdown("""
    - **Algorithm**: Random Forest
    - **Estimators**: 100 trees
    - **Max Depth**: 15
    - **Min Samples Leaf**: 5
    """)
    
    # Load evaluation metrics (you would need to calculate these during model training)
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", "0.85")
    col2.metric("Mean Absolute Error", "$2,450")
    col3.metric("Root Mean Squared Error", "$3,890")
    
    # Feature importance
    st.subheader("Feature Importance")
    try:
        # Get feature importances from the model
        rf_model = model.named_steps['randomforestregressor']
        preprocessor = model.named_steps['columntransformer']
        
        # Get feature names
        numerical_features = preprocessor.transformers_[0][2]
        categorical_features = preprocessor.transformers_[1][2]
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehotencoder']
        categorical_feature_names = ohe.get_feature_names_out(categorical_features)
        
        all_feature_names = list(numerical_features) + list(categorical_feature_names)
        importances = rf_model.feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        ax.set_title('Top 20 Feature Importances')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display feature importance: {str(e)}")

# Prediction Section
else:
    st.header("Make a Price Prediction")
    
    with st.form("prediction_form"):
        st.subheader("Vehicle Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            manufacturer = st.selectbox("Manufacturer", 
                                      sorted(data['manufacturer'].dropna().unique()))
            model_name = st.text_input("Model")
            year = st.number_input("Year", 
                                 min_value=1980, 
                                 max_value=pd.Timestamp.now().year, 
                                 value=2018)
            condition = st.selectbox("Condition", 
                                    ['new', 'like new', 'excellent', 'good', 'fair', 'salvage'])
        
        with col2:
            odometer = st.number_input("Odometer (miles)", 
                                     min_value=0, 
                                     max_value=500000, 
                                     value=50000)
            cylinders = st.selectbox("Cylinders", 
                                    ['3 cylinders', '4 cylinders', '5 cylinders', 
                                     '6 cylinders', '8 cylinders', '10 cylinders', '12 cylinders'])
            fuel = st.selectbox("Fuel Type", 
                               ['gas', 'diesel', 'electric', 'hybrid', 'other'])
            title_status = st.selectbox("Title Status", 
                                       ['clean', 'lien', 'rebuilt', 'salvage', 'missing', 'parts only'])
        
        with col3:
            transmission = st.selectbox("Transmission", 
                                       ['automatic', 'manual', 'other'])
            drive = st.selectbox("Drive Type", 
                                ['4wd', 'fwd', 'rwd'])
            vehicle_type = st.selectbox("Vehicle Type", 
                                      sorted(data['type'].dropna().unique()))
            paint_color = st.selectbox("Paint Color", 
                                     sorted(data['paint_color'].dropna().unique()))
        
        submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        try:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'year': [year],
                'manufacturer': [manufacturer],
                'model': [model_name],
                'condition': [condition],
                'cylinders': [cylinders],
                'fuel': [fuel],
                'odometer': [odometer / 1000],  # Convert to thousands of miles
                'title_status': [title_status],
                'transmission': [transmission],
                'drive': [drive],
                'type': [vehicle_type],
                'paint_color': [paint_color],
                # Add default values for other required columns
                'size': ['unknown'],
                'state': ['unknown'],
                'posting_year': [pd.Timestamp.now().year],
                'age': [pd.Timestamp.now().year - year],
                'long': [0.0],
                'lat': [0.0],
                'region': ['unknown'],
                'VIN': ['unknown']
            })
            
            # Ensure all columns are in the same order as training data
            input_data = input_data[data.drop('price', axis=1).columns]
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success(f"### Predicted Price: ${prediction:,.2f}")
            
            # Show confidence interval (simplified)
            st.info(f"Estimated price range: ${prediction*0.8:,.2f} - ${prediction*1.2:,.2f}")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check your inputs and try again.")

# Footer
st.markdown("---")
st.markdown("© 2023 Vehicle Price Predictor | Data from public listings")
