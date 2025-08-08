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
import requests
import gdown
from io import StringIO

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
The model was trained on a sample of vehicle listings for optimal performance.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section", 
                           ["Data Exploration", "Model Info", "Make Prediction"])

# Optimized data loading function
@st.cache_data(ttl=3600)
def load_data_sample():
    """Load a sample of the data for faster processing"""
    try:
        # Add progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('Downloading dataset...')
        progress_bar.progress(20)
        
        # Download the CSV from Google Drive
        file_id = "1xPOSLvTlZ-aI54s6D5wxAEa0UB56RzBL"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", "vehicles.csv", quiet=True)
        
        status_text.text('Processing data...')
        progress_bar.progress(40)
        
        # Read only a sample of the data for faster processing
        # First, get the total number of rows
        total_rows = sum(1 for _ in open("vehicles.csv")) - 1  # Subtract header
        
        # Calculate sample size (use 20% or max 150k rows for better performance)
        sample_size = min(150000, max(25000, total_rows // 5))
        
        status_text.text(f'Loading {sample_size:,} rows from {total_rows:,} total rows...')
        progress_bar.progress(60)
        
        # Read the CSV file with sampling
        # Use skiprows to randomly sample the data
        skip_rows = sorted(np.random.choice(range(1, total_rows), 
                                          size=total_rows - sample_size, 
                                          replace=False))
        
        data = pd.read_csv("vehicles.csv", 
                          skiprows=skip_rows,
                          on_bad_lines='skip', 
                          engine='python',
                          low_memory=False)
        
        status_text.text('Cleaning data...')
        progress_bar.progress(80)
        
        # Basic data cleaning
        initial_size = len(data)
        data = data[(data['price'] > 500) & (data['price'] < 100000)]
        data = data.dropna(subset=['price'])
        
        # Convert odometer to thousands of miles
        if 'odometer' in data.columns:
            data['odometer'] = data['odometer'] / 1000
        
        # Handle date columns
        if 'posting_date' in data.columns:
            data['posting_date'] = pd.to_datetime(data['posting_date'], utc=True, errors='coerce')
            data['posting_year'] = data['posting_date'].dt.year
        
        # Calculate age
        if 'year' in data.columns:
            current_year = pd.Timestamp.now().year
            data['age'] = current_year - data['year']
            data = data[data['age'] >= 0]  # Remove future years
        
        # Drop unnecessary columns
        cols_to_drop = ['id', 'url', 'region_url', 'image_url', 
                        'description', 'posting_date', 'county']
        data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
        
        # Remove columns with too many missing values (>50%)
        missing_threshold = 0.5
        data = data.loc[:, data.isnull().mean() < missing_threshold]
        
        # Limit categorical variables to top categories for better performance
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'price':
                # Increase top categories for larger dataset
                top_categories = data[col].value_counts().head(50).index
                data[col] = data[col].where(data[col].isin(top_categories), 'other')
        
        status_text.text('Data loaded successfully!')
        progress_bar.progress(100)
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Clean up the downloaded file to save space
        if os.path.exists("vehicles.csv"):
            os.remove("vehicles.csv")
        
        st.session_state.data_source = f"Google Drive (Sample: {len(data):,} from {total_rows:,} rows)"
        st.session_state.data_info = {
            'original_size': total_rows,
            'sample_size': len(data),
            'cleaned_size': len(data),
            'reduction_ratio': len(data) / total_rows
        }
        
        return data

    except Exception as e:
        st.error(f"❗ Failed to load dataset: {str(e)}")
        # Return sample fallback data
        return create_sample_data()

def create_sample_data():
    """Create sample data if real data fails to load"""
    np.random.seed(42)
    n_samples = 5000  # Increased sample size
    
    manufacturers = ['toyota', 'ford', 'chevrolet', 'honda', 'nissan', 'hyundai', 'volkswagen', 'bmw', 'mercedes-benz', 'audi']
    conditions = ['excellent', 'good', 'fair', 'like new', 'salvage']
    fuel_types = ['gas', 'diesel', 'hybrid', 'electric']
    transmissions = ['automatic', 'manual']
    vehicle_types = ['sedan', 'SUV', 'truck', 'coupe', 'hatchback', 'convertible']
    
    data = pd.DataFrame({
        'price': np.random.normal(20000, 8000, n_samples).clip(5000, 50000),
        'year': np.random.randint(2000, 2024, n_samples),
        'manufacturer': np.random.choice(manufacturers, n_samples),
        'odometer': np.random.normal(60, 30, n_samples).clip(5, 200),
        'condition': np.random.choice(conditions, n_samples),
        'fuel': np.random.choice(fuel_types, n_samples),
        'transmission': np.random.choice(transmissions, n_samples),
        'type': np.random.choice(vehicle_types, n_samples),
        'age': 0
    })
    
    data['age'] = 2024 - data['year']
    data['posting_year'] = 2023
    
    st.session_state.data_source = "Sample Data (Demo - 5K records)"
    return data

# Initialize data with loading indicator
if 'data' not in st.session_state:
    with st.spinner('Loading vehicle data... This may take a moment.'):
        st.session_state.data = load_data_sample()
        st.session_state.data_loaded = st.session_state.data is not None

data = st.session_state.data

if not st.session_state.get('data_loaded', False):
    st.error("❗ Dataset not loaded. Please refresh the page to try again.")
    st.stop()

# Display data source info
if 'data_source' in st.session_state:
    st.sidebar.success(f"**Data source:** {st.session_state.data_source}")
    
    if 'data_info' in st.session_state:
        info = st.session_state.data_info
        with st.sidebar.expander("Data Info"):
            st.write(f"**Original dataset:** {info['original_size']:,} rows")
            st.write(f"**Sample size:** {info['sample_size']:,} rows")
            st.write(f"**Sample ratio:** {info['reduction_ratio']:.1%}")

# Load or train model (cached and optimized)
@st.cache_resource
def load_model():
    """Load or train the prediction model"""
    model_path = 'rf_model_optimized.joblib'
    
    # Check if saved model exists and is recent
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except:
            pass  # If loading fails, retrain
    
    # Prepare data for modeling
    with st.spinner('Training prediction model...'):
        model_data = data.copy()
        
        # Ensure we have enough data
        if len(model_data) < 100:
            st.error("Not enough data to train model reliably.")
            return None
        
        X = model_data.drop('price', axis=1)
        y = model_data['price']
        
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
            OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20)
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create and train Random Forest model (optimized parameters)
        rf_pipeline = make_pipeline(
            preprocessor,
            RandomForestRegressor(
                n_estimators=100,  # Increased back for better accuracy
                max_depth=15,      # Increased for better performance with more data
                min_samples_leaf=5, # Reduced for better fitting
                random_state=42,
                n_jobs=-1
            )
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        rf_pipeline.fit(X_train, y_train)
        
        # Calculate and store metrics
        train_score = rf_pipeline.score(X_train, y_train)
        test_score = rf_pipeline.score(X_test, y_test)
        y_pred = rf_pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        st.session_state.model_metrics = {
            'train_r2': train_score,
            'test_r2': test_score,
            'mae': mae,
            'rmse': rmse
        }
        
        # Save the model
        joblib.dump(rf_pipeline, model_path)
        return rf_pipeline

# Load model
model = load_model()
if model is None:
    st.error("Could not train prediction model. Please check the data.")
    st.stop()

# Data Exploration Section
if app_mode == "Data Exploration":
    st.header("Data Exploration")
    
    # Data overview
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Listings", f"{len(data):,}")
    col2.metric("Features Available", len(data.columns))
    col3.metric("Average Price", f"${data['price'].mean():,.0f}")
    
    # Show raw data option
    if st.checkbox("Show sample data"):
        st.subheader("Sample Data")
        st.write(data.head(10))
        
        # Data types and missing values
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            st.write(data.dtypes)
        
        with col2:
            st.write("**Missing Values:**")
            missing = data.isnull().sum()
            st.write(missing[missing > 0])
    
    # Price distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['price'], bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Vehicle Prices')
    ax.set_xlabel('Price ($)')
    ax.axvline(data['price'].median(), color='red', linestyle='--', label=f'Median: ${data["price"].median():,.0f}')
    ax.legend()
    st.pyplot(fig)
    
    # Top manufacturers
    if 'manufacturer' in data.columns:
        st.subheader("Top Manufacturers")
        top_n = st.slider("Number of manufacturers to show", 5, 15, 10)
        fig, ax = plt.subplots(figsize=(10, 6))
        manufacturer_counts = data['manufacturer'].value_counts().head(top_n)
        manufacturer_counts.plot(kind='bar', ax=ax)
        ax.set_title(f'Top {top_n} Vehicle Manufacturers')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Price relationships
    st.subheader("Price Relationships")
    
    # Numeric features for comparison
    numeric_features = data.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != 'price']
    
    if len(numeric_features) > 0:
        feature = st.selectbox("Select feature to compare with price", numeric_features)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot with sample of data for performance
        sample_data = data.sample(min(1000, len(data)))
        sns.scatterplot(x=feature, y='price', data=sample_data, alpha=0.6, ax=ax)
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Price ($)')
        ax.set_title(f'Price vs {feature.replace("_", " ").title()}')
        
        # Add correlation coefficient
        corr = data[feature].corr(data['price'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        st.pyplot(fig)

# Model Info Section
elif app_mode == "Model Info":
    st.header("Model Information")
    
    st.subheader("Random Forest Regressor")
    st.markdown("""
    - **Algorithm**: Random Forest (Optimized for larger dataset)
    - **Estimators**: 100 trees
    - **Max Depth**: 15
    - **Min Samples Leaf**: 5
    - **Sample Size**: Up to 150,000 vehicles (20% of full dataset)
    - **Features**: Automatically selected based on data availability
    """)
    
    # Model performance metrics
    st.subheader("Model Performance")
    if hasattr(st.session_state, 'model_metrics'):
        metrics = st.session_state.model_metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training R²", f"{metrics['train_r2']:.3f}")
        col2.metric("Test R²", f"{metrics['test_r2']:.3f}")
        col3.metric("Mean Abs Error", f"${metrics['mae']:,.0f}")
        col4.metric("RMSE", f"${metrics['rmse']:,.0f}")
        
        # Interpretation
        if metrics['test_r2'] > 0.8:
            st.success("Model shows excellent predictive performance!")
        elif metrics['test_r2'] > 0.6:
            st.info("Model shows good predictive performance.")
        else:
            st.warning("Model performance is moderate. More data might improve accuracy.")
    
    # Feature importance (simplified)
    st.subheader("Key Factors in Price Prediction")
    st.markdown("""
    Based on the Random Forest model, the most important factors typically include:
    - **Vehicle Age**: Newer vehicles generally command higher prices
    - **Odometer Reading**: Lower mileage typically means higher value
    - **Manufacturer**: Brand reputation affects pricing
    - **Condition**: Vehicle condition significantly impacts price
    - **Year**: Manufacturing year is a key price determinant
    """)

# Prediction Section
else:
    st.header("Make a Price Prediction")
    
    # Get available options from data
    manufacturers = ['Unknown'] + sorted([x for x in data['manufacturer'].unique() if pd.notna(x)])
    conditions = ['Unknown'] + sorted([x for x in data['condition'].unique() if pd.notna(x) and 'condition' in data.columns])
    
    with st.form("prediction_form"):
        st.subheader("Vehicle Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            manufacturer = st.selectbox("Manufacturer", manufacturers)
            year = st.number_input("Year", 
                                 min_value=1980, 
                                 max_value=pd.Timestamp.now().year, 
                                 value=2018)
            odometer_miles = st.number_input("Odometer (miles)", 
                                           min_value=0, 
                                           max_value=500000, 
                                           value=50000)
        
        with col2:
            if 'condition' in data.columns and len(conditions) > 1:
                condition = st.selectbox("Condition", conditions)
            else:
                condition = 'good'
                st.info("Using default condition: good")
            
            if 'fuel' in data.columns:
                fuel_types = ['Unknown'] + sorted([x for x in data['fuel'].unique() if pd.notna(x)])
                fuel = st.selectbox("Fuel Type", fuel_types)
            else:
                fuel = 'gas'
            
            if 'transmission' in data.columns:
                transmissions = ['Unknown'] + sorted([x for x in data['transmission'].unique() if pd.notna(x)])
                transmission = st.selectbox("Transmission", transmissions)
            else:
                transmission = 'automatic'
        
        submitted = st.form_submit_button("Predict Price", type="primary")
    
    if submitted:
        try:
            # Create input DataFrame with available features
            current_year = pd.Timestamp.now().year
            age = current_year - year
            odometer_k = odometer_miles / 1000
            
            # Start with basic features
            input_dict = {
                'year': year,
                'manufacturer': manufacturer if manufacturer != 'Unknown' else 'other',
                'odometer': odometer_k,
                'age': age,
                'posting_year': current_year
            }
            
            # Add optional features if they exist in the training data
            training_columns = data.drop('price', axis=1).columns
            
            if 'condition' in training_columns:
                input_dict['condition'] = condition if condition != 'Unknown' else 'good'
            if 'fuel' in training_columns:
                input_dict['fuel'] = fuel if fuel != 'Unknown' else 'gas'
            if 'transmission' in training_columns:
                input_dict['transmission'] = transmission if transmission != 'Unknown' else 'automatic'
            
            # Add any other required columns with default values
            for col in training_columns:
                if col not in input_dict:
                    if data[col].dtype == 'object':
                        input_dict[col] = 'unknown'
                    else:
                        input_dict[col] = 0.0
            
            # Create DataFrame
            input_data = pd.DataFrame([input_dict])
            
            # Ensure column order matches training data
            input_data = input_data.reindex(columns=training_columns, fill_value='unknown')
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result with styling
            st.success(f"## 💰 Predicted Price: ${prediction:,.0f}")
            
            # Show confidence interval
            st.info(f"📊 **Estimated price range:** ${prediction*0.85:.0f} - ${prediction*1.15:.0f}")
            
            # Additional insights
            st.markdown("### 📈 Price Insights")
            
            # Compare to similar vehicles
            similar_vehicles = data[
                (data['manufacturer'] == (manufacturer if manufacturer != 'Unknown' else data['manufacturer'].mode().iloc[0])) &
                (abs(data['year'] - year) <= 2) &
                (abs(data['odometer'] - odometer_k) <= 20)
            ]
            
            if len(similar_vehicles) > 0:
                avg_similar = similar_vehicles['price'].mean()
                if prediction > avg_similar * 1.1:
                    st.warning(f"This prediction is higher than similar vehicles (avg: ${avg_similar:,.0f})")
                elif prediction < avg_similar * 0.9:
                    st.success(f"This prediction is lower than similar vehicles (avg: ${avg_similar:,.0f})")
                else:
                    st.info(f"This prediction aligns with similar vehicles (avg: ${avg_similar:,.0f})")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.error("Please check your inputs and try again.")
            
            # Debug information
            if st.checkbox("Show debug info"):
                st.write("Input data shape:", input_data.shape)
                st.write("Training data columns:", list(training_columns))
                st.write("Input data:")
                st.write(input_data)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p>© 2024 Vehicle Price Predictor | Optimized for performance with sampled data</p>
<p><small>This app uses a sample of vehicle listing data for fast predictions</small></p>
</div>
""", unsafe_allow_html=True)
