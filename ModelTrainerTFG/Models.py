from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from utils import detect_delimiter
from sklearn.cluster import KMeans

def preprocess_data(data):
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Check for non-numerical values
    for col in categorical_cols:
        unique_values = data[col].unique()
        if len(unique_values) > 20:  # Assuming more than 20 unique values is not suitable for one-hot encoding
            raise ValueError(f"Column '{col}' has too many unique values for one-hot encoding. Find more about this in the **Need help with your data** button below")

    # Fitting only on the training data
    onehot_encoder.fit(data[categorical_cols])

    # Transforming data using the already fitted encoder
    encoded_categories = onehot_encoder.transform(data[categorical_cols]).toarray()
    # Get feature names from encoder and format them
    feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_categories, columns=feature_names, index=data.index)

    # Drop original categorical columns and concatenate the new encoded columns
    data = data.drop(categorical_cols, axis=1)
    data = pd.concat([data, encoded_df], axis=1)

    # Fill missing numeric values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    return data
def test_out(selected_input_columns, selected_output_column=None, model_id=None):
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.markdown("##### Test with single input:")
        user_input_data = {}
        for column in selected_input_columns:
            unique_key = f"test_input_{model_id}_{column}"
            user_input_data[column] = st.text_input(f"Enter {column}:", key=unique_key)

        if st.button("Get prediction", key=f'classify_single_{model_id}'):
            # Check if all input values are filled
            if any(value == "" for value in user_input_data.values()):
                st.error("Please fill in all the input fields.")
            else:
                try:
                    model = st.session_state['model']
                    prediction_input = pd.DataFrame([user_input_data])
                    prediction = model.predict(prediction_input)
                    if st.session_state.model_type == 'Classification Model' or st.session_state.model_type == 'Regression Model':
                        st.info(f"The predicted {selected_output_column} is {prediction[0]}")
                    elif st.session_state.model_type == 'Clustering Model':
                        st.info(f"The predicted cluster for this data is {prediction[0]}")
                except KeyError:
                    st.error("No model found. Please train a model first.")

    with col2:
        st.markdown("##### Test by uploading a CSV file:")
        uploaded_file = st.file_uploader("Upload a CSV file for batch classification", type=["csv"])
        if uploaded_file:
            delimiter = detect_delimiter(uploaded_file)
            data = pd.read_csv(uploaded_file, delimiter=delimiter)
            data.replace('', pd.NA, inplace=True)
            data.dropna(inplace=True)

            if st.session_state.model_type == 'Regression Model' or st.session_state.model_type == 'Clustering Model':
                data = preprocess_data(data)
            if not set(selected_input_columns).issubset(data.columns):
                st.error("Uploaded file is missing some required columns.")
            else:
                model = st.session_state.get('model', None)
                if model:
                    predictions = model.predict(data[selected_input_columns])
                    if st.session_state.model_type == 'Classification Model' or st.session_state.model_type == 'Regression Model':
                        data.insert(0, selected_output_column + '_prediction', predictions)
                    elif st.session_state.model_type == 'Clustering Model':
                        data.insert(0, 'Predicted cluster', predictions)

                    st.dataframe(data)

                    # Download CSV button
                    csv = data.to_csv(index=False)
                    st.download_button("Download CSV with predictions", csv, "predictions.csv", "text/csv")

# CLASSIFICATION MODEL
def classification_model(data, selected_input_columns, selected_output_column, config):
    # Define a column transformer that applies TF-IDF vectorization to each selected text column separately
    column_transformer = ColumnTransformer(
        [(col, TfidfVectorizer(), col) for col in selected_input_columns],
        remainder='drop'
    )
    # Create a pipeline that includes vectorization and classification
    model = Pipeline([
        ('vectorizer', column_transformer),
        ('classifier', MultinomialNB(alpha=config['alpha'], fit_prior=config['fit_prior']))
    ])
    # Prepare input and output
    input_data = data[selected_input_columns]
    output_data = data[selected_output_column]

    # Training
    model.fit(input_data, output_data)

    return model

# REGRESSION MODEL
def configure_regression_model():
    regression_types = ['Linear Regression', 'Polynomial Regression', 'Random Forest Regression']
    regression_type = st.radio("Choose the type of regression:", regression_types, key='regression_type_selection')

    config = {}
    if regression_type == 'Polynomial Regression':
        config['degree'] = st.number_input("Degree of the polynomial:", min_value=1, max_value=6, value=2, key='poly_degree')
        st.info("You can leave the default value or change it to adapt to your needs")
    elif regression_type == 'Random Forest Regression':
        config['n_estimators'] = st.number_input("Number of estimators:", min_value=10, max_value=1000, value=100, key='rf_estimators')
        config['random_state'] = st.number_input("Random State", min_value=0, value=42, key='rf_random_state')
        st.info("You can leave the default values or change them to adapt to your needs")

    return regression_type, config
def regression_model(data, input_columns, output_column, regression_type, degree=None, n_estimators=None, random_state=None):
    # Prepare input and output
    input_data = data[input_columns]
    output_data = data[output_column]

    # Create and train the model
    if regression_type == 'Linear Regression':
        model = LinearRegression()
    elif regression_type == 'Polynomial Regression':
        degree = 2
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    elif regression_type == 'Random Forest Regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(input_data, output_data)
    return model

#CLUSTERING MODEL
def clustering_model(data, selected_input_columns, config):
    input_data = data[selected_input_columns]
    model = KMeans(n_clusters=config['n_clusters'], random_state=config['random_state'])
    model.fit(input_data)
    return model