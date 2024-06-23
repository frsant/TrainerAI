import streamlit as st
import pandas as pd
import pickle
from utils import detect_delimiter, plot_confusion_matrix
from Models import classification_model, regression_model, test_out, configure_regression_model, preprocess_data, \
    clustering_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, \
    explained_variance_score, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from pandas.errors import EmptyDataError, ParserError, DtypeWarning
import altair as alt
import matplotlib.pyplot as plt
from auth import create_user, authenticate_user, get_user_profile
from db import save_model, get_user_models
import json

def load_css(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
def save_model_to_profile(model, evaluation, model_name, test_data=None, regression_type=None, input=None, output=None):
    if 'user_id' in st.session_state:
        model_data = pickle.dumps(model)
        evaluation_json = json.dumps(evaluation)
        test_data_pickle = pickle.dumps(test_data) if test_data is not None else None
        input_json = json.dumps(input) if input is not None else None
        save_model(st.session_state['user_id'], st.session_state.model_type, model_name, model_data, evaluation_json,
                   test_data_pickle, regression_type, input_json, output)
    else:
        st.warning("User not logged in. Cannot save model to profile.")


def home():
    col1, col2, col3 = st.columns([0.5, 0.15, 0.35])
    with col1:
        st.markdown('<h1 class="home-title">Train your own AI model</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="home-subtitle">Create, train and implement AI models tailored to your needs, without any coding required.</p>',
            unsafe_allow_html=True)

        if st.button("Get Started", type="primary"):
            st.session_state.current_step = 1
            st.session_state.current_page = "trainer"
            st.rerun()

    with col3:
        st.image("images/HomeIcon.svg", use_column_width=True)
def myModels():
    if 'username' in st.session_state:
        user_profile = get_user_profile(st.session_state['username'])
        col1, col2 = st.columns([0.09, 0.8])
        with col1:
            st.image("images/Profile.png")
        with col2:
            st.title(f"{user_profile['username']}'s Profile")
        user_models = get_user_models(user_profile['id'])
        st.write("")
        if user_models:
            st.subheader("Saved Models")
            for model in user_models:
                if model['model_type'] == 'Classification Model':
                    image_path = 'images/Classification.png'
                elif model['model_type'] == 'Regression Model':
                    image_path = 'images/Regression.png'
                elif model['model_type'] == 'Clustering Model':
                    image_path = 'images/Clustering.png'

                col1, col2, col3 = st.columns([0.14, 0.60, 0.26])
                with col1:
                    st.image(image_path, use_column_width=True)
                with col2:
                    created_at = model['created_at']
                    created_at_time = created_at.strftime('%H:%M')
                    created_at_date = created_at.strftime('%Y-%m-%d')
                    st.write(f"#### {model['model_name']}")
                    if model['model_type'] == 'Regression Model':
                        st.write(f"{model['regression_type']} Model | {created_at_date} {created_at_time}")
                    else:
                        st.write(f"{model['model_type']} | {created_at_date} {created_at_time}")
                with col3:
                    st.markdown(
                        """
                        <style>
                        .centered-button {
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            height: 100%;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown('<div class="centered-button">', unsafe_allow_html=True)
                    if f"show_predictions_{model['id']}" not in st.session_state:
                        st.session_state[f"show_predictions_{model['id']}"] = False

                    if st.button("Make predictions", key=f"predictions_button_{model['id']}", type="primary"):
                        st.session_state[f"show_predictions_{model['id']}"] = not st.session_state[
                            f"show_predictions_{model['id']}"]

                if st.session_state[f"show_predictions_{model['id']}"]:
                    if model['model_type'] == 'Classification Model' or model['model_type'] == 'Regression Model':
                        test_out(json.loads(model['input']), model['output'])
                    elif model['model_type'] == 'Clustering Model':
                        test_out(json.loads(model['input']))

                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("Evaluation"):
                    evaluation = model['evaluation']
                    if isinstance(evaluation, str):
                        evaluation = json.loads(evaluation)  # Deserialize evaluation if it's a JSON string
                    if model.get('model_type') == 'Classification Model':
                        accuracy = evaluation.get('accuracy')
                        precision = evaluation.get('precision')
                        recall = evaluation.get('recall')
                        f1 = evaluation.get('f1')
                        accuracy_percentage = evaluation.get('accuracy_percentage')
                        conf_matrix = evaluation.get('conf_matrix')

                        st.subheader("Evaluation:")
                        metrics_df = pd.DataFrame({
                            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                            'Value': [accuracy, precision, recall, f1]
                        })

                        col1, col2 = st.columns(2, gap="medium")
                        with col1:
                            # Display metrics as a table
                            st.write("##### Model Performance Metrics for Advanced Users")
                            st.dataframe(metrics_df, use_container_width=True)
                        with col2:
                            st.markdown("##### Beginner-Friendly Explanations")
                            st.info(f"**Accuracy:** Out of all predictions, {accuracy_percentage:.2f}% were correct.")
                            st.info(f"**Recall:** Number of correctly identified positive instances.")
                            st.info(f"**F1 Score:** a combination of precision and recall.")

                        st.markdown("---", unsafe_allow_html=True)
                        col1, col2 = st.columns([0.4, 0.6])
                        with col2:
                            fig, ax = plt.subplots()
                            plot_confusion_matrix(np.array(conf_matrix), ax)
                            st.pyplot(fig)
                        with col1:
                            st.markdown("##### Confusion Matrix")
                            st.info(
                                "The confusion matrix shows how many predictions were correct and where the errors occurred. Each row represents the actual class, and each column represents the predicted class. The diagonal elements show the number of correct predictions, while the off-diagonal elements show where the model made mistakes.")
                    if model.get('model_type') == 'Regression Model':

                        r2 = evaluation.get('r2')
                        mae = evaluation.get('mae')
                        mse = evaluation.get('mse')
                        mape = evaluation.get('mape')
                        rmse = evaluation.get('rmse')
                        explained_variance = evaluation.get('explained_variance')

                        st.subheader("Evaluation:")
                        # Create a DataFrame for the metrics
                        metrics_df = pd.DataFrame({
                            'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)',
                                       'Mean Absolute Error (MAE)',
                                       'MAE percentage (MAPE)', 'R² Score', 'Explained Variance'],
                            'Value': [mse, rmse, mae, mape, r2, explained_variance]
                        })

                        col1, col2 = st.columns(2, gap="medium")
                        with col1:
                            # Display metrics as a table
                            st.write("##### Model Performance Metrics for Advanced Users")
                            st.dataframe(metrics_df)
                        with col2:
                            st.markdown("##### Beginner-Friendly Explanations")
                            st.write(" ")
                            st.write(" ")
                            st.write(" ")
                            st.info(
                                f"**Mean Absolute Error (MAE):** On average, the model's predictions are off by {mae:.2f} units,  {mape:.2f}%.")
                            st.info(
                                f"**R² Score:** {r2:.2f}, which means the model explains {r2 * 100:.2f}% of the variability in the data.")

                        col1, col2 = st.columns([0.8, 0.2])
                        with col1:
                            test_data = pickle.loads(model['test_data']) if model['test_data'] else None
                            if test_data is not None and 'regression_type' in model:
                                # Bar chart of actual vs predicted values with color difference
                                sample_size = 100  # Define a sample size
                                sampled_data = test_data.sample(sample_size, random_state=42) if len(
                                    test_data) > sample_size else test_data
                                model_instance = pickle.loads(model['model'])
                                input_columns = json.loads(model['input'])
                                sampled_predictions = model_instance.predict(sampled_data[input_columns])

                                actual_values = sampled_data[model['output']].values
                                predicted_values = sampled_predictions

                                # Create a DataFrame for Altair
                                chart_data = pd.DataFrame({
                                    'Index': np.arange(len(sampled_data)),
                                    'Actual': actual_values,
                                    'Predicted': predicted_values
                                })

                                base = alt.Chart(chart_data).encode(
                                    x=alt.X('Index:O', title='Index')
                                )

                                actual_column_title = f'Value of {model["output"]}'
                                actual_bars = base.mark_bar(color='blue', opacity=0.7).encode(
                                    y=alt.Y('Actual:Q', title=actual_column_title)
                                )

                                predicted_bars = base.mark_bar(color='orange', opacity=0.7).encode(
                                    y=alt.Y('Predicted:Q')
                                )

                                difference_bars = base.mark_bar().encode(
                                    y=alt.Y('Actual:Q', title=actual_column_title),
                                    y2='Predicted:Q',
                                    color=alt.condition(
                                        alt.datum.Actual > alt.datum.Predicted,
                                        alt.value('red'),  # The positive color
                                        alt.value('green')  # The negative color
                                    )
                                )

                                chart = actual_bars + predicted_bars + difference_bars
                                st.altair_chart(chart, use_container_width=True)
                        with col2:
                            st.write(" ")
                            st.write(" ")
                            st.error("The prediction overestimated the real value")
                            st.success("The prediction underestimated the real value")
                    elif model.get('model_type') == 'Clustering Model':
                        # Process clustering model evaluation metrics
                        silhouette_avg = evaluation.get('silhouette_avg')
                        davies_bouldin_index = evaluation.get('davies_bouldin_index')
                        calinski_harabasz_score = evaluation.get('calinski_harabasz_score')

                        st.subheader("Evaluation:")
                        metrics_df = pd.DataFrame({
                            'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Score'],
                            'Value': [silhouette_avg, davies_bouldin_index, calinski_harabasz_score]
                        })

                        col1, col2 = st.columns(2, gap="medium")
                        with col1:
                            st.write("##### Model Performance Metrics for Advanced Users")
                            st.dataframe(metrics_df)
                        with col2:
                            st.markdown("##### Beginner-Friendly Explanations")
                            st.info(
                                f"**Silhouette Score:** Indicates how similar an object is to its own cluster compared to other clusters. Higher is better.")
                            st.info(
                                f"**Davies-Bouldin Index:** Measures the average 'similarity ratio' of each cluster with the one that is most similar to it. Lower is better.")
                            st.info(
                                f"**Calinski-Harabasz Score:** Ratio of the sum of between-clusters dispersion to within-cluster dispersion. Higher is better.")

                        col1, col2 = st.columns([0.8, 0.2])
                        with col1:
                            test_data = pickle.loads(model['test_data']) if model['test_data'] else None
                            if test_data is not None:
                                model_instance = pickle.loads(model['model'])
                                input_columns = json.loads(model['input'])  # Parse input columns
                                cluster_labels = model_instance.predict(test_data[input_columns])

                                # Create a DataFrame for Altair
                                cluster_data = test_data.copy()
                                cluster_data['Cluster'] = cluster_labels

                                # Scatter plot of clusters
                                chart = alt.Chart(cluster_data).mark_circle(size=60).encode(
                                    x='Age:Q',
                                    y='Annual Income (k$):Q',
                                    color='Cluster:N',
                                    tooltip=['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']
                                ).interactive()

                                st.altair_chart(chart, use_container_width=True)
                        with col2:
                            st.write(" ")
                            st.write(" ")
                            st.success("Clusters plotted successfully")

                st.markdown("---", unsafe_allow_html=True)
    else:
        st.write("Not logged in")
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", key="login_login", type="primary"):
        user = authenticate_user(username, password)
        if user:
            st.session_state['username'] = user['username']
            st.session_state['user_id'] = user['id']  # Store user_id in session state
            st.session_state.current_page = "home"
            st.rerun()
    st.write("---", unsafe_allow_html=True)
    col1, col2 = st.columns([0.24,0.75])
    with col1:
        st.write("You don't have an account yet?")
    with col2:
        if st.button("Sign-Up", key="sign_up_login"):
            st.session_state.current_page = "signup"
            st.rerun()
def signup():
    st.title("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    email = st.text_input("Email")

    if st.button("Sign Up", key="signup_signup", type="primary"):
        create_user(username, password, email)
        st.success("User created successfully. Please login.")
        st.session_state.current_page = "login"
        st.rerun()
    st.write("---")
    col1, col2 = st.columns([0.23, 0.75])
    with col1:
        st.write("Do you already have an account?")
    with col2:
        if st.button("Login", key="login_sign_up"):
            st.session_state.current_page = "login"
            st.rerun()
def trainer():
    # Continue with the model training steps if no other page is selected
    st.title("Let's start training!")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Step 1", use_container_width=True, key="step1", type="primary"):
            st.session_state.current_step = 1
    with col2:
        if st.button("Step 2", disabled=st.session_state.current_step == 1,
                     use_container_width=True, key="step2", type="primary"):
            st.session_state.current_step = 2
    with col3:
        if st.button("Step 3", disabled=st.session_state.current_step in [1, 2],
                     use_container_width=True, key="step3", type="primary"):
            st.session_state.current_step = 3
    with col4:
        if st.button("Step 4", disabled=st.session_state.current_step in [1, 2, 3],
                     use_container_width=True, key="step4", type="primary"):
            st.session_state.current_step = 4
    with col5:
        if st.button("Step 5", disabled=st.session_state.current_step in [1, 2, 3, 4],
                     use_container_width=True, key="step5", type="primary"):
            st.session_state.current_step = 5

    # Display the corresponding step
    if st.session_state.current_step == 1:
        choose_model_type()
    elif st.session_state.current_step == 2:
        upload_data()
    elif st.session_state.current_step == 3:
        configure_model()
    elif st.session_state.current_step == 4:
        train_model()
    elif st.session_state.current_step == 5:
        test_model()


# All the different training steps
def choose_model_type():
    st.subheader("Step 1: Choose the Model Type")
    model_types = ['Classification Model', 'Regression Model', 'Clustering Model', 'Neural Networks']
    model_type = st.radio("Select which type of model you want to train:", model_types)

    col1, col2 = st.columns([0.11, 0.79])
    with col1:
        if st.button("Next", type="primary", use_container_width=True):
            if model_type in ['Classification Model', 'Regression Model', 'Clustering Model']:
                global is_fitted
                is_fitted = False  # Reset fitting status when switching models
                st.session_state.model_type = model_type
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.error("This model type is not yet implemented.")

    st.markdown("---", unsafe_allow_html=True)
    if st.button("Need help making a decision? ⬇️"):
        st.session_state.show_explanations = not st.session_state.get('show_explanations', False)
        st.rerun()

    if st.session_state.get('show_explanations', False):
        # Load the trained model
        with open("model_recommendation_model.pkl", "rb") as file:
            model = pickle.load(file)

        def recommend_model(description):
            return model.predict([description])[0]

        st.markdown("#### Model Recommendation System")

        user_input = st.text_area(
            "Describe what you want to predict and we will give you a suggestion on what model to choose")

        if st.button("Get Recommendation"):
            if user_input:
                recommendation = recommend_model(user_input)
                st.info(f"Recommendation: {recommendation}")
            else:
                st.write("Please enter a description to get a recommendation.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image("images/classification.png", use_column_width=True)
            st.markdown("<h4 style='text-align: center;'>Classification</h4>", unsafe_allow_html=True)
            st.write(
                "Use this model when you want to categorize data into different classes. This model is typically used for documents and words rather than numerical data.")
            st.write("**Example:** classifying emails as spam or not spam.")
            st.write(
                "**Data requirements:** the data you will use for training must label each element into one of the classes.")
            st.write("**Model:** Multinomial Naive Bayes.")

        with col2:
            st.image("images/regression.png", use_column_width=True)
            st.markdown("<h4 style='text-align: center;'>Regression</h4>", unsafe_allow_html=True)
            st.write("This model is used to predict continuous values. Usually for numerical values")
            st.write("**Example:** predicting the price of a house based on its features.")
            st.write("**Data requirements:** the data must have continuous output values for training.")
            st.write(
                "**Model:** Linear regression, Polynomial regression and Random Forest regression are available for selection.")

        with col3:
            st.image("images/clustering.png", use_column_width=True)
            st.markdown("<h4 style='text-align: center;'>Clustering</h4>", unsafe_allow_html=True)
            st.write("Use this model to group similar items together.")
            st.write("**Example:** customer segmentation based on purchasing behavior.")
            st.write(
                "**Data requirements:** the data should have features that can define similarities among elements. Contrary to classification, it does not need to have predefined labels, the model will label the data without need of input labels to learn from.")
            st.write("**Model:** K-Means clustering.")

        with col4:
            st.image("images/neuralnetworks.png", use_column_width=True)
            st.markdown("<h4 style='text-align: center;'>Neural Networks</h4>", unsafe_allow_html=True)
            st.write("Advanced models used for complex pattern recognition tasks.")
            st.write("**Example:** image or speech recognition.")
            st.write(
                "**Data requirements:** large datasets with labeled examples for training. The data should be processed and normalized.")
def upload_data():
    st.subheader("Step 2: Upload the Data to Train the Model")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            delimiter = detect_delimiter(uploaded_file)
            data = pd.read_csv(uploaded_file, delimiter=delimiter)
            data.replace('', pd.NA, inplace=True)
            data.dropna(inplace=True)

            # Validation for the number of columns
            if st.session_state.model_type in ['Regression Model', 'Classification Model']:
                if len(data.columns) < 2:
                    raise ValueError("The uploaded file must contain at least two columns, one input and one output.")

            if st.session_state.model_type == 'Regression Model' or st.session_state.model_type == 'Clustering Model':
                data = preprocess_data(data)
                st.markdown("##### Data preview after preprocessing:")
                st.dataframe(data)
            elif st.session_state.model_type == 'Classification Model':
                st.write("##### Data preview for classification:")
                st.dataframe(data)

            input_columns = st.multiselect("Select input columns:", data.columns)
            # Only show output_column selection for Regression and Classification models
            if st.session_state.model_type in ['Regression Model', 'Classification Model']:
                output_column_options = [' '] + [col for col in data.columns if col not in input_columns]
                output_column = st.selectbox("Select output column (target variable):", output_column_options)
                if output_column == ' ':
                    output_column = None
            else:
                output_column = None

            # Validate input columns
            for col in input_columns:
                if st.session_state.model_type == 'Regression Model':
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(
                            f"The input column '{col}' must contain numerical data for regression models. Find more about this in the **Need help with your data** button below")
                elif st.session_state.model_type == 'Classification Model':
                    if pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(
                            f"The input column '{col}' must contain non-numerical data for classification models. Find more about this in the **Need help with your data** button below")
                elif st.session_state.model_type == 'Clustering Model':
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(
                            f"The input column '{col}' must contain numerical data for clustering models. Find more about this in the **Need help with your data** button below")

            # Validation for the output column
            if st.session_state.model_type == 'Classification Model' and output_column:
                if not (isinstance(data[output_column].dtype, pd.CategoricalDtype)):
                    if pd.api.types.is_integer_dtype(data[output_column]) or pd.api.types.is_float_dtype(
                            data[output_column]):
                        data[output_column] = data[output_column].astype('category')
                    else:
                        raise ValueError(
                            "The selected output column for classification must contain categorical data. Find more about this in the **Need help with your data** button below")
            elif st.session_state.model_type == 'Regression Model' and output_column:
                if not pd.api.types.is_numeric_dtype(data[output_column]):
                    raise ValueError(
                        "The selected output column for regression must contain numerical data. Find more about this in the **Need help with your data** button below")

            if input_columns and (output_column or st.session_state.model_type == "Clustering Model"):
                st.session_state.data = data
                st.session_state.input_columns = input_columns
                if st.session_state.model_type == "Classification Model" or st.session_state.model_type == "Regression Model":
                    st.session_state.output_column = output_column

                st.write("")
                st.write("")
                col1, col2, col3 = st.columns([0.11, 0.78, 0.11])
                if col1.button("Previous", use_container_width=True):
                    st.session_state.current_step = 1
                    st.rerun()
                if col3.button("Next", type="primary", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()
        except EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        except ParserError:
            st.error("The uploaded file could not be parsed. Please ensure it is a valid CSV file.")
        except UnicodeDecodeError:
            st.error("The uploaded file has an encoding issue. Please ensure it is a UTF-8 encoded CSV file.")
        except DtypeWarning:
            st.warning("There are issues with the data types in your file. Please check your data.")
        except ValueError as ve:
            st.error(f"A value error occurred: {ve}")
        except TypeError as te:
            st.error(f"A type error occurred: {te}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    else:
        col1, col2 = st.columns([0.11, 0.89])
        with col1:
            if st.button("Previous"):
                st.session_state.current_step = 1
                st.rerun()

    st.markdown("---", unsafe_allow_html=True)
    if st.button("Need help with your data? ⬇️"):
        st.session_state.show_data_help = not st.session_state.get('show_data_help', False)
        st.rerun()

    if st.session_state.get('show_data_help', False):
        if st.session_state.model_type == 'Classification Model':
            st.write("### Classification Model Data Requirements")
            st.write(
                "Your dataset should contain **non-numerical input** features and a categorical target or output variable also known as label (which is the value that the model will predict for new data when it has been trained). If you wish to use numerical values try the regression or clustering models.")
            st.write("**Example Columns:**")
            st.write("- `EmailContent`: Text content of the email")
            st.write("- `SenderDomain`: Domain of the sender")
            st.write("- `IsSpam`: The target variable, which is either 'spam' or 'not spam'")
        elif st.session_state.model_type == 'Regression Model':
            st.write("### Regression Model Data Requirements")
            st.write(
                "Your dataset should contain **numerical input** features and a numerical target variable (which is the value that the model will predict for new data when it has been trained). The model supports non-numerical input only if it is a small number of repeated values like `OceanProximity` in the example.")
            st.write("**Example Columns:**")
            st.write("- `SquareFootage`: Size of the house in square feet")
            st.write("- `NumBedrooms`: Number of bedrooms")
            st.write("- `NumBathrooms`: Number of bathrooms")
            st.write("- `OceanProximity`: Can be >1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN ")
            st.write("- `HousePrice`: The target variable, house price in dollars")
        elif st.session_state.model_type == 'Clustering Model':
            st.write("### Clustering Model Data Requirements")
            st.write(
                "Your dataset should contain features that can define similarities among elements. The model will group similar items together. Clustering models do not require a target or output variable since they find patterns and groupings within the input data. Your dataset should contain **numerical input features**, the model supports non-numerical input only if it is a small number of repeated values like `Customer Type` in the example.")
            st.write("**Example Columns:**")
            st.write("- `Age`: Age of customers")
            st.write("- `Annual Income`: Income of customers")
            st.write("- `Spending Score`: Score based on spending behavior")
            st.write("- `Region`: Geographical region of customers")
            st.write("- `Customer Type`: Can be Regular, VIP, or New")
            st.write(
                "**Example Use Case:** Segmenting customers into different groups based on their purchasing behavior and demographics to target marketing strategies more effectively.")
def configure_model():
    st.subheader("Step 3: Model Configuration")

    if st.session_state.model_type == 'Classification Model':
        st.write("Modify the parameters for the Multinomial Naive Bayes model:")
        alpha = st.number_input("Alpha (smoothing parameter)", min_value=0.0, value=1.0, max_value=1.0, step=0.1)
        fit_prior = st.checkbox("Fit prior probabilities", value=True)
        st.info("You can leave the default configuration or change it to adapt to your needs")

        st.session_state.config = {
            'alpha': alpha,
            'fit_prior': fit_prior
        }
    elif st.session_state.model_type == 'Regression Model':
        st.session_state.regression_type, st.session_state.config = configure_regression_model()
    elif st.session_state.model_type == 'Clustering Model':
        n_clusters = st.number_input("Number of clusters:", min_value=1, value=3, key='n_clusters')
        random_state = st.number_input("Random State", value=42, key='clustering_random_state')
        st.info("You can leave the default values or change them to adapt to your needs")

        st.session_state.config = {
            'n_clusters': n_clusters,
            'random_state': random_state
        }


    col1, col2, col3 = st.columns([0.11, 0.78, 0.11])
    if col1.button("Previous", use_container_width=True):
        st.session_state.current_step = 2
        st.rerun()
    if col3.button("Next", type="primary", use_container_width=True):
        st.session_state.current_step = 4
        st.rerun()

    st.markdown("---", unsafe_allow_html=True)
    if st.button("Need help with configuration? ⬇️"):
        st.session_state.show_config_help = not st.session_state.get('show_config_help', False)
        st.rerun()

    if st.session_state.get('show_config_help', False):
        if st.session_state.model_type == 'Classification Model':
            st.write("### Classification Model Parameters")
            st.write("##### Alpha (Smoothing Parameter)")
            st.write(
                " - **Purpose**: Adds a small probability value to words that don't appear in the training data, this is called smoothing and it prevents the model from assigning a zero probability to unseen words.")
            st.write(
                " - **Example**: If set to 0, no smoothing is applied (a word that does not appear will be assigned 0 probability). If set to a higher value, more smoothing is applied.")
            st.write(
                " - **Recommendation**: Start with the default value of 1.0 and adjust based on model performance.")

            st.write("##### Fit Prior Probabilities")
            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                st.write(
                    " - **Purpose**: Decides whether to use the frequencies of the classes from the training data.")
                st.write(
                    " - **Example**: If set to True, the model will assume the probability of an element being in a class is the one it learned in the training. If set to False, it will asume the probability is uniform throughout all clases.")
                st.write(
                    " - **Recommendation**: Usually, leave this as True unless you have a specific reason to assume that the frequency of data among classes is uniform.")
            with col2:
                st.image("images/PriorProbabilities.png", use_column_width=True)
        elif st.session_state.model_type == 'Regression Model':
            st.write("### Regression Model Types and their Parameters")
            col1, col2, col3 = st.columns(3, gap="large")
            with col1:
                st.image("images/Linear.png", use_column_width=True)
                st.markdown("<h4 style='text-align: center;'>Linear</h4>", unsafe_allow_html=True)
                st.write(" Use when you expect a linear relationship between the input features and the output.")
                st.image("images/LinearRelationship.png", use_column_width=True)
                st.write(
                    "In this example, as the size of the house increases, the market value of the house increases linearly.")
            with col2:
                st.image("images/Polynomial.png", use_column_width=True)
                st.markdown("<h4 style='text-align: center;'>Polynomial</h4>", unsafe_allow_html=True)
                st.write(" Use when the relationship between the input features and the output is non-linear.")
                st.write(
                    "   - **Degree of Polynomial**: Determines the highest power of the input variables. **Default Value**: Degree 2 (quadratic) includes terms like \(x\), \(x^2\), \(y\), \(y^2\) and \(xy\) allowing the model to capture curves and more complex relationships.")
                st.write(
                    " **Recommendation**: Start with a degree of 2 or 3 and increase only if necessary to avoid overfitting.")
            with col3:
                st.image("images/RandomForest.png", use_column_width=True)
                st.markdown("<h4 style='text-align: center;'>Random Forest</h4>", unsafe_allow_html=True)
                st.write(
                    " Use when you want a more robust model that can handle complex relationships and interactions between variables.")
                st.write(
                    "   - **Number of Estimators**: Number of trees in the forest. More trees can improve accuracy but will increase computation time.")
                st.write(
                    "   - **Random State**: Seed for the random number generator to ensure reproducibility. **Default Value**: 42 (You can change it if you want different results on different runs).")
                st.write(
                    " **Recommendation**: Start with 100 estimators and adjust based on performance and computation time (you will see the models performance in Step 4).")
        elif st.session_state.model_type == 'Clustering Model':
            st.write("### Clustering Model Parameters")
            st.write("##### Number of clusters")
            st.write(" - **Purpose**: It is the number of groups the model will attempt to identify in your dataset.")
            st.write(
                " - **Example**: If you're segmenting customers and you choose 3 clusters, the model might group them into 'Low Spend', 'Medium Spend', and 'High Spend' categories.")
            st.write(
                " - **Recommendation**: Start with a small number of clusters, such as 3 to 5, and adjust based on the performance and interpretability of the results.")
            st.write("##### Random State")
            st.write(
                " - **Purpose**: Seed for the random number generator to ensure reproducibility. (You can change it if you want different results on different runs).")
            st.write(
                " - **Recommendation**: Use any fixed value, such as 42, to ensure the same results on different runs and change it in between runs for different initializations and potentially different cluster assignments.")
def train_model():
    st.subheader("Step 4: Training and Evaluation")

    train_data, test_data = train_test_split(st.session_state.data, test_size=0.2, random_state=42, shuffle=True)

    if st.session_state.model_type == 'Classification Model':
        model = classification_model(
            train_data,
            st.session_state.input_columns,
            st.session_state.output_column,
            st.session_state.config,
        )
        st.session_state.model = model

        # Calculate and display metrics
        predictions = model.predict(test_data[st.session_state.input_columns])
        accuracy = accuracy_score(test_data[st.session_state.output_column], predictions)
        precision = precision_score(test_data[st.session_state.output_column], predictions, average='weighted')
        recall = recall_score(test_data[st.session_state.output_column], predictions, average='weighted')
        f1 = f1_score(test_data[st.session_state.output_column], predictions, average='weighted')
        conf_matrix = confusion_matrix(test_data[st.session_state.output_column], predictions)
        accuracy_percentage = int(accuracy * 100)

        st.success("Model trained successfully!")
        evaluation = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'conf_matrix': conf_matrix.tolist(),
            'accuracy_percentage': accuracy_percentage
        }
        col1, col2, col3, col4 = st.columns([2, 0.55, 1.3, 0.35])
        with col1:
            st.write("")
            model_data = pickle.dumps(st.session_state.model)
            st.download_button("Download trained model", model_data, "trained_model.pkl", "application/octet-stream")
        with col3:
            model_name = st.text_input("Name your model to save it to your profile")
        with col4:
            st.write("")
            st.write("")
            if st.button("Save"):
                if model_name:
                    save_model_to_profile(model, evaluation, model_name, input=st.session_state.input_columns, output=st.session_state.output_column)

                    st.success("Model saved to profile")
                else:
                    st.error("You must name the model for saving")

        st.markdown("---")
        st.subheader("Evaluation:")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            # Display metrics as a table
            st.write("##### Model Performance Metrics for Advanced Users")
            st.dataframe(metrics_df, use_container_width=True)
        with col2:
            st.markdown("##### Beginner-Friendly Explanations")
            st.info(f"**Accuracy:** Out of all predictions, {accuracy_percentage:.2f}% were correct.")
            st.info(f"**Recall:** Number of correctly identified positive instances.")
            st.info(f"**F1 Score:** a combination of precision and recall.")

        st.markdown("---", unsafe_allow_html=True)
        col1, col2 = st.columns([0.4, 0.6])
        with col2:
            fig, ax = plt.subplots()
            plot_confusion_matrix(conf_matrix, ax)
            st.pyplot(fig)
        with col1:
            st.markdown("##### Confusion Matrix")
            st.info(
                "The confusion matrix shows how many predictions were correct and where the errors occurred. Each row represents the actual class, and each column represents the predicted class. The diagonal elements show the number of correct predictions, while the off-diagonal elements show where the model made mistakes.")

    elif st.session_state.model_type == 'Regression Model':
        scaler = StandardScaler()
        train_data[st.session_state.input_columns] = scaler.fit_transform(train_data[st.session_state.input_columns])
        test_data[st.session_state.input_columns] = scaler.transform(test_data[st.session_state.input_columns])
        model = regression_model(
            train_data,
            st.session_state.input_columns,
            st.session_state.output_column,
            st.session_state.regression_type,
            **st.session_state.config,
        )
        st.session_state.model = model

        # Calculate and display metrics
        predictions = model.predict(test_data[st.session_state.input_columns])
        mse = mean_squared_error(test_data[st.session_state.output_column], predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data[st.session_state.output_column], predictions)
        mape = mean_absolute_percentage_error(test_data[st.session_state.output_column], predictions) * 100
        r2 = r2_score(test_data[st.session_state.output_column], predictions)
        explained_variance = explained_variance_score(test_data[st.session_state.output_column], predictions)

        st.success("Model trained successfully!")
        evaluation = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'explained_variance': explained_variance
        }
        col1, col2 = st.columns(2)
        with col1:
            model_data = pickle.dumps(st.session_state.model)
            st.download_button("Download trained model", model_data, "trained_model.pkl", "application/octet-stream")
        with col2:
            col1, col2 = st.columns(2)
            with col1:
                model_name = st.text_input("Name your model to save it to your profile")
            with col2:
                if st.button("Save"):
                    if model_name:
                        save_model_to_profile(model, evaluation, model_name, test_data=test_data,
                                              regression_type=st.session_state.regression_type,
                                              input=st.session_state.input_columns,
                                              output=st.session_state.output_column)
                        st.success("Model saved to profile")
                    else:
                        st.error("You must name the model for saving")

        st.markdown("---")
        st.subheader("Evaluation:")
        # Create a DataFrame for the metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)',
                       'MAE percentage (MAPE)', 'R² Score', 'Explained Variance'],
            'Value': [mse, rmse, mae, mape, r2, explained_variance]
        })

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            # Display metrics as a table
            st.write("##### Model Performance Metrics for Advanced Users")
            st.dataframe(metrics_df)
        with col2:
            st.markdown("##### Beginner-Friendly Explanations")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.info(
                f"**Mean Absolute Error (MAE):** On average, the model's predictions are off by {mae:.2f} units,  {mape:.2f}%.")
            st.info(
                f"**R² Score:** {r2:.2f}, which means the model explains {r2 * 100:.2f}% of the variability in the data.")

        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            # Bar chart of actual vs predicted values with color difference
            sample_size = 100  # Define a sample size
            if len(test_data) > sample_size:
                sampled_data = test_data.sample(sample_size, random_state=42)
                sampled_predictions = model.predict(sampled_data[st.session_state.input_columns])
            else:
                sampled_data = test_data
                sampled_predictions = predictions

            actual_values = sampled_data[st.session_state.output_column].values
            predicted_values = sampled_predictions

            # Create a DataFrame for Altair
            chart_data = pd.DataFrame({
                'Index': np.arange(len(sampled_data)),
                'Actual': actual_values,
                'Predicted': predicted_values
            })

            base = alt.Chart(chart_data).encode(
                x=alt.X('Index:O', title='Index')
            )

            actual_bars = base.mark_bar(color='blue', opacity=0.7).encode(
                y=alt.Y('Actual:Q', title=f'Value of {st.session_state.output_column}')
            )

            predicted_bars = base.mark_bar(color='orange', opacity=0.7).encode(
                y=alt.Y('Predicted:Q')
            )

            difference_bars = base.mark_bar().encode(
                y=alt.Y('Actual:Q', title=f'Value of {st.session_state.output_column}'),
                y2='Predicted:Q',
                color=alt.condition(
                    alt.datum.Actual > alt.datum.Predicted,
                    alt.value('red'),  # The positive color
                    alt.value('green')  # The negative color
                )
            )

            chart = actual_bars + predicted_bars + difference_bars
            st.altair_chart(chart, use_container_width=True)

        with col2:
            st.write(" ")
            st.write(" ")
            st.error("The prediction overestimated the real value")
            st.success("The prediction underestimated the real value")

    elif st.session_state.model_type == 'Clustering Model':
        model = clustering_model(
            train_data,
            st.session_state.input_columns,
            st.session_state.config,
        )
        st.session_state.model = model

        # Calculate and display metrics
        predictions = model.predict(test_data[st.session_state.input_columns])
        silhouette_avg = silhouette_score(test_data[st.session_state.input_columns], predictions)
        davies_bouldin = davies_bouldin_score(test_data[st.session_state.input_columns], predictions)
        inertia = model.inertia_  # Available for KMeans and similar algorithms

        st.success("Model trained successfully!")
        evaluation = {
            'silhouette_avg': silhouette_avg,
            'davies_bouldin': davies_bouldin,
            'inertia': inertia
        }
        col1, col2 = st.columns(2)
        with col1:
            model_data = pickle.dumps(st.session_state.model)
            st.download_button("Download trained model", model_data, "trained_model.pkl", "application/octet-stream")
        with col2:
            col1, col2 = st.columns(2)
            with col1:
                model_name = st.text_input("Name your model to save it to your profile")
            with col2:
                if st.button("Save"):
                    if model_name:
                        save_model_to_profile(model, evaluation, model_name, test_data=test_data,
                                              input=st.session_state.input_columns)
                        st.success("Model saved to profile")
                    else:
                        st.error("You must name the model for saving")

        st.markdown("---")
        st.subheader("Evaluation:")
        metrics_df = pd.DataFrame({
            'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Inertia'],
            'Value': [silhouette_avg, davies_bouldin, inertia]
        })

        col1, col2 = st.columns([0.4, 0.6], gap="medium")
        with col1:
            st.write("##### Model Performance Metrics for Advanced Users")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.dataframe(metrics_df, use_container_width=True)
        with col2:
            st.markdown("##### Beginner-Friendly Explanations")
            st.info(
                f"**Silhouette Score:** Measures how similar each point is to its own cluster compared to other clusters. Higher values are better.")
            st.info(
                f"**Davies-Bouldin Index:** Measures the average similarity ratio of each cluster with its most similar cluster. Lower values indicate better clustering.")
            st.info(
                f"**Inertia:** Measures the sum of squared distances of samples to their closest cluster center. Lower values are better. ")

        col1, col2 = st.columns([0.2, 0.8])
        with col2:
            # Scatter plot of clusters
            scatter_df = test_data.copy()
            scatter_df['Cluster'] = predictions
            chart = alt.Chart(scatter_df).mark_circle(size=60).encode(
                x=alt.X(st.session_state.input_columns[0], title=st.session_state.input_columns[0]),
                y=alt.Y(st.session_state.input_columns[1], title=st.session_state.input_columns[1]),
                color='Cluster:N',
                tooltip=[st.session_state.input_columns[0], st.session_state.input_columns[1], 'Cluster']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        with col1:
            st.info(
                "The scatter plot shows how the data points are grouped into clusters. Only taking into consideration 2 features, so not to take as the only metric.")

    st.markdown("---", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.11, 0.78, 0.11])
    if col1.button("Previous", use_container_width=True):
        st.session_state.current_step = 3
        st.rerun()
    if col3.button("Next", type="primary", use_container_width=True):
        st.session_state.current_step = 5
        st.rerun()
def test_model():
    st.subheader("Step 4: Test out the model's predictions")
    st.markdown("---", unsafe_allow_html=True)
    if st.session_state.model_type == 'Classification Model' or st.session_state.model_type == 'Regression Model':
        test_out(st.session_state.input_columns, st.session_state.output_column)
    elif st.session_state.model_type == 'Clustering Model':
        test_out(st.session_state.input_columns)

    st.markdown("---", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.11, 0.78, 0.11])
    if col1.button("Previous", use_container_width=True):
        st.session_state.current_step = 4
        st.rerun()
    if col3.button("Exit", type="primary", use_container_width=True):
        st.session_state.current_page = "home"
        st.rerun()


def main():
    st.set_page_config(page_title="Model Trainer")

    # Load CSS for styling
    load_css("css/StreamlitStyle.css")

    # Initialize session state variables
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'show_explanations' not in st.session_state:
        st.session_state.show_explanations = False
    if 'show_data_help' not in st.session_state:
        st.session_state.show_data_help = False

    # Navigation buttons
    container = st.container(border=False)
    with container:
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1.5, 1, 1, 1, 1, 1, 1])
        with col1:
            st.image("images/logoTrainerAI.png", width=185)
        with col3:
            st.write("")
            if st.button("Home", key="nav_home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
        with col4:
            st.write("")
            if 'username' in st.session_state:
                if st.button("My Models", key="nav_profile", use_container_width=True):
                    st.session_state.current_page = "myModels"
                    st.rerun()
            elif st.session_state.current_page == "home":
                if st.button("Login", use_container_width=True):
                    st.session_state.current_page = "login"
                    st.rerun()
        with col5:
            st.write("")
            if 'username' in st.session_state:
                if st.button("Logout", key="nav_logout", use_container_width=True):
                    st.session_state.clear()
                    st.session_state.current_page = "home"
                    st.rerun()
            elif st.session_state.current_page == "home":
                if st.button("Sign-Up", use_container_width=True):
                    st.session_state.current_page = "signup"
                    st.rerun()
        with col7:
            if 'username' in st.session_state:
                col1, col2= st.columns(2)
                with col1:
                    st.image("images/Profile.png")
                with col2:
                    st.write(f"#### {st.session_state.username}")
        st.write("---")

    # Display the corresponding page
    if st.session_state.current_page == "home":
        home()
    elif st.session_state.current_page == "login":
        login()
    elif st.session_state.current_page == "signup":
        signup()
    elif st.session_state.current_page == "myModels":
        myModels()
    elif st.session_state.current_page == "trainer":
        trainer()

if __name__ == "__main__":
    main()
