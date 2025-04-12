import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from io import StringIO
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from model_explainer import ModelExplainer
from utils import load_data, download_data_if_needed

# Set page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .model-container {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main title
    st.markdown("<h1 class='main-header'>Titanic Survival Prediction</h1>", unsafe_allow_html=True)
    
    # Download data if needed
    with st.spinner("Checking and downloading dataset if needed..."):
        download_data_if_needed()
    
    # Load data
    train_data, test_data = load_data()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Exploration", 
        "🔍 Data Preprocessing", 
        "🤖 Model Training & Comparison", 
        "🔮 Make Predictions",
        "📚 Model Explanations"
    ])
    
    with tab1:
        data_exploration(train_data)
    
    with tab2:
        data_processor = DataProcessor(train_data, test_data)
        X_train, X_test, y_train, y_test, feature_names = data_preprocessing(data_processor)
    
    # Create session states for storing processed data, data processor, model trainer and evaluator
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {
            'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None, 'feature_names': None
        }
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = None
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = None
    if 'model_evaluator' not in st.session_state:
        st.session_state.model_evaluator = None
    
    # Store processed data in session state if preprocessing completed
    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        st.session_state.processed_data = {
            'X_train': X_train, 'X_test': X_test, 
            'y_train': y_train, 'y_test': y_test, 
            'feature_names': feature_names
        }
        # Store data processor in session state for predictions
        st.session_state.data_processor = data_processor
            
    with tab3:
        if st.session_state.processed_data['X_train'] is not None:
            if st.session_state.model_trainer is None:
                # Create model trainer with data from session state
                st.session_state.model_trainer = ModelTrainer(
                    st.session_state.processed_data['X_train'],
                    st.session_state.processed_data['y_train'],
                    st.session_state.processed_data['feature_names']
                )
                st.session_state.model_evaluator = ModelEvaluator(
                    st.session_state.model_trainer.models, 
                    st.session_state.processed_data['X_train'],
                    st.session_state.processed_data['X_test'],
                    st.session_state.processed_data['y_train'],
                    st.session_state.processed_data['y_test']
                )
            model_training_comparison(st.session_state.model_trainer, st.session_state.model_evaluator)
        else:
            st.error("Please complete data preprocessing first")
    
    with tab4:
        if st.session_state.model_trainer is not None and st.session_state.data_processor is not None:
            make_predictions(st.session_state.model_trainer, st.session_state.data_processor)
        elif st.session_state.model_trainer is None:
            st.error("Please train models first")
        else:
            st.error("Please complete data preprocessing first")
    
    with tab5:
        model_explanation()

def data_exploration(data):
    st.markdown("<h2 class='subheader'>Data Exploration</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        The Titanic dataset contains information about passengers aboard the RMS Titanic, which tragically 
        sank on its maiden voyage in April 1912. The dataset includes various attributes of passengers, 
        such as age, gender, class, fare paid, and most importantly, survival status.
    </div>
    """, unsafe_allow_html=True)
    
    # Show basic dataset info
    st.write("#### Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Passengers", len(data))
        st.metric("Survived", f"{data['Survived'].sum()} ({data['Survived'].mean()*100:.1f}%)")
    with col2:
        st.metric("Number of Features", len(data.columns))
        st.metric("Missing Values", data.isna().sum().sum())
    
    # Display raw data sample
    with st.expander("View Raw Data Sample"):
        st.dataframe(data.head(10))
    
    # Display data information
    with st.expander("Data Information"):
        # Use StringIO with context manager for better resource handling in Python 3.13
        from io import StringIO
        with StringIO() as buffer:
            data.info(buf=buffer)
            info_text = buffer.getvalue()
        st.text(info_text)
        st.write("#### Statistical Summary")
        st.dataframe(data.describe())
    
    # Visualizations
    st.write("#### Data Visualizations")
    
    # Survival by gender
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            data.groupby('Sex')['Survived'].mean().reset_index(), 
            x='Sex', y='Survived', 
            title='Survival Rate by Gender',
            color='Sex',
            labels={'Survived': 'Survival Rate'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Survival by class
    with col2:
        fig = px.bar(
            data.groupby('Pclass')['Survived'].mean().reset_index(), 
            x='Pclass', y='Survived', 
            title='Survival Rate by Passenger Class',
            color='Pclass',
            labels={'Survived': 'Survival Rate', 'Pclass': 'Passenger Class'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Age distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            data, x='Age', color='Survived', 
            marginal='box', 
            title='Age Distribution by Survival',
            labels={'Survived': 'Survived (1=Yes, 0=No)'},
            opacity=0.7,
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Fare distribution
    with col2:
        fig = px.histogram(
            data, x='Fare', color='Survived', 
            marginal='box', 
            title='Fare Distribution by Survival',
            labels={'Survived': 'Survived (1=Yes, 0=No)'},
            opacity=0.7,
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.write("#### Feature Correlations")
    numeric_data = data.select_dtypes(include=[np.number])
    fig = px.imshow(
        numeric_data.corr(),
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title='Correlation Heatmap'
    )
    st.plotly_chart(fig, use_container_width=True)

def data_preprocessing(data_processor):
    st.markdown("<h2 class='subheader'>Data Preprocessing</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        Data preprocessing is crucial for machine learning models. It involves handling missing values, 
        encoding categorical features, scaling numerical features, and engineering new features to 
        improve model performance.
    </div>
    """, unsafe_allow_html=True)
    
    # Display preprocessing options
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Missing Values Strategy")
        age_strategy = st.selectbox(
            "Age Missing Values", 
            ["median", "mean", "mode"], 
            index=0,
            help="Strategy to fill missing Age values"
        )
        
        embarked_strategy = st.selectbox(
            "Embarked Missing Values", 
            ["most_frequent", "drop"], 
            index=0,
            help="Strategy to handle missing Embarked values"
        )
    
    with col2:
        st.write("#### Feature Engineering")
        create_family_size = st.checkbox("Create Family Size Feature", value=True,
                                        help="Create a new feature representing family size")
        create_title_feature = st.checkbox("Extract Title from Name", value=True,
                                          help="Extract passenger titles from Name")
        create_age_groups = st.checkbox("Create Age Categories", value=True,
                                       help="Convert age to categorical groups")
    
    # Process data with selected options
    if st.button("Apply Preprocessing"):
        with st.spinner("Preprocessing data..."):
            progress_bar = st.progress(0)
            
            # Apply selected options to data processor
            data_processor.set_options(
                age_strategy=age_strategy,
                embarked_strategy=embarked_strategy,
                create_family_size=create_family_size,
                create_title_feature=create_title_feature,
                create_age_groups=create_age_groups
            )
            
            # Process data in steps to update progress bar
            progress_bar.progress(10)
            data_processor.handle_missing_values()
            progress_bar.progress(30)
            data_processor.engineer_features()
            progress_bar.progress(50)
            data_processor.encode_categorical_features()
            progress_bar.progress(70)
            X_train, X_test, y_train, y_test = data_processor.split_data()
            progress_bar.progress(90)
            data_processor.scale_features(X_train, X_test)
            progress_bar.progress(100)
            
            time.sleep(0.5)  # Give time for user to see 100%
            progress_bar.empty()
            
            st.success("Data preprocessing completed!")
            
            # Display preprocessed data samples
            st.write("#### Processed Features Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Training Features Sample")
                st.dataframe(pd.DataFrame(X_train[:5], columns=data_processor.feature_names))
            with col2:
                st.write("Target Variable (Survival)")
                survival_counts = pd.Series(y_train).value_counts()
                fig = px.pie(
                    values=survival_counts.values,
                    names=["Not Survived", "Survived"] if 0 in survival_counts.index else ["Survived", "Not Survived"],
                    title="Target Distribution in Training Set"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Return processed data
            return data_processor.X_train, data_processor.X_test, data_processor.y_train, data_processor.y_test, data_processor.feature_names
    
    return None, None, None, None, None

def model_training_comparison(model_trainer, model_evaluator):
    st.markdown("<h2 class='subheader'>Model Training & Comparison</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        In this section, we train and compare six different classification models.
        Each model has different strengths and approaches to classification.
    </div>
    """, unsafe_allow_html=True)
    
    # Train models button
    if st.button("Train All Models"):
        with st.spinner("Training models..."):
            progress_bar = st.progress(0)
            models = list(model_trainer.models.keys())
            
            for i, model_name in enumerate(models):
                progress_text = f"Training {model_name}..."
                st.text(progress_text)
                model_trainer.train_model(model_name)
                progress_bar.progress((i + 1) / len(models))
            
            progress_bar.empty()
            st.success("All models trained successfully!")
    
    # Model comparison
    st.write("#### Model Performance Comparison")
    
    # Check if models are trained
    if not all(model_trainer.trained_models.values()):
        st.warning("Please train the models first")
        return
    
    # Evaluate models
    metrics_df = model_evaluator.compare_models()
    
    # Display metrics as table
    st.dataframe(metrics_df)
    
    # Plot metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            metrics_df, x=metrics_df.index, y="Accuracy",
            title="Accuracy Comparison",
            color=metrics_df.index,
            labels={"x": "Model", "y": "Accuracy"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            metrics_df, x=metrics_df.index, y="F1 Score",
            title="F1 Score Comparison",
            color=metrics_df.index,
            labels={"x": "Model", "y": "F1 Score"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display ROC curves
    st.write("#### ROC Curves")
    fig = model_evaluator.plot_roc_curves()
    st.plotly_chart(fig, use_container_width=True)
    
    # Display confusion matrices
    st.write("#### Confusion Matrices")
    model_names = list(model_trainer.models.keys())
    selected_model = st.selectbox("Select model to view confusion matrix", model_names)
    
    fig = model_evaluator.plot_confusion_matrix(selected_model)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance for applicable models
    st.write("#### Feature Importance")
    models_with_importance = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]
    selected_model = st.selectbox("Select model to view feature importance", models_with_importance)
    
    try:
        fig = model_evaluator.plot_feature_importance(selected_model, model_trainer.feature_names)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Cannot display feature importance for {selected_model}: {str(e)}")

def make_predictions(model_trainer, data_processor):
    st.markdown("<h2 class='subheader'>Make Predictions</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        Enter passenger details to predict their survival probability using all trained models.
        You can compare how different models classify the same passenger.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models are trained
    if not all(model_trainer.trained_models.values()):
        st.warning("Please train the models first")
        return
    
    # Input form for passenger details
    st.write("#### Enter Passenger Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"], index=0)
        age = st.slider("Age", 0.5, 80.0, 30.0, 0.5)
        
    with col2:
        sibsp = st.slider("Number of Siblings/Spouses", 0, 8, 0)
        parch = st.slider("Number of Parents/Children", 0, 9, 0)
        fare = st.slider("Fare (£)", 0.0, 512.0, 32.0, 0.1)
        
    with col3:
        embarked = st.selectbox("Port of Embarkation", ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"], index=2)
        embarked = embarked[0]  # Get first letter (C, Q, or S)
        cabin_available = st.checkbox("Cabin Information Available", value=False)
        cabin = "C85" if cabin_available else None
    
    # Create passenger data
    passenger = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked,
        'Cabin': cabin,
        'Name': 'Unknown Passenger'  # Placeholder name
    }
    
    # Make prediction button
    if st.button("Predict Survival"):
        with st.spinner("Predicting..."):
            # Process passenger data
            processed_passenger = data_processor.preprocess_new_passenger(passenger)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for model_name, model in model_trainer.trained_models.items():
                if model is None:
                    continue
                    
                # Make prediction
                pred = model.predict(processed_passenger)[0]
                
                # Get probability for positive class (survival)
                try:
                    prob = model.predict_proba(processed_passenger)[0][1]
                except:
                    prob = float(pred)  # For models without predict_proba
                
                predictions[model_name] = "Survived" if pred == 1 else "Did Not Survive"
                probabilities[model_name] = prob
            
            # Display results
            st.write("#### Prediction Results")
            
            # Display predictions in cards
            cols = st.columns(3)
            for i, (model_name, prediction) in enumerate(predictions.items()):
                col_idx = i % 3
                with cols[col_idx]:
                    survival_prob = probabilities[model_name] * 100
                    
                    card_color = "#d4edda" if prediction == "Survived" else "#f8d7da"
                    text_color = "#155724" if prediction == "Survived" else "#721c24"
                    
                    st.markdown(f"""
                    <div style="background-color: {card_color}; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                        <h4 style="color: {text_color}; margin-bottom: 5px;">{model_name}</h4>
                        <h3 style="color: {text_color}; margin-top: 0;">{prediction}</h3>
                        <p style="color: {text_color};">Survival probability: {survival_prob:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display aggregated result
            avg_prob = sum(probabilities.values()) / len(probabilities)
            
            st.markdown("#### Consensus Prediction")
            consensus = "Survived" if avg_prob > 0.5 else "Did Not Survive"
            consensus_color = "#d4edda" if consensus == "Survived" else "#f8d7da"
            consensus_text = "#155724" if consensus == "Survived" else "#721c24"
            
            st.markdown(f"""
            <div style="background-color: {consensus_color}; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                <h2 style="color: {consensus_text}; margin-bottom: 5px;">Consensus: {consensus}</h2>
                <p style="color: {consensus_text}; font-size: 1.2rem;">Average Survival Probability: {avg_prob*100:.1f}%</p>
                <p style="color: {consensus_text};">Based on the average of all model predictions</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display comparison chart
            fig = go.Figure()
            for model_name, prob in probabilities.items():
                fig.add_trace(go.Bar(
                    x=[model_name],
                    y=[prob],
                    name=model_name
                ))
            
            fig.update_layout(
                title="Survival Probability by Model",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1]),
                hovermode="closest",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

def model_explanation():
    st.markdown("<h2 class='subheader'>Model Explanations</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        Understanding how different machine learning models work is essential for selecting the most 
        appropriate model for a task. Each model has its own strengths, weaknesses, and assumptions.
    </div>
    """, unsafe_allow_html=True)
    
    # Get model explanations
    model_explainer = ModelExplainer()
    explanations = model_explainer.get_explanations()
    
    # Display explanations
    for model_name, explanation in explanations.items():
        with st.expander(f"{model_name} - How it works"):
            st.markdown(explanation["description"], unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Pros")
                for pro in explanation["pros"]:
                    st.markdown(f"✅ {pro}")
            
            with col2:
                st.subheader("Cons")
                for con in explanation["cons"]:
                    st.markdown(f"⚠️ {con}")

if __name__ == "__main__":
    main()
