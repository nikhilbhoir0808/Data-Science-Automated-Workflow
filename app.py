import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import datetime
import io

# Page configuration
st.set_page_config(page_title="End-to-End DS App", layout="wide")
st.title("🚀 Enhanced End-to-End Data Science Workflow")

# Initialize session state
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

# Sidebar navigation
pages = [
    "1. Data Upload",
    "2. Data Cleaning",
    "3. Exploratory Data Analysis",
    "4. Feature Engineering",
    "5. Model Training",
    "6. Model Evaluation",
    "7. Model Optimization",
    "8. Live Predictions",
    "9. Monitoring Logs"
]
page = st.sidebar.selectbox("Select Workflow Step", pages)

# File paths
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
LOG_FILE = "predictions.log"

# Helper: Download button for cleaned data
def get_csv_download_link(df, filename="cleaned_data.csv"):
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer

# 1. Data Upload
if page == "1. Data Upload":
    st.header("Step 1: Data Collection - Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_original = df.copy()
            st.session_state.df = df.copy()
            st.success("✅ Dataset uploaded successfully!")
            st.write("Preview:")
            st.dataframe(df.head())

            st.subheader("Select Target Column (for supervised learning)")
            possible_targets = [col for col in df.columns if df[col].nunique() < 20 or df[col].dtype == 'object']
            default_idx = 0
            if possible_targets:
                default_idx = df.columns.tolist().index(possible_targets[0])
            target = st.selectbox("Choose the target/label column", df.columns, index=default_idx)
            st.session_state.target_col = target
            st.info(f"Target column set to: **{target}**")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("👆 Upload a CSV file to begin.")

# Block if no data
if st.session_state.df is None and page != "1. Data Upload":
    st.error("❌ No dataset loaded. Please go to **1. Data Upload** first.")
    st.stop()

df = st.session_state.df

# 2. Data Cleaning - With Auto Encoding Warning
if page == "2. Data Cleaning":
    st.header("Step 2: Data Cleaning - Select Multiple Options")
    st.write(f"Current dataset shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.write("Preview:")
    st.dataframe(df.head())

    csv_buffer = get_csv_download_link(df)
    st.download_button(
        label="📥 Download Current Dataset as CSV",
        data=csv_buffer,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    st.subheader("Choose Cleaning Operations")

    col1, col2 = st.columns(2)
    
    with col1:
        handle_missing = st.checkbox("Handle Missing Values", value=True)
        if handle_missing:
            missing_option = st.radio(
                "Missing values strategy",
                ["Drop rows with any missing values",
                 "Impute (median for numeric, mode for categorical)"]
            )
        
        remove_outliers = st.checkbox("Remove Outliers (IQR method on numeric columns)")
        
    with col2:
        encode_categorical = st.checkbox("Label Encode ALL Categorical Columns (including features)", value=True)
        st.warning("Important: Machine learning models require numeric features. This option will encode ALL string columns (except target).")
        
        drop_duplicates = st.checkbox("Drop Duplicate Rows")
        
        reset_index = st.checkbox("Reset Index After Cleaning")

    if st.button("Apply Selected Cleaning Operations", type="primary"):
        new_df = df.copy()
        
        if handle_missing:
            if missing_option == "Drop rows with any missing values":
                new_df = new_df.dropna()
                st.info("Dropped rows with missing values.")
            else:
                for col in new_df.columns:
                    if new_df[col].dtype == 'object':
                        mode_val = new_df[col].mode()
                        fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                    else:
                        fill_val = new_df[col].median()
                    new_df[col].fillna(fill_val, inplace=True)
                st.info("Imputed missing values.")
        
        if remove_outliers:
            numeric_cols = new_df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                Q1 = new_df[numeric_cols].quantile(0.25)
                Q3 = new_df[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                mask = ~((new_df[numeric_cols] < (Q1 - 1.5 * IQR)) | (new_df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
                new_df = new_df[mask]
                st.info("Outliers removed using IQR method.")
            else:
                st.warning("No numeric columns for outlier removal.")
        
        if encode_categorical:
            encoded_count = 0
            for col in new_df.select_dtypes(include='object').columns:
                if col != st.session_state.target_col:  # Keep target as-is for now
                    le = LabelEncoder()
                    new_df[col] = le.fit_transform(new_df[col].astype(str))
                    encoded_count += 1
            if encoded_count > 0:
                st.info(f"Encoded {encoded_count} categorical feature column(s).")
            else:
                st.info("No categorical features to encode.")
        
        if drop_duplicates:
            before = new_df.shape[0]
            new_df = new_df.drop_duplicates()
            after = new_df.shape[0]
            st.info(f"Dropped {before - after} duplicate rows.")
        
        if reset_index:
            new_df = new_df.reset_index(drop=True)
            st.info("Index reset.")
        
        st.session_state.df = new_df
        
        st.success("Cleaning operations applied successfully!")
        st.write(f"New shape: **{new_df.shape[0]} rows × {new_df.shape[1]} columns**")
        st.write("Updated Preview:")
        st.dataframe(new_df.head())
        
        csv_buffer = get_csv_download_link(new_df)
        st.download_button(
            label="📥 Download Updated Cleaned Dataset",
            data=csv_buffer,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

# 3. Exploratory Data Analysis (unchanged from last stable version)
if page == "3. Exploratory Data Analysis":
    st.header("Step 3: Exploratory Data Analysis")
    st.dataframe(df.describe())

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Correlation Heatmap (Numeric Features Only)")
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        elif numeric_df.shape[1] == 1:
            st.info("Only one numeric column – no correlations to show.")
        else:
            st.info("No numeric columns available for correlation heatmap.")

    with col2:
        if df.shape[1] > 0:
            col_select = st.selectbox("Select column for histogram", df.columns)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[col_select], kde=True, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No columns available for histogram.")

    st.write("### Pairplot")
    all_columns = df.columns.tolist()
    if len(all_columns) < 2:
        st.warning(f"Pairplot requires at least 2 columns. Current dataset has only {len(all_columns)} column(s).")
    else:
        hue = None
        if (st.session_state.target_col in df.columns and 
            df[st.session_state.target_col].nunique() > 1 and 
            df[st.session_state.target_col].nunique() < 50):
            hue = st.session_state.target_col
            plot_df = df.copy()
            st.info(f"Pairplot colored by target: **{hue}**")
        else:
            plot_df = df.copy()
            st.info("Pairplot without coloring.")

        if plot_df.shape[1] < 2:
            st.warning("Not enough columns for pairplot.")
        else:
            if st.button("Generate Pairplot"):
                if plot_df.shape[1] > 10:
                    st.warning("Many columns – may be slow.")
                with st.spinner("Generating..."):
                    try:
                        fig = sns.pairplot(plot_df, hue=hue, diag_kind='kde')
                        st.pyplot(fig)
                        plt.close('all')
                    except Exception as e:
                        st.error(f"Pairplot error: {e}")

# 4. Feature Engineering
if page == "4. Feature Engineering":
    st.header("Step 4: Feature Engineering")
    if st.button("Apply Standard Scaling to numeric features"):
        numeric_cols = df.select_dtypes(include=np.number).columns
        numeric_cols = numeric_cols.drop(st.session_state.target_col, errors='ignore')
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            st.session_state.df = df.copy()
            st.session_state.scaler = scaler
            joblib.dump(scaler, SCALER_PATH)
            st.success("Numeric features scaled!")
        else:
            st.info("No numeric features to scale.")

# 5. Model Training - FIXED STRING-TO-FLOAT ERROR
if page == "5. Model Training":
    st.header("Step 5: Model Training")
    target = st.session_state.target_col
    
    # Auto-detect and warn about remaining string columns in features
    feature_cols = [c for c in df.columns if c != target]
    string_features = df[feature_cols].select_dtypes(include='object').columns.tolist()
    
    if string_features:
        st.error(f"Found string columns in features: {string_features}")
        st.info("These must be encoded before training. Go back to **Data Cleaning** and enable 'Label Encode ALL Categorical Columns'.")
        st.stop()
    
    if len(feature_cols) == 0:
        st.error("No feature columns available!")
        st.stop()
    
    X = df[feature_cols]
    y = df[target]
    
    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )
    
    if st.button("Train Selected Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == "XGBoost":
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        
        with st.spinner(f"Training {model_choice}..."):
            model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        st.session_state.model = model
        st.session_state.model_name = model_choice
        joblib.dump(model, MODEL_PATH)
        
        st.success(f"{model_choice} trained successfully!")
        st.write(f"**Test Accuracy:** {acc:.2f}")

# Remaining pages (6-9) unchanged...
# (Same as previous version for evaluation, optimization, predictions, monitoring)

if page == "6. Model Evaluation":
    st.header("Step 6: Model Evaluation")
    if st.session_state.model is None:
        st.warning("No model trained yet.")
    else:
        st.info(f"Current model: **{st.session_state.model_name}**")
        target = st.session_state.target_col
        X = df.drop(columns=[target])
        y = df[target]
        preds = st.session_state.model.predict(X)
        st.text("Classification Report:")
        st.code(classification_report(y, preds))

if page == "7. Model Optimization":
    st.header("Step 7: Model Optimization")
    if st.session_state.model is None:
        st.warning("Train a model first.")
    else:
        st.info(f"Current model: {st.session_state.model_name}")
        if st.session_state.model_name == "Logistic Regression":
            target = st.session_state.target_col
            X = df.drop(columns=[target])
            y = df[target]
            if st.button("Run Grid Search Tuning"):
                param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
                grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
                grid.fit(X, y)
                st.session_state.model = grid.best_estimator_
                joblib.dump(grid.best_estimator_, MODEL_PATH)
                st.success(f"Optimized! Best params: {grid.best_params_}")
        else:
            st.info("Advanced tuning for tree models coming soon.")

if page == "8. Live Predictions":
    st.header("Step 8: Live Predictions (Deployment)")
    if st.session_state.model is None:
        st.warning("Train a model first!")
    elif st.session_state.scaler is None:
        st.warning("Scale features first!")
    else:
        st.info(f"Using model: **{st.session_state.model_name}**")
        feature_cols = df.drop(columns=[st.session_state.target_col]).columns
        inputs = []
        for col in feature_cols:
            default_val = df[col].mean() if np.issubdtype(df[col].dtype, np.number) else 0.0
            val = st.number_input(col, value=float(default_val))
            inputs.append(val)

        if st.button("Make Prediction"):
            input_df = pd.DataFrame([inputs], columns=feature_cols)
            scaled_input = st.session_state.scaler.transform(input_df)
            prediction = st.session_state.model.predict(scaled_input)[0]
            probabilities = st.session_state.model.predict_proba(scaled_input)[0]

            st.success(f"Prediction: **{prediction}**")
            st.write("Class Probabilities:")
            for cls, prob in zip(st.session_state.model.classes_, probabilities):
                st.write(f"• Class {cls}: {prob:.3f}")

            with open(LOG_FILE, "a") as f:
                f.write(f"{datetime.datetime.now()} | Model: {st.session_state.model_name} | Input: {inputs} | Prediction: {prediction}\n")
            st.info("Prediction logged.")

if page == "9. Monitoring Logs":
    st.header("Step 9: Monitoring - Prediction Logs")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = f.readlines()
        st.code("".join(reversed(logs)))
    else:
        st.info("No predictions made yet.")

st.sidebar.success("✅ String-to-float error fixed! All categorical features must be encoded before training.")