import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

from joblib import dump
from joblib import load



# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource # Use this to avoid downloading the file every time the page refreshes
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
FEATURE_DEFAULTS = {
    'EMA_15': 12.659130, 'MOM_15': 0.479972, 'STD_15': 0.470133,
    'RSI_14': 56.814686, 'MACD': 0.199328,
    'corr_PYPL': 0.001308, 'corr_PYPL_lag1': 0.001317,
    'corr_MRNA': 0.0, 'corr_MRNA_lag1': 0.0,
    'corr_AMZN': 0.002250, 'corr_AMZN_lag1': 0.002271,
    'corr_BKNG': 0.001022, 'corr_BKNG_lag1': 0.001031,
    'corr_FTV': 0.001192, 'corr_FTV_lag1': 0.001211,
    'pair_EXPD': 0.000656, 'pair_EXPD_lag1': 0.000649,
    'pair_TPL': 0.002117, 'pair_TPL_lag1': 0.002194,
    'pair_PGR': 0.001175, 'pair_PGR_lag1': 0.001201,
    'pair_CPRT': 0.001378, 'pair_CPRT_lag1': 0.001390,
    'pair_PTC': 0.001148, 'pair_PTC_lag1': 0.001167,
    'sentiment_LSTM': 0.746875, 'sentiment_lex': 0.323968,
    'sentiment_LSTM_lag1': 0.728542, 'sentiment_lex_lag1': 0.273911
}

ALL_FEATURES = [
    'EMA_15', 'MOM_15', 'STD_15', 'RSI_14', 'MACD',
    'corr_PYPL', 'corr_PYPL_lag1', 'corr_MRNA', 'corr_MRNA_lag1',
    'corr_AMZN', 'corr_AMZN_lag1', 'corr_BKNG', 'corr_BKNG_lag1',
    'corr_FTV', 'corr_FTV_lag1', 'pair_EXPD', 'pair_EXPD_lag1',
    'pair_TPL', 'pair_TPL_lag1', 'pair_PGR', 'pair_PGR_lag1',
    'pair_CPRT', 'pair_CPRT_lag1', 'pair_PTC', 'pair_PTC_lag1',
    'sentiment_LSTM', 'sentiment_lex', 'sentiment_LSTM_lag1', 'sentiment_lex_lag1'
]

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": 'explainer_sentiment.shap',
    "pipeline": 'finalized_sentiment_model.tar.gz',
    "keys": ['sentiment_LSTM', 'sentiment_lex', 'sentiment_LSTM_lag1', 'sentiment_lex_lag1'],
    "inputs": [
        {"name": "sentiment_LSTM",      "min": -1.0, "max": 1.0, "default": 0.75, "step": 0.01},
        {"name": "sentiment_lex",       "min": -1.0, "max": 1.0, "default": 0.32, "step": 0.01},
        {"name": "sentiment_LSTM_lag1", "min": -1.0, "max": 1.0, "default": 0.73, "step": 0.01},
        {"name": "sentiment_lex_lag1",  "min": -1.0, "max": 1.0, "default": 0.27, "step": 0.01},
    ]
}

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename=MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename, 
        Bucket=bucket, 
        Key= f"{key}/{os.path.basename(filename)}")
        # Extract the .joblib file from the .tar.gz
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    # Load the full pipeline
    return joblib.load(f"{joblib_file}")

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    local_path = local_path

    # Only download if it doesn't exist locally to save time
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
        
    with open(local_path, "rb") as f:
        return load(f)
        #return shap.Explainer.load(f)

# Prediction Logic
def call_model_api(input_df):

    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer() 
    )

    try:
        # Build full 29-feature row using defaults, override with user inputs
        full_row = FEATURE_DEFAULTS.copy()
        for col in input_df.columns:
            full_row[col] = input_df[col].values[0]
        full_df = pd.DataFrame([full_row])[ALL_FEATURES]
        raw_pred = predictor.predict(full_df.values)
        pred_val = float(np.array(raw_pred).flatten()[0])
        return round(pred_val, 6), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# Local Explainability
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(session, aws_bucket, posixpath.join('explainer', explainer_name), os.path.join(tempfile.gettempdir(), explainer_name))
    
    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    
    # Fix: align input columns to match training feature names
    expected_features = best_pipeline.named_steps['imputer'].feature_names_in_
    input_df = input_df.reindex(columns=expected_features)
    
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-1])
    input_df_transformed = preprocessing_pipeline.transform(input_df)
    feature_names = best_pipeline[:-1].get_feature_names_out()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values = explainer(input_df_transformed)
    feature_names = best_pipeline[:-2].get_feature_names_out()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values = explainer(input_df_transformed)
    
    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    #shap.plots.waterfall(shap_values[0], max_display=10)
    shap.plots.waterfall(shap_values[0])
    st.pyplot(fig)
    # top feature 
    # top_feature = pd.Series(shap_values[0].values, index=shap_values[0].feature_names).abs().idxmax()
    top_feature = pd.Series(shap_values[0].values, index=shap_values[0].feature_names).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader(f"Inputs")
    cols = st.columns(2)
    user_inputs = {}
    
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], max_value=inp['max'], value=inp['default'], step=inp['step']
            )
    
    submitted = st.form_submit_button("Run Prediction")

if submitted:

    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]
    # Prepare data
    # base_df = df_features
    # input_df = pd.concat([base_df, pd.DataFrame([data_row], columns=base_df.columns)])
    input_df = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])
    
    res, status = call_model_api(input_df)
    if status == 200:
        st.metric("Predicted Next-Day Return", f"{res:.4%}")
        display_explanation(input_df,session, aws_bucket)
    else:
        st.error(res)



