import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the previously saved models
svr = joblib.load("models/svr_model.pkl")
nb_resi = joblib.load("models/nb_resi_model.pkl")

st.title("Prediction of ED Risk and Recovery Level")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with the data", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    st.write("File preview:")
    st.dataframe(df)

    # Check if there is data
    if not df.empty:
        # Make the prediction with the SVR model
        try:
            y = df.filter(like='EAT')
            for column in y.columns:
                y[column] = pd.to_numeric(y[column], errors='coerce')
            y = y.sum(axis=1)
            y = np.where(y > 20, 1, 0)
            df['ED at present?'] = np.where(y == 1, 'Yes', 'No')


            df_ec = pd.DataFrame()
            c = df.filter(regex='^WHOQOL').columns.tolist()
            df_ec['WHOQOL'] = df[c].sum(axis=1)
            c = df.filter(regex='^HAD').columns.tolist()
            df_ec['HAD'] = df[c].sum(axis=1)
            c = df.filter(regex='^EAT').columns.tolist()
            df_ec['EAT'] = df[c].sum(axis=1)
            c = df.filter(regex='^RESI[^_]').columns.tolist()
            df_ec['RESI'] = df[c].sum(axis=1)
            c = df.filter(regex='^SEIGOODDOING').columns.tolist()
            df_ec['SEIGGOODDOING'] = df[c].sum(axis=1)
            c = df.filter(regex='^RESI_').columns.tolist()
            df_ec['RESI_ULTIM'] = df[c].sum(axis=1)

            predictions_svr = svr.predict(df_ec)
            df["Recovery Level Prediction"] = predictions_svr  # Add column with SVR predictions
            
            # Preprocessing for the NB classification model (leave this empty and fill it yourself)
            df_nb = df.filter(regex='^RESI[^_]')

            predictions_nb = nb_resi.predict(df_nb)
            df["NB Classification Prediction"] = predictions_nb  # Add column with NB predictions
            
            # Show the prediction results
            df_result = pd.DataFrame()
            df_result["ED at present?"] = df["ED at present?"]
            df_result["Risk of ED in 1 year?"] = predictions_nb
            df_result["Risk of ED in 1 year?"] = df_result["Risk of ED in 1 year?"].apply(lambda x: "No risk" if x == 0 else "Risk")
            df_result["Recovery Level in 1 year"] = predictions_svr
            df_result["Recovery Level in 1 year"] = (df_result["Recovery Level in 1 year"] * 100).round(-1)
            
            st.write("Prediction results:")
            st.dataframe(df_result)

            # Allow downloading the file with predictions
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")
