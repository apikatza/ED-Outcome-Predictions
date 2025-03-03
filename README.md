# ED Predictions

This repository contains a machine learning project for predicting Eating Disorder (ED) risk and recovery levels using various regression models, including Support Vector Regression (SVR) and Naïve Bayes models. The project is implemented in Python and utilizes Streamlit for an interactive web-based interface.

## Features
- **Data Preprocessing**: Cleaning and transforming raw data for modeling.
- **Synthetic Data Generation**: Using SMOTE to generate synthetic samples.
- **Model Training**: Training multiple models, including SVR and Naïve Bayes.
- **Prediction Application**: A Streamlit-based app for making predictions based on uploaded CSV files.

## Folder Structure
```
ED_PREDICTIONS/
│── models/                 # Saved trained models (.pkl)
│   ├── svr_model.pkl       # Support Vector Regression model
│   ├── nb_resi_model.pkl   # Naïve Bayes model for resilience
│── Notebooks/              # Jupyter notebooks for data analysis and training
│   ├── Predict ED Risk.ipynb
│   ├── Predict Recovery Level.ipynb
│── sample data/            # Example datasets
│   ├── synthetic_data.csv  # Generated synthetic data
│── app.py                  # Streamlit app for predictions
│── README.md               # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ED_PREDICTIONS.git
   cd ED_PREDICTIONS
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload a CSV file containing the necessary input features.
2. The app will preprocess the data and apply the trained models.
3. The predicted ED risk and recovery levels will be displayed.

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- joblib
- streamlit

## License
This project is licensed under the MIT License. Feel free to use and modify it.

## Author
[Your Name] - [Your Contact Info]

