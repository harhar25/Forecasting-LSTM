# Enrollment Forecasting Web Application

A web-based application for forecasting student enrollment using machine learning models, built with Streamlit.

## Overview

This system provides enrollment forecasting capabilities through a user-friendly web interface. It utilizes SARIMA and potentially other time series models to predict future enrollment trends based on historical data.

## Features

- **Interactive Web Interface**: Built with Streamlit for easy interaction
- **Multiple Forecasting Models**: Implements SARIMA and potentially other statistical models
- **Data Visualization**: Displays historical trends and forecast results with charts
- **File Upload Support**: Accepts CSV files with enrollment data
- **Forecast Period Customization**: Allows users to select forecast duration
- **Model Evaluation**: Provides performance metrics for forecasting accuracy

## Project Structure

    Web-page_Enrollment-Forecasting/
    ├── app.py # Main Streamlit application
    ├── requirements.txt # Python dependencies
    ├── pages/ # Additional application pages
    │ └── about.py # About page
    ├── data/ # Data directory (for uploaded files)
    ├── models/ # Saved model files
    ├── utils/ # Utility functions
    │ └── helpers.py # Helper functions for data processing
    └── images/ # Image assets


## Installation

## 1. Clone the repository:
## git clone https://github.com/harhar25/Web-page_Enrollment-Forecasting.git
## cd Web-page_Enrollment-Forecasting

## pip install -r requirements.txt


**USAGE**

## streamlit run app.py
## Open your web browser and navigate to the local URL provided (typically http://localhost:8501)

## Upload your enrollment data in CSV format or use the sample data

## Configure forecast parameters:
  **Select forecast period (number of periods to predict)** 
  **Choose model parameters (if customizable)**  
## View the forecast results and performance metrics



## DATA FORMAT
  **Your CSV file should contain at least two columns:**

    ~ A date column (format: YYYY-MM-DD or similar)
    ~ An enrollment count column
    
              ## Example:
              date,enrollment
              2020-01-01,150
              2020-02-01,165
              2020-03-01,172


