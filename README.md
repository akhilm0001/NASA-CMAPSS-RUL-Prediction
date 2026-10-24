# ✈️ Predictive Maintenance & Remaining Useful Life (RUL) Prediction

**Machine Learning | NASA C-MAPSS | Python | Scikit-learn | Streamlit | Hugging Face**

A machine learning project that predicts the Remaining Useful Life (RUL) of turbofan engines using NASA's C-MAPSS engine degradation simulation dataset. The project covers data preprocessing, feature engineering, regression model development, performance evaluation, and an interactive Streamlit application.

🚀 **[Launch the Live RUL Prediction App](https://nasa-cmapss-rul-prediction-takc47paf5jckgwuuuovsz.streamlit.app/)**

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Technologies Used](#-technologies-used)
- [Machine Learning Models](#-machine-learning-models)
- [Model Performance](#-model-performance)
- [Visualizations](#-visualizations)
- [Live Streamlit Application](#-live-streamlit-application)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Key Findings](#-key-findings)
- [Future Improvements](#-future-improvements)
- [References](#-references)
- [Author](#-author)

---

## 🚀 Project Overview

Aircraft engines operate under demanding conditions. Unexpected failures can cause costly maintenance, operational delays, and safety concerns.

Predictive maintenance uses historical operating data and machine learning to estimate equipment degradation and support maintenance planning before failure occurs.

This project uses NASA's C-MAPSS turbofan engine degradation simulation dataset to develop regression models that estimate an engine's Remaining Useful Life based on operational settings and sensor measurements.

The project evaluates four datasets — FD001, FD002, FD003, and FD004 — using Random Forest and Gradient Boosting regression models.

The trained models are integrated into an interactive Streamlit application that supports engine data selection, manual feature input, and RUL prediction.

### 🎯 Objectives

- 📂 Analyze the four NASA C-MAPSS datasets.
- 🧹 Preprocess engine sensor readings and operational settings.
- ⚙️ Prepare Remaining Useful Life targets for supervised learning.
- 🔧 Perform feature engineering.
- 🤖 Train and compare machine learning regression models.
- 📊 Evaluate predictions using MAE and RMSE.
- ✈️ Estimate the remaining useful life of turbofan engines.
- 🖥️ Deploy an interactive prediction application using Streamlit.
- ☁️ Host larger trained model files on Hugging Face.

### 🌐 Live Application

**[Open the NASA C-MAPSS RUL Prediction Dashboard](https://nasa-cmapss-rul-prediction-takc47paf5jckgwuuuovsz.streamlit.app/)**

The application supports all four datasets and two input modes:

- **NASA Test Data:** Select an engine and predict its RUL using the latest available sensor readings.
- **Manual / New Engine Data:** Enter the feature values expected by the selected model and generate a prediction.

---

## 📊 Dataset

This project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**.

📥 **Official dataset download:**

[Download NASA C-MAPSS Dataset (ZIP)](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)

The dataset contains simulated turbofan engine run-to-failure data, including engine identifiers, operating cycles, operational settings, and sensor measurements.

### Dataset Files

| Dataset | Training file | Test file | Test RUL file |
|---|---|---|---|
| FD001 | `train_FD001.txt` | `test_FD001.txt` | `RUL_FD001.txt` |
| FD002 | `train_FD002.txt` | `test_FD002.txt` | `RUL_FD002.txt` |
| FD003 | `train_FD003.txt` | `test_FD003.txt` | `RUL_FD003.txt` |
| FD004 | `train_FD004.txt` | `test_FD004.txt` | `RUL_FD004.txt` |

### Dataset Characteristics

| Dataset | Operating conditions | Fault modes |
|---|---|---|
| FD001 | Single | Single |
| FD002 | Multiple | Single |
| FD003 | Single | Multiple |
| FD004 | Multiple | Multiple |

### Role of the RUL Files

- **Training files:** Contain engine operating cycles and sensor readings used to train the models. Training RUL targets can be derived from run-to-failure trajectories.
- **Test files:** Contain sensor readings for test engines up to their last observed cycles.
- **RUL files:** Provide reference remaining useful life values for test engines and can be used to evaluate predictions.

The four subsets enable model evaluation under different operating conditions and fault modes.

---

## 🔄 Project Workflow

```text
             ✈️ NASA C-MAPSS Dataset
                       |
                       v
                 📂 Data Loading
                       |
                       v
            🔍 Exploratory Data Analysis
                       |
                       v
                🧹 Data Preprocessing
                       |
                       v
                ⚙️ RUL Target Creation
                       |
                       v
                🔧 Feature Engineering
                       |
                       v
                 📊 Model Training
                       |
                       v
              🤖 Regression Models
                       |
                       v
               📈 Model Evaluation
                  (MAE and RMSE)
                       |
                       v
                 🏆 Model Selection
                       |
                       v
                  ✈️ RUL Prediction
                       |
                       v
                🖥️ Streamlit App
                       |
                       v
                  ☁️ Deployment
```

### Workflow Description

1. **Data loading:** Load the FD001, FD002, FD003, and FD004 datasets.
2. **Exploratory data analysis:** Examine sensor measurements, operating conditions, engine cycles, and degradation patterns.
3. **Data preprocessing:** Prepare sensor and operational data for machine learning.
4. **RUL target creation:** Generate training targets and use supplied RUL files as reference targets for test evaluation.
5. **Feature engineering:** Prepare relevant features for regression modelling.
6. **Model training:** Train Random Forest and Gradient Boosting regressors.
7. **Model evaluation:** Measure prediction errors using MAE and RMSE.
8. **Model selection:** Select a model for each dataset based on the evaluated results.
9. **Application development:** Integrate the trained models into a Streamlit interface.
10. **Cloud deployment:** Deploy the application on Streamlit Community Cloud and host larger model files on Hugging Face.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computation |
| Scikit-learn | Model training and evaluation |
| Joblib | Saving and loading trained models |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Experimentation and analysis |
| Streamlit | Interactive web application |
| Hugging Face Hub | Hosting and downloading larger model files |
| Git and GitHub | Version control and source code hosting |

---

## 🤖 Machine Learning Models

### 1. Random Forest Regressor

Random Forest combines predictions from multiple decision trees to estimate RUL. It can model nonlinear relationships between engine sensor readings and degradation.

**Selected models:** FD002 and FD004.

### 2. Gradient Boosting Regressor

Gradient Boosting builds decision trees sequentially, with each successive tree learning to reduce the errors made by the existing ensemble.

**Selected models:** FD001 and FD003.

### Evaluation Metrics

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted RUL. Lower values indicate smaller average prediction errors.

**Root Mean Squared Error (RMSE)**

Measures the square root of the average squared prediction error. Larger errors have a greater effect on this metric.

For both metrics, lower values indicate better predictive performance when the evaluation setup is comparable.

---

## 📈 Model Performance

The following table presents the reported independent test results for the selected models.

| Dataset | Selected Model | Test MAE | Test RMSE |
|---|---|---:|---:|
| FD001 | Gradient Boosting | 18.18 | 24.86 |
| FD002 | Random Forest | 20.34 | 28.14 |
| FD003 | Gradient Boosting | 32.25 | 42.50 |
| FD004 | Random Forest | 27.96 | 39.42 |

### Performance Insights

- **FD001** achieved the lowest reported MAE and RMSE.
- **FD003** recorded the highest reported prediction errors.
- Gradient Boosting was selected for FD001 and FD003.
- Random Forest was selected for FD002 and FD004.

These results represent this project's reported evaluation. Comparisons with other implementations require consistent preprocessing, RUL definitions, and evaluation procedures.

---

## 📉 Visualizations

The following visualizations are referenced from the repository's `images/` directory.

### 1. ✈️ Streamlit RUL Prediction Dashboard

![Streamlit RUL Prediction Dashboard](./images/streamlit_prediction.png)

### 2. 📊 Model Performance Comparison

![Model Performance Comparison](./images/model-performance.png)

### 3. 🔍 Feature Importance — FD003

![Feature Importance FD003](./images/feature_importance_fd003.png)

### 4. 📈 Actual vs. Predicted RUL — FD003

![Actual vs Predicted RUL FD003](./images/actual%20vs%20predicted%20fd003.png)

**Note:** Ensure that each image filename matches the file actually uploaded to GitHub. Image names and capitalization must match exactly for the previews to display correctly.

---

## 🖥️ Live Streamlit Application

The deployed application allows users to interact with the trained models without running the project locally.

### Application Features

- Select FD001, FD002, FD003, or FD004.
- Choose between NASA test data and manual engine data.
- Select an engine from the available test records.
- Review the latest available sensor readings.
- Generate an estimated RUL prediction.
- Load locally hosted models and download larger models from Hugging Face.

### Try the Application

🚀 **[Launch the Live Application](https://nasa-cmapss-rul-prediction-takc47paf5jckgwuuuovsz.streamlit.app/)**

### Model Hosting

The larger FD002 and FD004 model files are hosted separately on Hugging Face to avoid GitHub file-size limitations.

🤗 **[View the Hugging Face Model Repository](https://huggingface.co/akhilm0001/nasa-cmaps-rul-models)**

The application loads FD001 and FD003 models from the GitHub repository and retrieves FD002 and FD004 models from Hugging Face.

---

## 📁 Project Structure

The following structure reflects the main files used by the deployed application and the original analysis project. Some training and visualization files are optional for running the web application.

```text
NASA-CMAPSS-RUL-Prediction/
│
├── README.md
├── app.py
├── main.py
├── predictive_maintenance.ipynb
├── requirements.txt
├── runtime.txt
│
├── CMAPS_Data/
│   ├── test_FD001.txt
│   ├── test_FD002.txt
│   ├── test_FD003.txt
│   └── test_FD004.txt
│
├── models/
│   ├── fd001_model.pkl
│   └── fd003_model.pkl
│
└── images/
    ├── streamlit_prediction.png
    ├── model-performance.png
    ├── feature_importance_fd003.png
    └── actual vs predicted fd003.png
```

**Note:** FD002 and FD004 model files are hosted in the separate Hugging Face repository. The training notebook, additional scripts, and image files should be retained according to the actual contents of your GitHub repository.

---

## ⚙️ Installation

### Prerequisites

- Python
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/akhilm0001/NASA-CMAPSS-RUL-Prediction.git
cd NASA-CMAPSS-RUL-Prediction
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal to access the application.

**Important:** The required local model files and NASA test data must be present in the expected folders. The FD002 and FD004 models are downloaded from Hugging Face by the application.

---

## ▶️ How to Run

### Run the Analysis Notebook

If the notebook is present in your repository:

```bash
jupyter notebook predictive_maintenance.ipynb
```

### Run the Main Python Script

If `main.py` exists and is intended to be executed independently:

```bash
python main.py
```

### Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The Streamlit command starts the interactive prediction application.

---

## 🔑 Key Findings

- Tree-based regression models can estimate RUL from simulated engine sensor data.
- The selected model differed across the four evaluated datasets.
- FD001 achieved the lowest reported test errors.
- FD003 had the highest reported test errors among the four subsets.
- Random Forest and Gradient Boosting provided the selected models for the respective datasets.
- Combining trained regression models with an interactive web interface makes predictive maintenance results easier to explore.
- Cloud deployment makes the application accessible through a browser.

---

## 🔮 Future Improvements

- Evaluate XGBoost and LightGBM regression models.
- Explore LSTM and other sequence-based models for time-series sensor data.
- Perform systematic hyperparameter tuning.
- Improve feature selection and sensor trend analysis.
- Add prediction uncertainty estimates.
- Enhance the dashboard with engine-level degradation trends.
- Add actual-versus-predicted comparisons and downloadable prediction results.
- Improve automated model evaluation and deployment testing.

---

## 📚 References

1. A. Saxena and K. Goebel (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
2. [NASA C-MAPSS Dataset — Official Download](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)
3. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
4. [Streamlit Documentation](https://docs.streamlit.io/)
5. [Hugging Face Hub Documentation](https://huggingface.co/docs/huggingface_hub/index)

---

## 👨‍💻 Author

**Akhil M**

Mechanical Engineering Graduate | Machine Learning | Predictive Maintenance | Data Analytics

- **GitHub:** [@akhilm0001](https://github.com/akhilm0001)
- **Project Repository:** [NASA-CMAPSS-RUL-Prediction](https://github.com/akhilm0001/NASA-CMAPSS-RUL-Prediction)
- **Live Application:** [NASA C-MAPSS RUL Prediction](https://nasa-cmapss-rul-prediction-takc47paf5jckgwuuuovsz.streamlit.app/)
- **Model Repository:** [Hugging Face Models](https://huggingface.co/akhilm0001/nasa-cmaps-rul-models)

---

⭐ If you find this project useful, consider giving the repository a star!
