# 🏠 House Price Prediction
### QSkill Internship | Python Development | Slab 1

A professional Machine Learning web application that predicts house prices using multiple ML algorithms, built with Python, Scikit-learn, XGBoost and Streamlit.

---

## 📁 Project Structure
```
house_price_prediction/
├── app.py                  ← Streamlit web dashboard
├── model.py                ← ML model training & evaluation
├── requirements.txt        ← Dependencies
├── README.md               ← Project documentation
├── data/
│   └── house_data.csv      ← Kaggle KC House dataset
├── models/
│   ├── house_price_model.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
└── plots/
    ├── eda_dashboard.png
    └── model_performance.png
```

---

## ⚙️ Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Real Kaggle Dataset
- Go to: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
- Download `kc_house_data.csv`
- Rename to `house_data.csv`
- Place it at: `data/house_data.csv`

### 3. Train the Model
```bash
python model.py
```

### 4. Launch the Web App
```bash
streamlit run app.py
```

---

## 🚀 Features
- 📊 Full Exploratory Data Analysis with 6 visualizations
- 🤖 4 ML algorithms compared automatically
- 🏆 Best model selected automatically
- 🌐 Interactive Streamlit web dashboard
- 🔮 Real-time house price prediction with input sliders
- 📈 Model evaluation: MAE, RMSE, R² Score
- 📉 Feature importance visualization
- 🔧 24 engineered features

---

## 🧠 Technologies Used
| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Matplotlib & Seaborn | Visualizations |
| Scikit-learn | ML models |
| XGBoost | Boosting algorithm |
| Streamlit | Web dashboard |
| Joblib | Model persistence |

---

## 📊 Model Performance
| Model | R² Score |
|---|---|
| Linear Regression | 68.95% |
| Random Forest | 73.28% |
| Gradient Boosting | 74.64% ✅ Best |
| XGBoost | 74.41% |

- **Dataset:** Real Kaggle KC House Data (21,000+ houses)
- **Features:** 24 engineered features
- **Train/Test Split:** 80% / 20%

---

*QSkill Internship | Python Development Domain | April–May 2026*
