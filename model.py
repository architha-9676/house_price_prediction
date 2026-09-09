# ============================================================
#   model.py - ML Model Training & Evaluation (FINAL v4)
#   House Price Prediction | QSkill Internship
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

# ============================================================
#   STEP 1: LOAD & ENGINEER FEATURES
# ============================================================

def load_data():
    kaggle_path = "data/house_data.csv"
    if os.path.exists(kaggle_path):
        print("✅ Loaded real Kaggle dataset!")
        df = pd.read_csv(kaggle_path)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df = prepare_kaggle(df)
        return df
    else:
        print("⚠️  Generating dataset...")
        return generate_dataset()


def prepare_kaggle(df):
    if 'yr_built'      in df.columns: df['age']           = 2025 - df['yr_built']
    if 'yr_renovated'  in df.columns: df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)
    if 'sqft_basement' in df.columns: df['garage']        = (df['sqft_basement'] > 0).astype(int)

    if 'sqft_living' in df.columns and 'sqft_lot' in df.columns:
        df['living_lot_ratio']    = df['sqft_living'] / (df['sqft_lot'] + 1)
    if 'sqft_living' in df.columns and 'sqft_living15' in df.columns:
        df['living_vs_neighbors'] = df['sqft_living'] / (df['sqft_living15'] + 1)
    if 'sqft_lot' in df.columns and 'sqft_lot15' in df.columns:
        df['lot_vs_neighbors']    = df['sqft_lot'] / (df['sqft_lot15'] + 1)
    if 'sqft_living' in df.columns and 'bathrooms' in df.columns:
        df['sqft_per_bath']       = df['sqft_living'] / (df['bathrooms'] + 1)
    if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
        df['bed_bath_ratio']      = df['bedrooms'] / (df['bathrooms'] + 1)
    if 'sqft_living' in df.columns and 'floors' in df.columns:
        df['sqft_per_floor']      = df['sqft_living'] / (df['floors'] + 1)
    if 'grade' in df.columns and 'sqft_living' in df.columns:
        df['grade_sqft']          = df['grade'] * df['sqft_living']
    if 'view' in df.columns and 'waterfront' in df.columns:
        df['view_waterfront']     = df['view'] * (df['waterfront'] + 1)

    df['log_price'] = np.log1p(df['price'])

    keep = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
        'floors', 'waterfront', 'view', 'condition', 'grade',
        'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15',
        'age', 'was_renovated', 'garage',
        'living_lot_ratio', 'living_vs_neighbors', 'lot_vs_neighbors',
        'sqft_per_bath', 'bed_bath_ratio', 'sqft_per_floor',
        'grade_sqft', 'view_waterfront',
        'price', 'log_price'
    ]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df = df[(df['price'] > 50000) & (df['price'] < 5000000)]
    df = df[(df['bedrooms'] > 0) & (df['bedrooms'] < 15)]
    return df


def generate_dataset(n=1000):
    np.random.seed(42)
    bedrooms    = np.random.randint(1, 7, n)
    bathrooms   = np.random.randint(1, 5, n)
    sqft_living = np.random.randint(500, 6000, n)
    sqft_lot    = np.random.randint(1000, 15000, n)
    floors      = np.random.choice([1, 1.5, 2, 2.5, 3], n)
    age         = np.random.randint(0, 60, n)
    condition   = np.random.randint(1, 6, n)
    garage      = np.random.choice([0, 1], n, p=[0.3, 0.7])
    price = np.clip(
        60000 + sqft_living*130 + bedrooms*9000 + bathrooms*7000
        + floors*6000 + condition*4000 + garage*12000 - age*600
        + np.random.normal(0, 20000, n), 50000, 3000000).astype(int)
    df = pd.DataFrame({
        'bedrooms': bedrooms, 'bathrooms': bathrooms,
        'sqft_living': sqft_living, 'sqft_lot': sqft_lot,
        'floors': floors, 'age': age, 'condition': condition,
        'garage': garage, 'price': price
    })
    df['log_price'] = np.log1p(df['price'])
    return df


# ============================================================
#   STEP 2: TRAIN MODEL
# ============================================================

def train_model(df):
    feature_cols = [c for c in df.columns if c not in ['price', 'log_price']]
    X     = df[feature_cols]
    y_log = df['log_price']
    y_raw = df['price']

    X_train, X_test, ylog_train, ylog_test, yraw_train, yraw_test = train_test_split(
        X, y_log, y_raw, test_size=0.2, random_state=42)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = {}

    # Linear Regression
    print("\n🔄 Training Linear Regression...")
    lr      = LinearRegression()
    lr.fit(X_train_s, ylog_train)
    lr_pred = np.expm1(lr.predict(X_test_s))
    lr_r2   = r2_score(yraw_test, lr_pred)
    print(f"   Linear Regression R²  : {lr_r2*100:.2f}%")
    results['Linear Regression'] = (lr, lr_pred, lr_r2)

    # Random Forest
    print("🔄 Training Random Forest...")
    rf      = RandomForestRegressor(n_estimators=200, max_depth=15,
                                     min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, ylog_train)
    rf_pred = np.expm1(rf.predict(X_test_s))
    rf_r2   = r2_score(yraw_test, rf_pred)
    print(f"   Random Forest R²      : {rf_r2*100:.2f}%")
    results['Random Forest'] = (rf, rf_pred, rf_r2)

    # Gradient Boosting
    print("🔄 Training Gradient Boosting...")
    gb      = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                         max_depth=5, subsample=0.8,
                                         min_samples_split=5, random_state=42)
    gb.fit(X_train_s, ylog_train)
    gb_pred = np.expm1(gb.predict(X_test_s))
    gb_r2   = r2_score(yraw_test, gb_pred)
    print(f"   Gradient Boosting R²  : {gb_r2*100:.2f}%")
    results['Gradient Boosting'] = (gb, gb_pred, gb_r2)

    # XGBoost (most powerful!)
    print("🔄 Training XGBoost (most powerful)...")
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_train_s, ylog_train,
            eval_set=[(X_test_s, ylog_test)],
            verbose=False)
    xgb_pred = np.expm1(xgb.predict(X_test_s))
    xgb_r2   = r2_score(yraw_test, xgb_pred)
    print(f"   XGBoost R²            : {xgb_r2*100:.2f}%")
    results['XGBoost'] = (xgb, xgb_pred, xgb_r2)

    # Pick best
    best_name = max(results, key=lambda k: results[k][2])
    best_model, best_pred, best_r2 = results[best_name]
    print(f"\n🏆 Best Model: {best_name} ({best_r2*100:.2f}%)")

    joblib.dump(best_model,   'models/house_price_model.pkl')
    joblib.dump(scaler,       'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')

    return best_model, scaler, X_test_s, yraw_test, feature_cols, best_pred


# ============================================================
#   STEP 3: EVALUATE
# ============================================================

def evaluate_model(y_test, y_pred):
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print("\n" + "=" * 50)
    print("   📊 FINAL MODEL EVALUATION")
    print("=" * 50)
    print(f"   MAE  : ${mae:>12,.2f}")
    print(f"   RMSE : ${rmse:>12,.2f}")
    print(f"   R²   : {r2:>13.4f}  ({r2*100:.2f}% accuracy)")
    print("=" * 50)
    return mae, rmse, r2


# ============================================================
#   STEP 4: PLOTS
# ============================================================

def generate_plots(df, model, y_test, y_pred, feature_cols):
    sns.set_style("whitegrid")

    # EDA
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("House Price - Exploratory Data Analysis", fontsize=17, fontweight='bold')

    axes[0,0].hist(df['price'], bins=40, color='#4C72B0', edgecolor='white')
    axes[0,0].set_title('Price Distribution')
    axes[0,0].set_xlabel('Price ($)')
    axes[0,0].set_ylabel('Count')

    axes[0,1].scatter(df['sqft_living'], df['price'], alpha=0.2, color='#DD8452', s=8)
    axes[0,1].set_title('Price vs Living Area')
    axes[0,1].set_xlabel('Sqft Living')
    axes[0,1].set_ylabel('Price ($)')

    avg = df.groupby('bedrooms')['price'].mean()
    axes[0,2].bar(avg.index, avg.values, color='#55A868', edgecolor='white')
    axes[0,2].set_title('Avg Price by Bedrooms')
    axes[0,2].set_xlabel('Bedrooms')
    axes[0,2].set_ylabel('Avg Price ($)')

    if 'grade' in df.columns:
        avg_g = df.groupby('grade')['price'].mean()
        axes[1,0].bar(avg_g.index, avg_g.values, color='#C44E52', edgecolor='white')
        axes[1,0].set_title('Avg Price by Grade')
        axes[1,0].set_xlabel('Grade')
        axes[1,0].set_ylabel('Avg Price ($)')
    elif 'condition' in df.columns:
        avg_c = df.groupby('condition')['price'].mean()
        axes[1,0].bar(avg_c.index, avg_c.values, color='#4C72B0', edgecolor='white')
        axes[1,0].set_title('Avg Price by Condition')

    num_cols = ['price','sqft_living','bedrooms','bathrooms']
    if 'grade'     in df.columns: num_cols.append('grade')
    if 'age'       in df.columns: num_cols.append('age')
    if 'waterfront'in df.columns: num_cols.append('waterfront')
    num_df = df[num_cols].dropna()
    corr   = num_df.corr()
    im     = axes[1,1].imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1,1].set_xticks(range(len(corr.columns)))
    axes[1,1].set_yticks(range(len(corr.columns)))
    axes[1,1].set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    axes[1,1].set_yticklabels(corr.columns, fontsize=8)
    axes[1,1].set_title('Correlation Heatmap')
    plt.colorbar(im, ax=axes[1,1])

    if 'age' in df.columns:
        axes[1,2].scatter(df['age'], df['price'], alpha=0.2, color='#8172B2', s=8)
        axes[1,2].set_title('Price vs House Age')
        axes[1,2].set_xlabel('Age (years)')
        axes[1,2].set_ylabel('Price ($)')

    plt.tight_layout()
    plt.savefig('plots/eda_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/eda_dashboard.png")

    # Performance
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle("Model Performance Analysis", fontsize=15, fontweight='bold')

    axes2[0].scatter(y_test, y_pred, alpha=0.2, color='#4C72B0', s=8)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes2[0].plot([mn,mx],[mn,mx],'r--',lw=2,label='Perfect Fit')
    axes2[0].set_xlabel('Actual Price ($)')
    axes2[0].set_ylabel('Predicted Price ($)')
    axes2[0].set_title('Actual vs Predicted')
    axes2[0].legend()

    residuals = y_test.values - y_pred
    axes2[1].hist(residuals, bins=35, color='#DD8452', edgecolor='white')
    axes2[1].axvline(0, color='black', linestyle='--', lw=1.5)
    axes2[1].set_xlabel('Residual')
    axes2[1].set_ylabel('Frequency')
    axes2[1].set_title('Residual Distribution')

    importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_)
    top_n      = min(12, len(feature_cols))
    sorted_idx = np.argsort(importance)[-top_n:]
    axes2[2].barh(range(len(sorted_idx)), importance[sorted_idx], color='#55A868', edgecolor='white')
    axes2[2].set_yticks(range(len(sorted_idx)))
    axes2[2].set_yticklabels([feature_cols[i] for i in sorted_idx], fontsize=8)
    axes2[2].set_xlabel('Importance')
    axes2[2].set_title('Top Feature Importance')

    plt.tight_layout()
    plt.savefig('plots/model_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/model_performance.png")


# ============================================================
#   MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  🏠 HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("=" * 50)

    df = load_data()
    print(f"✅ Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")

    model, scaler, X_test_s, y_test, feature_cols, y_pred = train_model(df)
    mae, rmse, r2 = evaluate_model(y_test, y_pred)
    generate_plots(df, model, y_test, y_pred, feature_cols)

    print("\n✅ Training Complete! Model saved in /models folder.")
    print("▶  Now run:  streamlit run app.py")
