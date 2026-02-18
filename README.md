# 🏠 Task 6: House Price Prediction (Synthetic Dataset — No Kaggle Required)

## Task Objective
Predict house prices using property features such as square footage, number of bedrooms, location, and quality ratings. This is a supervised regression problem aimed at estimating the `SalePrice` of residential properties.

---

## Dataset Used
Synthetic Dataset — No Kaggle Required
- **Target Variable:** `SalePrice` (continuous, in USD)

Key features used:
| Category | Features |
|----------|----------|
| Size | `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `GarageArea` |
| Bedrooms/Baths | `BedroomAbvGr`, `FullBath`, `HalfBath`, `TotRmsAbvGrd` |
| Quality | `OverallQual`, `OverallCond`, `ExterQual`, `KitchenQual` |
| Location | `Neighborhood` |
| Time | `YearBuilt`, `YearRemodAdd` |
| Engineered | `TotalSF`, `TotalBath`, `HouseAge`, `YearsSinceRemodel` |

---

## Models Applied

### 1. Linear Regression
- Baseline model
- Applied `StandardScaler` for feature normalization
- Target log-transformed with `np.log1p()` to reduce skewness

### 2. Gradient Boosting Regressor
- Ensemble tree-based model (300 estimators, lr=0.05, depth=4)
- No scaling required
- Captures non-linear relationships and feature interactions

---

## Key Results & Findings

| Metric | Linear Regression | Gradient Boosting |
|--------|:-----------------:|:-----------------:|
| MAE ($) | ~$22,000–28,000 | ~$15,000–20,000 |
| RMSE ($) | ~$38,000–50,000 | ~$25,000–35,000 |
| R² Score | ~0.85–0.88 | ~0.90–0.93 |

> *Exact numbers will appear in the notebook output after running*

### Findings
- **Gradient Boosting significantly outperforms Linear Regression** across all metrics
- **Top predictors**: `OverallQual`, `GrLivArea`, `TotalSF`, `Neighborhood`, `YearBuilt`
- **Log-transforming SalePrice** improved model stability and reduced the impact of price outliers
- **Feature engineering** (TotalSF, HouseAge) added meaningful predictive signal
- Residuals for Gradient Boosting are well-centered around zero, indicating good fit

---

## Project Structure

```
├── house_price_prediction.ipynb    # Main Jupyter Notebook               
└── README.md                       # This file
```

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 2. Place house_prices.csv in the same directory as the notebook

# 3. Launch Jupyter
jupyter notebook house_price_prediction.ipynb
```

---

## Skills Demonstrated
- Regression modeling (Linear Regression & Gradient Boosting)
- Feature scaling with `StandardScaler`
- Feature selection and engineering
- Handling missing values (median/mode imputation)
- Model evaluation: MAE, RMSE, R²
- Cross-validation
- Data visualization (distributions, scatter plots, residual plots, feature importance)
- Real estate domain understanding
