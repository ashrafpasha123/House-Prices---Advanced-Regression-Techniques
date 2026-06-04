#House Prices — Advanced Regression Techniques
**Kaggle Competition | Ames Housing Dataset**
## Project Overview

Predict the final sale price of residential homes in Ames, Iowa using 79 explanatory features covering every aspect of a property — from basement ceiling height to proximity to railroads.

- **Goal:** Predict `SalePrice` for 1,459 test homes
- **Metric:** RMSE on log-transformed prices (RMSLE)
- **Dataset:** Ames Housing Dataset by Dean De Cock

##  File Structure

```
├── train.csv                 # 1460 rows, 81 columns (includes SalePrice)
├── test.csv                  # 1459 rows, 80 columns (no SalePrice)
├── data_description.txt      # Full feature descriptions
├── sample_submission.csv     # Required submission format
└── submission.csv            # Final predictions (ready to upload)

## Pipeline Summary

### 1. Missing Value Imputation
| Strategy | Features |
|---|---|
| Fill `'None'` | PoolQC, MiscFeature, Alley, Fence, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1/2, MasVnrType, MSSubClass |
| Fill `0` | GarageYrBlt, GarageArea, GarageCars, BsmtFinSF1/2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath, MasVnrArea |
| Neighborhood median | LotFrontage |
| Mode | All remaining categoricals |

### 2. Feature Engineering
| Feature | Formula |
|---|---|
| `TotalSF` | TotalBsmtSF + 1stFlrSF + 2ndFlrSF |
| `TotalBath` | FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath |
| `TotalPorch` | OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch |
| `HouseAge` | YrSold − YearBuilt |
| `RemodAge` | YrSold − YearRemodAdd |
| `HasPool`, `HasGarage`, `HasBsmt`, `HasFireplace` | Binary flags |

### 3. Encoding
- All categorical features label-encoded after combining train + test

### 4. Models & Ensemble

| Model | CV RMSE (log) | Weight |
|---|---|---|
| XGBoost | 0.12144 | 40% |
| LightGBM | 0.12913 | 40% |
| Gradient Boosting | 0.12339 | 20% |

Final predictions = weighted blend, then `expm1()` to convert back to dollars.

##  Submission Validation

| Check | Status |
|---|---|
| Rows |  1459 |
| Columns |  `Id`, `SalePrice` |
| ID range |  1461–2919 |
| Nulls |  0 |
| Negative prices |  0 |
| Price range |  $45k–$530k |

##  How to Submit

1. Go to [Kaggle Competition Page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
2. Click **Submit Predictions**
3. Upload `submission.csv`
4. View your RMSLE score on the leaderboard

##  Dependencies

pandas, numpy, scikit-learn, xgboost, lightgbm
Install:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm


## Dataset Features (Key)

| Category | Features |
|---|---|
| Location | Neighborhood, MSZoning, Condition1/2 |
| Size | GrLivArea, TotalBsmtSF, LotArea, GarageArea |
| Quality | OverallQual, OverallCond, ExterQual, KitchenQual |
| Age | YearBuilt, YearRemodAdd |
| Rooms | FullBath, BedroomAbvGr, TotRmsAbvGrd, Fireplaces |
| Extras | PoolArea, WoodDeckSF, MiscFeature |

Full description: `data_description.txt`

---

*SR University | Afreed | Kaggle Getting Started Competition*
