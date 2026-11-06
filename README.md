# 💻 Laptop Fair-Price & Value Engine

**IT managers and buyers can't easily tell whether a machine is worth its price. This project estimates a laptop's fair price from its specs, flags quotes as good deal / fair / overpriced, and sorts machines into hardware tiers.**

🔗 **Live app:** _add your Streamlit Community Cloud link here_  
📸 **Screenshot / GIF:** _add after deploying (e.g. `assets/app.png`)_

## Results (held-out test set, 255 laptops)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 28,663 | 43,464 | -0.10 |
| Ridge | 14,094 | 28,207 | 0.54 |
| Random Forest | 11,174 | 20,641 | 0.75 |
| Gradient Boosting | 10,996 | 19,895 | 0.77 |

Best model by cross-validation: **Gradient Boosting** (test MAE 10,996 vs 28,663 for the baseline; R² 0.77).
Median error 11.5%; 73.3% of predictions are within 20% of the real price.
Errors grow for expensive machines (median error 9.3% in the cheapest quarter vs 16.4% in the priciest quarter).
Prices are in the dataset's own currency units.

**Learned model vs hand-written ranking.** Spearman correlation with real price: rule-based hardware score **0.734** vs learned model **0.938**.

**Hardware tiers (K-Means, specs only, k=4):**

| Tier | Laptops (train) | Median price | RAM GB | SSD GB | HDD GB | Dedicated GPU |
|---|---|---|---|---|---|---|
| Tier 1 · Entry | 278 | 30,077 | 4 | 0 | 500 | 0% |
| Tier 2 · Mainstream | 295 | 44,276 | 8 | 128 | 1000 | 100% |
| Tier 3 · Slim Premium | 320 | 74,272 | 8 | 256 | 0 | 7% |
| Tier 4 · High-Performance | 126 | 102,005 | 16 | 256 | 1000 | 100% |

Cluster separation is modest (silhouette 0.26 at best), so tiers are a convenient grouping, not sharp boundaries.

## What's inside
| File | Purpose |
|---|---|
| `Laptop_Price_Value_Engine.ipynb` | Full pipeline: framing, EDA, preprocessing, modeling, tuning, evaluation, tiers, insights |
| `features.py` | Row-wise parsing of raw spec text into features (shared by notebook and app) |
| `app.py` | Streamlit app (single machine check + fleet CSV upload) |
| `models/` | Saved pipeline (`.joblib`), tier model, app metadata |
| `data/laptop_data.csv` | Small dataset (1,303 rows) |
| `results.json` | Every number above, written by the notebook |

## Method in short
- Removed exact duplicate rows **before** the split, then made an 80/20 random train/test split.
- Text columns (CPU, RAM, storage, screen) parsed into numeric features; imputing, scaling and one-hot encoding live inside a scikit-learn `Pipeline`.
- Target modeled as `log(price)` via `TransformedTargetRegressor`.
- Baseline + Ridge, Random Forest, Gradient Boosting; `RandomizedSearchCV` (5-fold, MAE) on training data only.
- Final deployed pipeline is refit on all rows; reported scores come from the train-only model on the untouched test set.

## Run it
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Laptop_Price_Value_Engine.ipynb   # or Restart & Run All
streamlit run app.py
```
Deploy: push to GitHub, then create an app on [Streamlit Community Cloud](https://streamlit.io/cloud) pointing at `app.py`.

## Limitations
~1,274 unique laptops from one snapshot, new retail prices only, no date/age/condition, coarse CPU/GPU grouping. Decision aid, not a valuation.

## Data credit
`laptop_data.csv` comes from the GitHub repo [AAdewunmi/laptop_prices_prediction_project](https://github.com/AAdewunmi/laptop_prices_prediction_project) and appears to be the widely used Kaggle *Laptop Price* dataset; check the original source and license before publishing.
