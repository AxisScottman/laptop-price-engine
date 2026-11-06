# Portfolio text (numbers from results.json)

## CV bullet
Built a laptop fair-price and hardware-tier engine on 1,274 laptops (Gradient Boosting + K-Means, R² 0.77, MAE cut 62% vs baseline), outperforming a rule-based hardware score (Spearman 0.938 vs 0.734), deployed as a Streamlit app with fleet CSV upload (scikit-learn, pandas, Streamlit).

## LinkedIn post
Is that laptop quote actually fair? IT teams often can't tell.

For my ML final project I built an app that estimates a laptop's fair price from its specs, flags quotes as good deal / fair / overpriced, and sorts machines into hardware tiers.

• Compared 3 models + a baseline on 1,274 laptops, tuned with cross-validation
• Best model: R² 0.77, median error 11.5%
• It ranks machines by price better than a hand-written points score (0.938 vs 0.734 Spearman)
• Lesson: raw text columns (CPU, storage, screen) hide the signal; parsing them mattered more than model choice, and duplicates had to go before splitting to avoid leakage.

Try it: [app link] · Code: [GitHub link]
#MachineLearning #DataScience

## Slide outline (10 slides)
1. Problem & who it helps  2. Data (size, columns, cleaning)  3. EDA insight: price skew, RAM/SSD/CPU drive price  4. EDA insight: brand/type shift  5. Pipeline & leakage control  6. Models & tuning  7. Test results table + actual-vs-predicted (assets/eval.png)  8. Error analysis by price band + feature importance  9. Model vs rule-based ranking + tiers (assets/tiers_pca.png)  10. Demo + recommendation + limitations
