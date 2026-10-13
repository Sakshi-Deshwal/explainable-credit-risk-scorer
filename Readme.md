# Explainable Credit-Risk Scorer

A model that flags borrowers likely to become seriously delinquent, with the **decision threshold chosen by business cost** (not by the default 0.5) and **SHAP explanations** for why each applicant is flagged.

> **Result (on 29,946 untouched test borrowers, 1,976 seriously delinquent):** the Random Forest, flagged at threshold 0.050, catches **82%** of delinquent borrowers while flagging **31%** of all applicants (precision 18%). Under the assumed 20:1 cost ratio, its total cost is **29.3M cost units, 63% lower than approving everyone and 48% lower than flagging everyone**. The default 0.5 threshold would cost 69.9M, which is 25% *worse* than flagging everyone.

## The problem

A lender must decide which applicants to flag as high risk. The two mistakes are not equally expensive:

- **False negative** (approve someone who later defaults): costly. Assumed **40,000** cost units.
- **False positive** (flag a safe borrower): cheaper. Assumed **2,000** cost units.

Accuracy is a poor yardstick here because only about 6.6% of borrowers default (a model that approves everyone is "93% accurate"). So this project picks the threshold that minimises **total cost**, and reports how sensitive that choice is to the cost assumption.

## Data

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle): 150,000 borrowers, 10 features, target `SeriousDlqin2yrs` (90+ days past due within 2 years). Please check Kaggle's terms before redistributing the data; this repo does not include it. Download `cs-training.csv` and place it next to the notebook.

**Cleaning decisions (and why)**

| Issue found | What I did |
|---|---|
| 1 row with age 0 | Removed |
| 269 rows with sentinel codes 96/98 in the "times 90 days late" column | Removed (these are placeholder codes, not real counts) |
| Utilization values far above 2 (371 rows) | Clipped at 2 |
| Extreme `DebtRatio` values | Log-transformed inside the pipeline |
| `MonthlyIncome` missing (~20%), `NumberOfDependents` missing | Median imputation **inside the pipeline**, plus missing-indicator columns |

After cleaning: 149,730 rows.

## Method

1. **One stratified 60/20/20 split** (train 89,838 / validation 29,946 / test 29,946), same 6.6% positive rate in each.
2. **Imputation and scaling live inside a scikit-learn Pipeline**, so they are fitted only on training folds (no leakage).
3. **Baselines first:** approve-everyone, flag-everyone and a dummy classifier.
4. **Four models tuned with GridSearchCV** (3-fold, scored by PR-AUC, which suits imbalanced data): Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
5. **Model and threshold chosen on the validation set only.** Models are ranked by the lowest cost at their cheapest threshold.
6. **The test set is used once**, at the very end.

## Results

### Model comparison (validation set)

| Model | Val ROC-AUC | Val PR-AUC | Cheapest threshold | Cost at cheapest |
|---|---|---|---|---|
| **Random Forest** | 0.873 | 0.426 | 0.045 | 27.1M |
| Gradient Boosting | 0.871 | 0.419 | 0.045 | 27.6M |
| Decision Tree | 0.857 | 0.395 | 0.040 | 28.7M |
| Logistic Regression | 0.861 | 0.406 | 0.045 | 29.1M |
| Dummy | 0.500 | 0.066 | n/a | n/a |

Random Forest and Gradient Boosting are within 1.5% of each other, so they are effectively tied on this data. I deployed Random Forest because it had the lowest validation cost.

### Choosing the threshold

The cheapest threshold was 0.045, and every threshold from 0.040 to 0.050 is within 2% of that cost. Because the cost curve is flat near the minimum, I chose the **highest threshold inside that band (0.050)**, which flags fewer people for nearly the same cost.

### Final test performance (threshold 0.050, evaluated once)

| Metric | Value |
|---|---|
| ROC-AUC | 0.857 |
| PR-AUC | 0.391 (no-skill baseline: 0.066) |
| Recall | 0.820 |
| Precision | 0.177 |
| F1 | 0.291 |
| Confusion matrix | TN 20,429 · FP 7,541 · FN 356 · TP 1,620 |

| Strategy | Total cost | Saving vs approve-all |
|---|---|---|
| Approve everyone | 79.0M | 0% |
| Flag everyone | 55.9M | 29% |
| Random Forest at default 0.5 | 69.9M | 12% |
| **Random Forest at 0.050** | **29.3M** | **63%** |

Cost per 1,000 applicants was 915k on validation and 979k on test (+7%), a small rise that is expected because the threshold was tuned on validation.

### Explainability (SHAP)

The strongest drivers of risk are credit-card utilization, the number of 90+ day late payments, and the number of 30–59 day late payments. Income, debt ratio and missing-value indicators matter little. A waterfall plot explains the single highest-risk applicant (predicted risk 0.74: six recent 30–59 day lates, five serious lates, 109% utilization).

*(Add the SHAP bar plot, beeswarm and waterfall images here, e.g. `![SHAP summary](images/shap_summary.png)`.)*

## Sensitivity to the cost assumption

The cost ratio is an **assumption**, and the best threshold moves with it:

| FN : FP cost ratio | Cheapest threshold | Extra cost if I keep threshold 0.050 |
|---|---|---|
| 5 : 1 | 0.180 | +52% |
| 10 : 1 | 0.120 | +12% |
| 20 : 1 (used) | 0.045 | +1% |
| 40 : 1 | 0.030 | +10% |

A real lender would supply its own costs and re-run the threshold step. The method is the point of this project, not the specific 20:1 number.

## Limitations

- Costs are assumed, not measured. See the sensitivity table.
- Test ROC-AUC (0.857) is a little lower than validation (0.873). I did not compute confidence intervals; bootstrapping the test set would quantify this.
- Three of the four models' best settings sat at the edge of their search grid, so a wider search might gain a little. The models are close enough that I would not expect a different conclusion.
- The data is a single historical snapshot with no time dimension, so I cannot test drift over time.
- This is a modelling exercise. A real credit decision would also require fairness checks, regulatory review and monitoring.

## Next steps

- Bootstrap confidence intervals on the test metrics.
- Calibrate the predicted probabilities and check calibration curves.
- Fairness audit across age groups.
- Wrap the final pipeline in a small API.

## Reproduce

```bash
pip install pandas numpy scikit-learn matplotlib shap
# put cs-training.csv next to the notebook, then run Week4_project_v2.ipynb top to bottom
```

Full run takes roughly 15 minutes, mostly the Random Forest and Gradient Boosting grid searches.

**Stack:** Python, pandas, scikit-learn (Pipelines, GridSearchCV), SHAP, matplotlib.
