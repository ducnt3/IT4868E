## Overview

This document summarizes results from two follow-up experiments and compares them against the original baseline:

- **Baseline (Original)**: `personality_classification_colab.ipynb`
- **Experiment 1 (Leakage-aware preprocessing / split)**: `personality_classification_no_leakage.ipynb`
- **Experiment 2 (Larger dataset)**: `personality_classification_reddit.ipynb`

All numbers below are taken from the **saved outputs in the notebooks after execution**.

## Experimental setup (what changed)

### Baseline: `personality_classification_colab.ipynb`
- **Dataset**: MBTI (`mbti_1.csv`), **8,675 samples**
- **Features**: TF‑IDF (max 10,000 features, 1–2 grams)
- **Models compared**: Logistic Regression, Linear SVM, Random Forest, Naive Bayes, XGBoost
- **Metric reported**: **Accuracy** and **weighted F1**

### Experiment 1: Leakage-aware preprocessing / split (`personality_classification_no_leakage.ipynb`)
- Reformulated data to **post-level samples** (split user “50 posts” into individual posts).
- Used **GroupShuffleSplit by user_id** so **all posts from a user appear only in train OR test**.
- TF‑IDF is fit **only on train** (then transformed on test).

### Experiment 2: Larger dataset (`personality_classification_reddit.ipynb`)
- **Reddit dataset subset used**:
  - Loaded: **500,000** posts, **9,918** authors
  - After filtering (author ≥ 5 posts): **493,736** posts, **7,028** authors
  - After balanced sampling (`MAX_PER_TYPE = 50,000`): **331,467** posts, **7,015** authors
- Used **GroupShuffleSplit by author** to avoid leakage.
- TF‑IDF is fit **only on train**.

## Results (per trait)

### Accuracy (higher is better)

| Trait | Original | No‑Leakage | Reddit Large |
|------|---------:|-----------:|-------------:|
| Extraversion | 0.7550 | 0.7383 | 0.7315 |
| Openness | 0.8317 | 0.8704 | 0.9159 |
| Agreeableness | 0.7781 | 0.5910 | 0.5713 |
| Conscientiousness | 0.6478 | 0.5407 | 0.5111 |
| **Average** | **0.7532** | **0.6851** | **0.6825** |

### Weighted F1 (higher is better)

| Trait | Original (best=LogReg) | No‑Leakage (best model) | Reddit Large (best model) |
|------|------------------------:|-------------------------:|---------------------------:|
| Extraversion | 0.7593 | 0.6896 (Random Forest) | 0.6190 (Naive Bayes) |
| Openness | 0.8356 | 0.8104 (Naive Bayes) | 0.8759 (Naive Bayes) |
| Agreeableness | 0.7785 | 0.5915 (Logistic Regression) | 0.5689 (XGBoost) |
| Conscientiousness | 0.6478 | 0.5428 (Logistic Regression) | 0.5113 (Logistic Regression) |
| **Average** | **(see below)** | **0.6586** | **0.6438** |

**Baseline note (Original)**: average weighted F1 per model across 4 traits was:
- Logistic Regression: **0.7553** (best overall)
- Linear SVM: 0.7430
- XGBoost: 0.7152
- Random Forest: 0.6908
- Naive Bayes: 0.6852

## Key takeaways (comparison vs. baseline)

- **Leakage-aware experiment (No‑Leakage)**:
  - Overall **drops vs. Original** on **Extraversion / Agreeableness / Conscientiousness**.
  - This is consistent with a **harder, more realistic evaluation**: post-level prediction + strict user-group split reduces “author-style leakage”.
  - Openness remains strong (Acc 0.8704, F1 0.8104), but still below Reddit on this trait.

- **Larger dataset experiment (Reddit Large)**:
  - **Clear improvement on Openness** (Acc **0.9159**, F1 **0.8759**) vs both Original and No‑Leakage.
  - **Worse on the other three traits**, and **average accuracy is slightly below No‑Leakage** (0.6825 vs 0.6851).
  - This suggests that **more data alone** (especially noisy, post-level social text) does **not automatically** improve all traits under leakage-safe evaluation.

## Final conclusion

- The **Original notebook** reports the **best overall average accuracy** (0.7532), but its evaluation is not directly comparable to the stricter post-level, group-split settings; it can also be **optimistically biased** in practice when author-style cues or dataset artifacts are present.
- **No‑Leakage** provides the **most conservative** (and typically more trustworthy) estimate of generalization to unseen users: **average accuracy 0.6851**.
- **Reddit Large** shows that scaling data can help **selectively** (Openness improves substantially), but for other traits the task remains difficult without stronger representations (e.g., better feature engineering, author-level aggregation, or transformer fine-tuning) and/or improved noise handling.

