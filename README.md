# 2025_DSAI4204_HAAP: Heart Attack Analysis & Prediction

This repository contains the project work for DSAI4204, focusing on automating risk screening for heart attacks using distinct mathematical approaches: geometric grouping (Unsupervised Learning) and decision logic (Supervised Learning). The project culminates in a "Merged" pipeline that integrates both methodologies for enhanced predictive performance.

## Project Structure

The project is organized into three main directories, each representing a distinct approach to heart attack prediction:

*   **`Supervised/`**: Contains the implementation for supervised learning models. This pipeline focuses on finding direct patterns in the data to classify heart attack risk.
    *   `config.py`: Configuration settings for supervised models, including data paths, feature lists, and model hyperparameters.
    *   `dataloader.py`: Handles loading and initial cleaning of the dataset.
    *   `preprocessing.py`: Implements comprehensive data preprocessing steps such as handling missing values, outlier detection, feature transformations, and scaling.
    *   `train.py`: Manages the training of individual supervised models (Logistic Regression, SVM, Random Forest, Gradient Boosting) and the final Voting Classifier.
    *   `evaluate.py`: Provides functions for evaluating model performance, generating classification reports, confusion matrices, and ROC curves.
    *   `runSupervisedHaap.py`: The main script to execute the supervised learning pipeline.

*   **`Unsupervised/`**: Contains the implementation for unsupervised learning models. This pipeline explores the inherent structure and groupings within the data without prior labels.
    *   `config.py`: Configuration settings for unsupervised models, including data paths, feature lists, and clustering hyperparameters (K-Means, DBSCAN).
    *   `dataloader.py`: Handles loading and initial cleaning of the dataset.
    *   `preprocessing.py`: Implements data preprocessing tailored for unsupervised learning, including handling missing values, outlier detection, and feature transformations.
    *   `train.py`: Manages the training of clustering models (K-Means, DBSCAN) and association rule mining.
    *   `evaluate.py`: Provides functions for evaluating clustering results and analyzing association rules.
    *   `runUnsupervisedHaap.py`: The main script to execute the unsupervised learning pipeline.

*   **`Merged/`**: Integrates concepts from both supervised and unsupervised learning to create a hybrid prediction pipeline. This is the core innovation of the project, where cluster labels from unsupervised learning are used as features in the supervised models.
    *   `config.py`: Configuration settings for the merged pipeline, extending supervised configurations with cluster label integration.
    *   `dataloader.py`: Loads the dataset and merges it with pre-computed cluster labels from the unsupervised pipeline.
    *   `preprocessing.py`: Applies preprocessing steps similar to the supervised pipeline, adapted for the inclusion of cluster labels.
    *   `train.py`: Trains the ensemble of supervised models using the dataset augmented with cluster labels.
    *   `evaluate.py`: Evaluates the performance of the merged ensemble model.
    *   `runMergedHaap.py`: The main script to execute the merged learning pipeline.

## How to Run

To run the different pipelines, navigate to the root directory of the project and execute the respective `run` scripts.

### Supervised Learning Pipeline

```bash
python Supervised/runSupervisedHaap.py
```

### Unsupervised Learning Pipeline

```bash
python Unsupervised/runUnsupervisedHaap.py
```

### Merged Learning Pipeline

```bash
python Merged/runMergedHaap.py
```

---

## Executive Summary

The objective was to automate risk screening by comparing two distinct mathematical approaches: geometric grouping (Unsupervised Learning) and decision logic (Supervised Learning).

**Key Achievements:**

*   The final Ensemble Voting Model achieved a Test Accuracy of 90.32% and an ROC-AUC of 0.937, significantly outperforming baseline models.
*   A Merged Pipeline was developed that integrates latent cluster labels from unsupervised learning as predictive features into the supervised model, successfully bridging the gap between clustering and classification.

## Objective

The goal of this project was to construct an algorithmic solution that processes clinical datasets (e.g., age, cholesterol, chest pain) to output a classification risk score: 0 (Healthy) or 1 (High Risk).

## Methodology

As a requirement, the project utilized the Heart Attack Analysis & Prediction (HAAP) dataset from Kaggle. There are two Excel sheets: `heart.csv`, containing 303 patient records; `o2Saturation.csv` was analyzed but rejected because of significant row mismatch (303 vs. 3585 rows). For features, there are 13 clinical variables including `cp` (chest pain), `trtbps` (blood pressure), and `thalachh` (max heart rate).

I planned an experiment with two main parts to find the best way to process the data. This led to a final “Merged” pipeline that combines different methods to get better results.

### Track A: Unsupervised Learning (The “Geometry” Test)

This part focuses on the shape of the data. My idea was that patients with similar health problems would naturally group together. I used tools like K-Means Clustering and DBSCAN to look for these hidden groups without telling the computer who is sick or healthy. Even though clustering by itself cannot perfectly separate sick people from healthy ones, which the low Silhouette scores confirmed, it still finds important geometric patterns that other models might miss.

### Track B: Supervised Learning (The “Pattern” Test)

This part focuses on finding rules. My idea here was that heart attack risk comes from a complicated mix of different symptoms. To solve this, I used a group of models including Logistic Regression, SVM, Random Forest, and Gradient Boosting. I then combined all their answers using a Voting Classifier. This method uses the “wisdom of crowds” to make the predictions more stable and accurate when facing new data.

### About the Hybrid “Merged” Pipeline

The main new thing I did in this project was combining the two tracks above. The steps were simple:

First, I ran K-Means clustering on the data to give every patient a "cluster label" (Group 0 or Group 1). Next, I added this label back into the dataset as a new piece of information.

Finally, I trained the supervised models again using this updated data.

**Algorithms:** An ensemble of Logistic Regression, Support Vector Machine (SVM), Random Forest, and Gradient Boosting.

For optimization, a Voting Classifier was employed to aggregate predictions, leveraging the “wisdom of crowds” to reduce variance and improve generalization.

The reason for doing this is to let the supervised models see which “geometric group” a patient belongs to. This acts as an extra hint, making it easier for the computer to decide if a patient is high-risk or not.

## Implementation Details

I built the solution using Python and relied on standard libraries like scikit-learn and pandas to keep the code modular and easy to manage.

### Preprocessing Pipeline

I cleaned the data carefully by following a strict set of steps. First, I filled in the missing values for the `thall` column using the mode, which was 2.0. Then, I handled extreme values, known as outliers, using a method called Winsorization. This involved capping variables like `trtbps` and `oldpeak` at the 95th to 98th percentiles so they would not distort the model. For example, this step successfully reduced the skewness of `oldpeak` from 0.996 down to 0.136. After fixing the outliers, I removed features that had a weak connection to the target to reduce noise. I dropped columns like `chol`, `fbs`, and `restecg`, specifically noting that `chol` only had a correlation of 0.086. Finally, I used RobustScaler to scale the remaining numbers, as this tool handles outliers better than standard normalization methods.

### Model Training

For the training process, I split the dataset so that 90% was used for training (268 samples, including 10% used for validation) and 10% was reserved for testing (31 samples). I did this to ensure that the final evaluation scores would honestly reflect how well the model performs on new, unseen data.

## Analysis & Experimental Results

![Confusion Matrix](Figure_1_Confusion_Matrix.png)
*Figure 1: Confusion Matrix showing the model's performance on the test set.*

### Quantitative Performance

The final Ensemble Voting Model produced excellent results on the test set. It achieved an accuracy of 90.32%, meaning it correctly diagnosed 28 out of the 31 patients. The ROC-AUC score was 0.9370, which shows the model is very good at distinguishing between high-risk and low-risk patients. When predicting high-risk cases, the model had a precision of 0.89, so it was correct 89% of the time. Most importantly, the recall was 0.94, indicating that the model successfully identified 94% of the actual heart attack cases.

![Model Comparison](Figure_2_Model_Comparison.png)
*Figure 2: Output of comparing different models*

### Model Comparison

The logs showed that combining different models worked better than using them individually. The SVM model underperformed with only 66.67% accuracy, likely because the data boundaries were too complex for it. While the Logistic Regression and Random Forest models hit 100% accuracy on some validation tests, the Voting Classifier was the best choice overall. It provided the most stable performance, resulting in the final 90.32% accuracy on the test set.

![ROC Curve](Figure_3_ROC_Curve.png)
*Figure 3: ROC Curve. The Area Under Curve (0.937) indicates superior ranking ability.*

### Feature Importance Analysis

I looked at the correlation analysis to see which clinical factors mattered the most. The strongest predictor was Exercise Induced Angina (`exng`) with a score of 0.441. This suggests that chest pain triggered specifically by exercise is a major warning sign. The second strongest factor was Chest Pain Type (`cp`) at 0.432, confirming that atypical chest pain is a dominant indicator. ST Depression (`oldpeak`) came in third with a score of 0.420, proving that ECG abnormalities are highly relevant for predicting risk.

## Evaluation

My analysis proves that the “Merged Pipeline” strategy was a success, achieving a high ROC-AUC score of 0.937. By feeding the `cluster_label` directly into the supervised model, the system was able to use the hidden geometric structure of the data to make better decisions. The logs confirm that this new feature was relevant to the decision logic, showing a correlation of 0.188 with the target.

![Evaluation Report](Figure_4_Evaluation_Report.png)
*Figure 4: Output of evaluation report*

The model achieved a Recall of 0.94 on the test group, meaning it missed very few actual heart attack cases. In a medical reality setting, missing a heart attack is fatal, whereas a false alarm is just merely inconvenient. To ensure no at-risk patients are missed in the future, I recommend lowering the classification threshold from 0.50 to 0.35. This adjustment would prioritize sensitivity, catching every potential case even if it results in a slight increase in false alarms.

## Conclusion

This project has proven that heart attack risk can be predicted with over 90% accuracy using data mining techniques. The investigation highlighted that while unsupervised clustering alone is not enough to diagnose patients, it captures valuable latent patterns that should be utilized. My idea is integrating these unsupervised clusters as features into the Supervised Ensemble Model resulted in a highly accurate.

## Reference

[1] parthpandit05, “2025 heart attack analysis and prediction,” Kaggle, https://www.kaggle.com/code/parthpandit05/2025-heart-attack-analysis-and-prediction.

## License

This project is developed for academic purposes as part of PolyU DSAI4204 - Data Mining and Data Warehousing course. All rights reserved for educational use.
