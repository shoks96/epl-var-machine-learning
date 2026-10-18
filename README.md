Premier League Match Analysis Using Machine Learning

Machine learning analysis of Premier League match dynamics and outcome prediction before and after the introduction of VAR.

Completed as part of my MSc Applied Data Science at the University of Essex.

Overview

This project investigates how the introduction of the Video Assistant Referee (VAR) relates to Premier League match dynamics and to the prediction of match outcomes. The 2019/20 season marks the introduction of VAR.

Note: The findings describe associations and predictive performance. They do not establish that VAR caused the observed changes.

Research Questions
How did selected match statistics differ between the pre-VAR and post-VAR periods?
How well did machine-learning models trained on pre-VAR matches predict outcomes in the post-VAR period?
Which match statistics were most influential in predicting match outcomes?
Dataset
Source: English Premier League & Championship Full Dataset (Kaggle), credited to Joseph Buchdahl (12Xpert). The Premier League file was used. The dataset is not redistributed in this repository; please download it from the link above and check its licence terms.
Raw file: 12,153 Premier League matches, 1993/94 to 2024/25
After cleaning: 9,329 matches, 2000/01 to 2024/25. Seasons before 2000/01 were removed because shots, fouls, corners and card data were missing.
Pre-VAR (training): 7,220 matches (2000/01 to 2018/19)
Post-VAR (testing): 2,109 matches (2019/20 to 2024/25, including a partial 2024/25 season)
COVID flag: 2020/21 (played without crowds) is flagged and tested separately.
Leakage control: Full-time goal variables were excluded from the models.

The 12 model features (home and away for each): shots, shots on target, fouls, corners, yellow cards, red cards.

These are in-match statistics, so the models classify match outcomes from how a match played out. They are not pre-match forecasting models.

Methods
Data cleaning and exploratory data analysis
Welch's t-test and Mann-Whitney U test
Negative Binomial regression (VAR as predictor)
Multinomial logistic regression (match outcome ~ VAR)
Machine-learning model comparison
Robustness and sensitivity checks (recent-seasons window, COVID season excluded)
SHAP-based feature interpretation
Machine Learning

Models evaluated: Logistic Regression, Decision Tree, Random Forest, XGBoost, Multilayer Perceptron (MLP) and a Deep Neural Network (DNN, 128-64-32 dense layers with batch normalisation, dropout and early stopping).

Models were trained on pre-VAR matches and evaluated on post-VAR matches, to test how well predictions generalised across the two periods.

Results
Model comparison (trained pre-VAR, tested post-VAR)
Model	Test accuracy	Weighted F1	5-fold CV accuracy (pre-VAR)
Majority-class baseline (always predict home win)	43.7%	n/a	n/a
Deep Neural Network	60.36%	0.549	n/a
Logistic Regression	59.27%	0.541	55.83%
Random Forest	58.42%	0.554	53.95%
XGBoost	55.33%	0.540	52.70%
MLP	52.16%	0.508	49.10%
Decision Tree	46.18%	0.468	42.34%

The DNN was the most accurate model, but only about one percentage point ahead of Logistic Regression, so a simple linear model performs almost as well on these features.

Robustness checks
Recent-seasons window (2014/15 to 2018/19 vs 2019/20 to 2023/24, 1,900 matches each): Logistic Regression 61.84%, Random Forest 58.95%, XGBoost 56.84%, MLP 50.47%, Decision Tree 46.74%.
COVID season excluded: results were almost unchanged (Logistic Regression 59.22%, Random Forest 58.18%, XGBoost 55.35%).
Changes in match statistics (pre- vs post-VAR)
Statistic	Pre-VAR mean	Post-VAR mean	Rate ratio (Neg. Binomial)	Significant (full period)
Away goals	1.137	1.325	1.166	Yes
Home goals	1.530	1.559	1.019	No
Home yellow cards	1.394	1.697	1.217	Yes
Away yellow cards	1.760	1.873	1.064	Yes
Home red cards	0.064	0.055	0.861	No
Away red cards	0.093	0.062	0.662	Yes
Home fouls	11.486	10.678	0.930	Yes
Away fouls	12.058	10.835	0.899	Yes

Match outcomes shifted from 46.5% home wins, 28.3% away wins and 25.2% draws pre-VAR to 43.7%, 33.0% and 23.3% post-VAR.

Key findings
Away goals, away red cards and away fouls differed significantly between periods, and these differences remained significant in the balanced recent-seasons window (away goals rate ratio 1.110, away red cards 0.767).
Yellow-card and home-foul differences weakened or disappeared in the recent-seasons window, which suggests part of the full-period difference may reflect longer-term trends rather than a change at the point VAR was introduced.
Shots on target were the most influential features in the SHAP analysis (home shots on target 0.083, away shots on target 0.065 mean |SHAP|), well ahead of shots, corners, fouls and cards.
SHAP values were computed on a 20-tree Random Forest using a 200-match post-VAR sample, so they describe that model rather than the DNN.
Tools and Technologies

Python, pandas, NumPy, Matplotlib, Seaborn, SciPy, statsmodels, scikit-learn, XGBoost, TensorFlow/Keras, SHAP, Jupyter Notebook

Reproducibility
Clone the repository:
   git clone https://github.com/shoks96/epl-var-machine-learning.git
   cd epl-var-machine-learning
Create an environment and install dependencies:
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
Download the dataset from the Kaggle link above and save it as Dataset1_Premier_League_1993_2024.csv. Update the file path in the notebook's data-loading cell (it was written in Google Colab and reads from /content/sample_data/).
Open and run VAR_Dissertation_Analysis_Complete.ipynb in Jupyter.

Random seeds are fixed (42) for the classical models and the DNN.

Limitations
Results are associations, not causal effects. VAR's introduction coincides with other changes (rule interpretations, playing trends, the COVID-affected 2020/21 season).
The post-VAR period includes a partial 2024/25 season.
Features are in-match statistics, so the models do not forecast matches before kick-off.
The DNN result comes from a single trained model, and its margin over Logistic Regression is small.
Draws are hard to predict; weighted F1 scores (about 0.55) are lower than accuracy.
What I Learned

This project strengthened my skills in statistical analysis, data preparation, machine-learning evaluation and communicating findings from complex datasets. It also highlighted the importance of identifying data leakage and testing models on data from a different period.

Author

Gbenga Shokunbi · LinkedIn · GitHub
