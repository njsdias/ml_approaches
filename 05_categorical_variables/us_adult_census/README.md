The objective of this dataset is to predict the salary bracket.


The dataset is available in:
https://www.kaggle.com/uciml/adult-census-income

## Dataset Description

This dataset has the following columns:
- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


By data exploration we observed a skewed in target feature(income):
- `<=50k`: 24720 records
- `<50k`: 7841 records  

it represents ~24% of the total number of samples.

The most appropriated metric for this problem is AUC.



## Run code
The -W ignore is to ignore all warnings

`python3 -W ignore model_file.py` in which, 
`model_file.py` can take the values:
- `ohe_logres.py` : Logistic Regression with One-Hot Encoder
- `lbl_xgb.py`: XGBoost onl with categorical features
- `lbl_xgb_num.py`: XGBoost with numerical features applying LabelEncoder to categorical features 
- `lbl_xgb_num_feat.py`: XGBoost with Feature Engineering
- `target_encoding.py`: XGBoost with Target Encoding

## Results
|Model|AUC|
|-----|-----|
|Logistic Regression| 0.8787|
|XGBoost only with categorical features| 0.8750|
|XGBoost with numerical features applying LabelEncoder to categorical features| 0.9262|
|XGBoost with Feature Engineering| 0.9240|
|XGBoost using Target Encoding|0.9259|
|Neural Network with Entity Embedding|dfgdfgh



### Logistic Regression 

Here, we work only with categorical features.

The AUC values are similar in each fold. The average AUC is 0.78787 .

### XGBoost only with categorical features
Here, we work only with categorical features.

For applying XGBoost, instead of one-hot encoding a `LabelEncoder`
was applied in every categorical feature column to convert them in numeric values.

The AUC values are similar in each fold. The average AUC is 0.8750 .

### XGBoost with numerical features applying LabelEncoder to categorical features
Here, the numerical columns was keep. All categorical variables were label encoded. So, our final
feature matrix consists of numerical columns (as it is) and encoded categorical
columns.

The AUC values are similar in each fold. The average AUC is 0.9262 .

### XGBoost with Feature Engineering
Here all the categorical columns were taken and create all combinations of degree two.

The AUC values are similar in each fold. The average AUC is 0.9240 .

### XGBoost using Target Encoding

We must be very careful when using target encoding as it is too prone to overfitting.
In this situation an advice is use scikit-learn with target encoding with smoothing.

The AUC values are similar in each fold. The average AUC is 0.9259 .
