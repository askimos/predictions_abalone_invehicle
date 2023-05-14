[![DOI](https://zenodo.org/badge/640607845.svg)](https://zenodo.org/badge/latestdoi/640607845)

## 1. Setup 
- Install Python 3.9.13, Jupyter, and the imported libraries
- Run `abalone_age_prediction.ipynb` and/or `in-vehicle_coupon_prediction.ipynb`
- Execute all code cells from top to bottom
- View the results in the notebook or view/use the created result files in results folder

## 2. Task
The objective of this study is to use two machine learning models to address a classification problem for two distinct datasets. Each dataset will be assessed with both machine learning models, utilizing various parameter configurations and preprocessing methods. The intention is to compare the outcomes of the two algorithms on the different datasets and analyze the findings.

## 3. Chosen datasets
Two datasets were used for the experiments:
 
### Dataset “Abalone Age” (OpenML [[1]](https://openml.org/d/44956))
- \# of instances:	4177
- \# of attributes:	8
- dataset characteristics:	Multivariate
- type of data:	    Categorical, Integer, Real
- missing values:	no
- file format	    .csv (1 file)

### Dataset “in-vehicle_coupon_recommendation” (UCI [[2]](https://archive-beta.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation))
- \# of instances:	12684
- \# of attributes:	23
- type of data:	    Numerical, Categorical
- missing values:	yes
- file format	    .csv (1 file) 

The task in the Abalone Age UCI dataset is to predict the age of abalone (a type of shellfish) based on a number of physical attributes such as length, diameter, height, and weight.
The goal in in-vehicle coupon recommendation dataset is to train a classifier that takes the features as input and predicts the target variable, which is whether or not the customer will redeem the coupon.


## 4. Experiment setup
Jupyter notebooks was used as a programming platform. Implementations of the chosen machine learning models were taken from the scikit-learn [[3]](https://scikit-learn.org/stable/) framework for Python [[4]](https://www.python.org/). The three chosen alogirthms were k-Nearest-Neighbours, Random Forest and Neural Networks.

### 4.1. Preprocessing
*Abalone Age*
- Missing values were handled by deletion.
- Categorical attributes where One-Hot encoded to allow compatability with k-NN and neural Networks.
- Simple search for outliers was carried out and found outliers has been removed from the dataset.
- Numerical attributes were either unscaled or normalized.

The official description of the dataset provided by the authors claimed that there were no missing values
present in the data. However, upon further investigation, we discovered that there were instances where the
Height variable had a value of 0.0. We decided to treat them as missing values and therefore removed them
from the dataset, since there are only 2 values that are missing, information loss is negligible.
One-hot encoding was preferred over label encoding since the sex feature has no intrinsic order or relationship. With only one categorical variable, this typically computationally expensive approach was stillfeasible.
The numerical data was preprocessed in two ways, namely unscaled and normalized using the StandardScaler from the sklearn library.
To prevent data leakage, all pre-processing techniques were applied after splitting the data into training
and testing sets.

*In-vehicle Coupon Recommendation*
Only three attributes are numerical (time of departure, coupon expiration and temperature) with few fixed
values that could be chosen. Hence outliers are not present. Due to the sub-optimal time format (’2PM’,
’10AM’ etc.) for time of departure, and coupon expiration (’1d’, ’2h’) that cannot be used numerically,
time of departure and coupon expiration where converted into hours (0-24, e.g. ’2PM’ = 14). The other
20 attributes are categorical or ordinal (e.g. the coupon type ’Restaurant(≤20)’, ’Carry out Take away’
etc.). The data set contains a significant amount of features with missing values. Data
preparation strategies use
- Missing values were either dealt with by making them an own category, or inserting the most frequent
value of all samples.
- Categorical attributes where One-Hot encoded to allow compatability with k-NN and neural Networks.
- A standard time format was used to allow for numerical computation of numerical attributes.
- Numerical attributes were either unscaled, min-max scaled or normalized. Scaling makes sense since
the time attributes might be weighted too much compared to the One-Hot encoded attributes otherwise.



## 5. Findings
In Abalone Age task every algorithm did much better than the baseline as expected. Scaling helps mainly in KNN case as expected as it is sensitive to differences in the scales of the features. . All algorithms performed similarly when fine-tuned for the given problem. We could say that KNN performed the worst and Neural Networks the best, but the difference is negligible.


The results of the In-Vehicle Coupon Recommendation task show that the Random Forest (RF) algorithm performs the best in terms of accuracy, while k-NN achieves the highest recall score. As expected, scaling the data has no effect on the performance of RF, but has a positive effect on the other algorithms. Both hold-out and cross-validation (CV) lead to similar accuracy scores, but there are stronger fluctuations in the recall scores. It is worth noting that different scaling techniques have been tested for other parameters, and similar results have been achieved with roughly the same hyperparameters leading to an optimal outcome. The baseline accuracy, which is the accuracy achieved by guessing the majority class, is 0.57. Therefore, the models significantly outperform guessing in terms of accuracy.

## 6. Metadata 
- Date of experiment completion: 14.05.2023
- `final_comparison.png`: Left: Comparison of optimal hyper parameters with cross validation (left) on the training set. Right:
Retrained model on entire training set with optimal hyper parameters and final generalization error
estimate on training set; we distinguish between different scaling strategies: 1 refers to no scaling,
2 refers to min-max scaling and 3 to normalization.
- other png results should be read and interpreted as barcharts/plots according to the labeling on the axes.
- src folder contains source code needed to run the experiment
- results folder contains result files as png images
- eda folder contains EDA visualizations to help understand the data as png images
- data folder contains input data used in the experiment
- performance metrics used are: MSE and Accuracy for Abalone Age dataset, Recall and Accuracy for in-vehicle coupon recommendation dataet
- software needed to run the experiment is Python version 3.9.13
- more information about specific attributes of the data is provided on original webpages specified in paragraph 3.
