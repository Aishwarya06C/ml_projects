# Results summary

## 01_house_price
```
              count         mean  ...          75%           max
MedInc      20640.0     3.870671  ...     4.743250     15.000100
HouseAge    20640.0    28.639486  ...    37.000000     52.000000
AveRooms    20640.0     5.429000  ...     6.052381    141.909091
AveBedrms   20640.0     1.096675  ...     1.099526     34.066667
Population  20640.0  1425.476744  ...  1725.000000  35682.000000
AveOccup    20640.0     3.070655  ...     3.282261   1243.333333
Latitude    20640.0    35.631861  ...    37.710000     41.950000
Longitude   20640.0  -119.569704  ...  -118.010000   -114.310000

[8 rows x 8 columns]
LinearRegression: MAE=0.533  R2=0.576
RandomForest: MAE=0.328  R2=0.804
Saved results.png
```

## 02_titanic
```
sex
female    0.727468
male      0.190985
Name: survived, dtype: float64
LogisticRegression: acc=0.809 f1=0.740
RandomForest: acc=0.786 f1=0.720
Saved results.png
```

## 03_sms_spam
```
label
ham     4825
spam     747
Name: count, dtype: int64
NaiveBayes
              precision    recall  f1-score   support

         ham       0.97      1.00      0.98       966
        spam       1.00      0.78      0.88       149

    accuracy                           0.97      1115
   macro avg       0.98      0.89      0.93      1115
weighted avg       0.97      0.97      0.97      1115

LogisticRegression
              precision    recall  f1-score   support

         ham       0.96      1.00      0.98       966
        spam       1.00      0.76      0.86       149

    accuracy                           0.97      1115
   macro avg       0.98      0.88      0.92      1115
weighted avg       0.97      0.97      0.97      1115

[1 0]
```

## 04_iris_wine
```
dataset        model  cv_accuracy   std
   Iris          KNN        0.960 0.025
   Iris          SVM        0.967 0.021
   Iris DecisionTree        0.953 0.034
   Wine          KNN        0.949 0.038
   Wine          SVM        0.983 0.022
   Wine DecisionTree        0.865 0.044
Saved results.png
```

## 05_churn
```
Churn rate: 0.141
LogisticRegression: ROC-AUC=0.870
RandomForest: ROC-AUC=0.902
GradientBoosting: ROC-AUC=0.916
              precision    recall  f1-score   support

           0       0.96      0.99      0.97       859
           1       0.90      0.74      0.81       141

    accuracy                           0.95      1000
   macro avg       0.93      0.86      0.89      1000
weighted avg       0.95      0.95      0.95      1000

Saved results.png
```

## 06_movie_recommender
```
3000  Emperor's New Groove, The (2000)  ...  3.72
1706                       Antz (1998)  ...  3.24
6486            Shrek the Third (2007)  ...  3.02

[5 rows x 4 columns]

Because you liked: Matrix, The (1999)
                                  title  ...  mean
474                 Blade Runner (1982)  ...  4.10
939              Terminator, The (1984)  ...  3.90
4118                 Equilibrium (2002)  ...  3.88
1961                    eXistenZ (1999)  ...  3.86
8696  Captain America: Civil War (2016)  ...  3.61

[5 rows x 4 columns]

Because you liked: Dark Knight, The (2008)
                              title  ...  mean
5917           Batman Begins (2005)  ...  3.86
7768  Dark Knight Rises, The (2012)  ...  3.99
7372               Inception (2010)  ...  4.07
8438        Edge of Tomorrow (2014)  ...  3.98
8252                 Gravity (2013)  ...  3.58

[5 rows x 4 columns]
```

## 07_sentiment
```
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/aishwarya/Downloads/ml-projects/07_sentiment/main.py", line 15, in <listcomp>
    parts = [pd.read_csv(z.open(n), sep="\t", names=["text", "label"], quoting=3)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 873, in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 300, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 1645, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 1922, in _make_engine
    return mapping[engine](f, **self.options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/pandas/io/parsers/c_parser_wrapper.py", line 95, in __init__
    self._reader = parsers.TextReader(src, **kwds)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/parsers.pyx", line 568, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 760, in pandas._libs.parsers.TextReader._get_header
  File "pandas/_libs/parsers.pyx", line 868, in pandas._libs.parsers.TextReader._tokenize_rows
  File "pandas/_libs/parsers.pyx", line 885, in pandas._libs.parsers.TextReader._check_tokenize_status
  File "pandas/_libs/parsers.pyx", line 2076, in pandas._libs.parsers.raise_parser_error
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa2 in position 37: invalid start byte
```

## 08_fraud
```
Fraud rate: 0.1727%
Traceback (most recent call last):
  File "/Users/aishwarya/Downloads/ml-projects/08_fraud/main.py", line 17, in <module>
    X["Time"] = StandardScaler().fit_transform(X[["Time"]])
                                               ~^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/pandas/core/frame.py", line 4384, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 6302, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 6352, in _raise_if_missing
    raise KeyError(f"None of [{key}] are in the [{axis_name}]")
KeyError: "None of [Index(['Time'], dtype='str')] are in the [columns]"
```

## 09_diabetes
```
preg      0
plas      5
pres     35
skin    227
insu    374
mass     11
pedi      0
age       0
dtype: int64
LogisticRegression: cv=0.782 test_acc=0.708 f1=0.545 recall=0.500
RandomForest: cv=0.766 test_acc=0.747 f1=0.614 recall=0.574
SVM: cv=0.766 test_acc=0.740 f1=0.600 recall=0.556
```

## 10_digits
```
SVM: accuracy=0.987
MLP: accuracy=0.973
Saved results.png
```

## 11_segmentation
```
Silhouette by k: {2: np.float64(0.44), 3: np.float64(0.493), 4: np.float64(0.509), 5: np.float64(0.555), 6: np.float64(0.51), 7: np.float64(0.472), 8: np.float64(0.422)} -> best k = 5
          age  income  spending  size
cluster                              
0        29.5    83.2      87.1    59
1        49.3    30.2      17.7    60
2        39.5    55.2      50.1    61
3        45.5    85.4      20.0    61
4        24.7    23.6      79.7    59
Saved results.png
```

## 12_fake_news
```
Kaggle files not found -> using 20newsgroups fallback
PassiveAggressive
                    precision    recall  f1-score   support

     comp.graphics       0.93      0.93      0.93       195
  rec.sport.hockey       0.96      0.89      0.92       200
         sci.space       0.83      0.90      0.87       197
talk.politics.misc       0.88      0.86      0.87       155

          accuracy                           0.90       747
         macro avg       0.90      0.90      0.90       747
      weighted avg       0.90      0.90      0.90       747

LogisticRegression
                    precision    recall  f1-score   support

     comp.graphics       0.95      0.94      0.95       195
  rec.sport.hockey       0.96      0.90      0.93       200
         sci.space       0.83      0.92      0.87       197
talk.politics.misc       0.91      0.87      0.89       155

          accuracy                           0.91       747
         macro avg       0.91      0.91      0.91       747
      weighted avg       0.91      0.91      0.91       747

```

## 13_loan
```
LogisticRegression: ROC-AUC=0.759
RandomForest: ROC-AUC=0.786
GradientBoosting: ROC-AUC=0.773

Best: RandomForest
              precision    recall  f1-score   support

      reject       0.63      0.45      0.52        60
     approve       0.79      0.89      0.84       140

    accuracy                           0.76       200
   macro avg       0.71      0.67      0.68       200
weighted avg       0.74      0.76      0.74       200

```

## 14_forecasting
```
SeasonalNaive(lag12): MAE=43.0 RMSE=47.2
LinearRegression: MAE=15.1 RMSE=17.9
RandomForest: MAE=31.2 RMSE=44.2
```

