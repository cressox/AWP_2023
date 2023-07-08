# Classification Results

## Performance Metrics

| Classifier              |   F1-Score |   Recall |   Precision |   Accuracy |
|:------------------------|-----------:|---------:|------------:|-----------:|
| KNN Classifier with k=3 |   0.608929 |    0.625 |    0.739583 |      0.625 |
| Support Vector Machine  |   0.59375  |    0.625 |    0.725    |      0.625 |
| Logistic Regression     |   0.38125  |    0.5   |    0.308333 |      0.5   |

## Confusion Matrix - KNN Classifier with k=3

|        |   Predicted 0 |   Predicted 1 |   Predicted 2 |
|:-------|--------------:|--------------:|--------------:|
| True 0 |             3 |             0 |             0 |
| True 1 |             0 |             1 |             2 |
| True 2 |             1 |             0 |             1 |

## Confusion Matrix - Support Vector Machine

|        |   Predicted 0 |   Predicted 1 |   Predicted 2 |
|:-------|--------------:|--------------:|--------------:|
| True 0 |             3 |             0 |             0 |
| True 1 |             1 |             1 |             1 |
| True 2 |             1 |             0 |             1 |

## Confusion Matrix - Logistic Regression

|        |   Predicted 0 |   Predicted 1 |   Predicted 2 |
|:-------|--------------:|--------------:|--------------:|
| True 0 |             3 |             0 |             0 |
| True 1 |             1 |             0 |             2 |
| True 2 |             1 |             0 |             1 |