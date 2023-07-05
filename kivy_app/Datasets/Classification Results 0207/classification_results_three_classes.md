# Classification Results

## Performance Metrics

| Classifier              |   F1-Score |   Recall |   Precision |   Accuracy |
|:------------------------|-----------:|---------:|------------:|-----------:|
| KNN Classifier with k=3 |   0.712233 | 0.724138 |    0.71176  |   0.724138 |
| Logistic Regression     |   0.677586 | 0.689655 |    0.782819 |   0.689655 |
| Support Vector Machine  |   0.34431  | 0.448276 |    0.608753 |   0.448276 |

## Confusion Matrix - KNN Classifier with k=3

|        |   Predicted 0 |   Predicted 1 |   Predicted 2 |
|:-------|--------------:|--------------:|--------------:|
| True 0 |            11 |             0 |             0 |
| True 1 |             2 |             6 |             2 |
| True 2 |             0 |             4 |             4 |

## Confusion Matrix - Logistic Regression

|        |   Predicted 0 |   Predicted 1 |   Predicted 2 |
|:-------|--------------:|--------------:|--------------:|
| True 0 |            11 |             0 |             0 |
| True 1 |             5 |             5 |             0 |
| True 2 |             3 |             1 |             4 |

## Confusion Matrix - Support Vector Machine

|        |   Predicted 0 |   Predicted 1 |   Predicted 2 |
|:-------|--------------:|--------------:|--------------:|
| True 0 |            11 |             0 |             0 |
| True 1 |             9 |             1 |             0 |
| True 2 |             6 |             1 |             1 |