# Classification Results

## Performance Metrics

| Classifier              |   F1-Score |   Recall |   Precision |   Accuracy |
|:------------------------|-----------:|---------:|------------:|-----------:|
| KNN Classifier with k=3 |   0.743962 | 0.758621 |    0.801724 |   0.680542 |
| Logistic Regression     |   0.663643 | 0.689655 |    0.671121 |   0.602709 |
| Support Vector Machine  |   0.552622 | 0.62069  |    0.79454  |   0.489655 |

## Confusion Matrix - KNN Classifier with k=3

|        |   Predicted 0 |   Predicted 1 |   Predicted 2 |
|:-------|--------------:|--------------:|--------------:|
| True 0 |            13 |             0 |             0 |
| True 1 |             0 |             3 |             6 |
| True 2 |             0 |             1 |             6 |

## Confusion Matrix - Logistic Regression

|        |   Predicted 0 |   Predicted 1 |   Predicted 2 |
|:-------|--------------:|--------------:|--------------:|
| True 0 |            13 |             0 |             0 |
| True 1 |             2 |             3 |             4 |
| True 2 |             1 |             2 |             4 |

## Confusion Matrix - Support Vector Machine

|        |   Predicted 0 |   Predicted 1 |   Predicted 2 |
|:-------|--------------:|--------------:|--------------:|
| True 0 |            13 |             0 |             0 |
| True 1 |             8 |             1 |             0 |
| True 2 |             3 |             0 |             4 |