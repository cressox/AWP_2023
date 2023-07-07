# Classification Results

## Performance Metrics

| Classifier              |   F1-Score |   Recall |   Precision |   Accuracy |
|:------------------------|-----------:|---------:|------------:|-----------:|
| KNN Classifier with k=3 |    1       | 1        |    1        |   0.947368 |
| Logistic Regression     |    1       | 1        |    1        |   0.904094 |
| Support Vector Machine  |    0.88797 | 0.894737 |    0.908772 |   0.712865 |

## Confusion Matrix - KNN Classifier with k=3

|        |   Predicted 0 |   Predicted 1 |
|:-------|--------------:|--------------:|
| True 0 |            13 |             0 |
| True 1 |             0 |             6 |

## Confusion Matrix - Logistic Regression

|        |   Predicted 0 |   Predicted 1 |
|:-------|--------------:|--------------:|
| True 0 |            13 |             0 |
| True 1 |             0 |             6 |

## Confusion Matrix - Support Vector Machine

|        |   Predicted 0 |   Predicted 1 |
|:-------|--------------:|--------------:|
| True 0 |            13 |             0 |
| True 1 |             2 |             4 |