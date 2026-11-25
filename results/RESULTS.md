# Results

Portuguese dataset, 649 students. Target: at risk = G3 < 10. 519 training students, 130 locked test students.
**CV** = 5-fold x 3 repeats on the 519 training students. **Test** = the 130 locked students, scored once.

| Model | CV accuracy | CV macro-F1 | CV at-risk recall | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test macro-F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Majority class | 0.846 ± 0.001 | 0.458 ± 0.000 | 0.000 | 0.500 | 0.846 | 0.000 | 0.000 | 0.458 | 0.500 |
| Naive rule (G2 < 10) | 0.896 ± 0.031 | 0.831 ± 0.046 | 0.887 | 0.963 | 0.900 | 0.621 | 0.900 | 0.837 | 0.953 |
| Logistic regression | 0.934 ± 0.027 | 0.870 ± 0.056 | 0.771 | 0.970 | 0.900 | 0.652 | 0.750 | 0.819 | 0.944 |
| Random forest | 0.921 ± 0.029 | 0.867 ± 0.045 | 0.908 | 0.974 | 0.862 | 0.531 | 0.850 | 0.784 | 0.946 |
| XGBoost | 0.929 ± 0.023 | 0.866 ± 0.045 | 0.804 | 0.971 | 0.900 | 0.640 | 0.800 | 0.825 | 0.925 |
| DNN (MLP) | 0.925 ± 0.027 | 0.846 ± 0.051 | 0.671 | 0.971 | 0.915 | 0.714 | 0.750 | 0.841 | 0.953 |

## DNN on the locked test set

- Saved model (used by the app): accuracy 0.915, Wilson 95% CI 0.855–0.952, bootstrap 95% CI 0.862–0.954.
- 5 retrained copies (seeds 0, 100, 200, 300, 400): accuracy 0.923 ± 0.008, macro-F1 0.850 ± 0.017. This spread measures only the random start, not the small test set.
- Confusion matrix: 104 true pass, 6 false alarms, 5 missed at-risk, 15 caught at-risk.

```
              precision    recall  f1-score   support

        Pass      0.954     0.945     0.950       110
     At risk      0.714     0.750     0.732        20

    accuracy                          0.915       130
   macro avg      0.834     0.848     0.841       130
weighted avg      0.917     0.915     0.916       130
```

McNemar exact test, DNN vs each model on the same 130 students (p < 0.05 would mean a real difference):

| Compared with | Only DNN right | Only other right | p-value |
|---|---|---|---|
| Majority class | 15 | 6 | 0.078 |
| Naive rule (G2 < 10) | 6 | 4 | 0.754 |
| Logistic regression | 6 | 4 | 0.754 |
| Random forest | 9 | 2 | 0.065 |
| XGBoost | 5 | 3 | 0.727 |

Selected features (10): school_MS, higher_yes, Medu, Fedu, studytime, failures, famrel, absences, G1, G2.
Hyperparameters: `{'n_layers': 2, 'width': 16, 'dropout': 0.47863507341115996, 'batchnorm': True, 'lr': 0.001996353900698521, 'weight_decay': 0.00843330780209246, 'batch_size': 32, 'class_weight': False, 'k': 10}`; best epoch 176.

![confusion](figures/confusion.png) ![roc](figures/roc.png)
![learning curve](figures/learning_curve.png)
