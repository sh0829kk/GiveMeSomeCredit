from sklearn.ensemble import RandomForestClassifier

grid = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={
        'n_estimators':[30,50,80,100,200]
    },
    scoring='roc_auc',
    verbose=3
)

grid.fit(train_X, train_y)
# for result in grid.cv_results_:
    # print(result, grid.cv_results_[result])
grid.best_params_['n_estimators']