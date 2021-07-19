from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(val_X, val_y)
y_pred = gaussian.predict_proba(val_X)[:,1]
score = roc_auc_score(val_y, y_pred)
print(score)

# 绘制ROC曲线
FPR_gaussian, TPR_gaussian, THRESH_gaussian = roc_curve(val_y, y_pred)
draw_roc(FPR_gaussian, TPR_gaussian)
