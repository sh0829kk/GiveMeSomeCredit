# 引入降采样模块
from imblearn.under_sampling import RandomUnderSampler
# Counter类的目的是用来跟踪值出现的次数
from collections import Counterx
print('Original dataset shape :', Counter(train_y))

# 调用模块
rus = RandomUnderSampler(random_state=111)

# 直接降采样后返回采样后的数值
X_resampled, y_resampled = rus.fit_resample(train_X, train_y)
print('Resampled dataset shape:', Counter(y_resampled))

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train_rus, X_test_rus, y_train_rus, y_test_rus = train_test_split(X_resampled, y_resampled, random_state=111)
X_train_rus.shape, y_train_rus.shape

# 对重采样以后的数据进行分类
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
logit_resampled = LogisticRegression(random_state=111, solver='saga', penalty='l1', class_weight='balanced', C=1.0, max_iter=500)

logit_resampled.fit(X_resampled, y_resampled)
logit_resampled_proba_res = logit_resampled.predict_proba(X_resampled)
logit_resampled_scores = logit_resampled_proba_res[:, 1]
fpr_logit_resampled, tpr_logit_resampled, thresh_logit_resampled = roc_curve(y_resampled, logit_resampled_scores)
draw_roc(fpr_logit_resampled, tpr_logit_resampled)
print('AUC score: ', roc_auc_score(y_resampled, logit_resampled_scores))

# 采用随机森林法分类和梯度上升法
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
forest = RandomForestClassifier(n_estimators=300, random_state=111, max_depth=5, class_weight='balanced')
forest.fit(X_train_rus, y_train_rus)
y_scores_prob = forest.predict_proba(X_train_rus)
y_scores = y_scores_prob[:, 1]
fpr, tpr, thresh = roc_curve(y_train_rus, y_scores)
draw_roc(fpr, tpr)
print('AUC score:', roc_auc_score(y_train_rus, y_scores))