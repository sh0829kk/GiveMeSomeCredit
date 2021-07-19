from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts



#split the data to  7:3
X_train,X_test,y_train,y_test = ts(train_X,train_y,test_size=0.1)

# select different type of kernel function and compare the score

# kernel = 'rbf'
begin = time.time()
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train,y_train)
score_rbf = clf_rbf.score(X_test,y_test)
print("The score of rbf is : %f"%score_rbf)
end = time.time()
print("training time:",end-begin,"s")
