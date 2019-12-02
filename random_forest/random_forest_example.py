from __future__ import division, print_function
import numpy as np
from sklearn import datasets
from utils import train_test_split, accuracy_score, Plot
from random_forest.random_forest_model import RandomForest
from sklearn.metrics import make_scorer

def loadDataSet(filename):
    fr = open(filename)
    data = []
    count = 0
    for line in fr.readlines():
        count = count + 1
        if count == 1 :
            continue
        lineAttr = line.strip().split(',')
        data.append([float(x) for x in lineAttr[1:]])
    #print(data)
    return np.array(data)

def loadLabel(filename):
    fr = open(filename)
    label = []
    count = 0
    for line in fr.readlines():
        count = count + 1
        if count == 1 :
            continue
        lineAttr = line.strip().split(',')
        label.append(float(lineAttr[-1])+1)
    #print(label)
    return np.array(label)

def KS(y_test, y_pred):
    '计算模型的ks值，返回float结果'
    m = len(y_test)
    tp = 0.0
    fp = 0.0
    fn = 0.0
    for i in range(m):
        if y_pred[i] > 1 and y_test[i] > 1:
            tp += 1
        if y_pred[i] < 1 and y_test[i] > 1:
            fp += 1
        if y_pred[i] > 1 and y_test[i] < 1:
            fn += 1
    if tp+fp == 0 :
        precision = 0
        print("tp+fp == 0")
    else:
        precision = tp / (tp + fp)

    if tp+fn == 0:
        recall = 0
        print("tp+fn == 0")
    else:
        recall = tp / (tp + fn)

    if precision+recall == 0:
        f1_score = 0
        print("precision+recall == 0")
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


def tiaocan(data,label):
    # 导入网格搜索模块
    from sklearn.model_selection import GridSearchCV

    rfr_best = RandomForest(n_estimators=50)
    params = [{
        'n_estimators': list(range(10, 50, 5)),
        'min_samples_split':  list(range(3, 20, 2)),
        'min_gain': [0, 0.5, 1, 1.5, 2, 2.5],
        'max_depth': list(range(5, 25, 2)),
        'max_features': list(range(1, 22, 1)),
    }]

    KS_score = make_scorer(KS, greater_is_better=True, needs_proba=False) # needs_threshold=True
    scoring = {'KS': KS_score}
    gs = GridSearchCV(rfr_best, params, cv=4, scoring=scoring, refit='KS',n_jobs=5, iid=True, pre_dispatch='2*n_jobs')
    gs.fit(data, label)

    # 查验优化后的超参数配置
    print(gs.best_score_)
    print(gs.best_params_)

def main():
    data = loadDataSet('../x_train.csv')
    label = loadLabel('../y_train.csv')

    # tiaocan(data,label)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.4, seed=19260817)
    print("X_train.shape:", X_train.shape)
    print("Y_train.shape:", y_train.shape)

    clf = RandomForest(n_estimators=35)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    m = len(y_test)
    tp = 0.0
    fp = 0.0
    fn = 0.0
    print(y_test)
    print(y_pred)
    for i in range(m):
        if y_pred[i] > 1 and y_test[i] > 1:
            tp += 1
        if y_pred[i] < 1 and y_test[i] > 1:
            fp += 1
        if y_pred[i] > 1 and y_test[i] < 1:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    print(tp, fp, fn)
    print("precision is: ", precision)
    print("recall is: ", recall)
    print("f1 score is: ", f1_score)


    Plot().plot_in_2d(X_test, y_pred, title="Random Forest") #accuracy=accuracy,#legend_labels=data.target_)


if __name__ == "__main__":
    main()