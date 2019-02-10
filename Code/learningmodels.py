import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

class Model_Validation:
    def __init__(self, train_x, train_y, model, features):
        self.train_x = train_x
        self.train_y = train_y
        self.estimator = model
        self.features = features
        self.has_valid = False

    def set_estimator(self, estimator):
        self.estimator = estimator

    def set_valid_data(self, valid_x, valid_y):
        self.has_valid = True
        self.valid_x = valid_x
        self.valid_y = valid_y

    def plot_learning_curve(self, title, ylim= None, cv = None, n_jobs = 1, train_sizes=np.linspace(.1, 1.0, 10)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training samples")
        plt.ylabel("Error")
        train_sizes, train_scores, valid_scores = learning_curve(
            self.estimator, self.train_x, self.train_y, cv = cv, n_jobs = n_jobs, train_sizes=train_sizes)
        valid_scores = 1 - valid_scores
        valid_scores_mean = np.mean(valid_scores, axis=1)
        valid_scores_std = np.std(valid_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                        valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, valid_scores_mean, color="g",
                label="Cross-validation Error")
        plt.legend(loc="best")
        plt.savefig("../Figures/"+title+".png")
        return plt

    def plot_validation_curve_data(self, title, x_label, train_scores, valid_scores, new_valid_scores, param_range, minx=-1, maxx=-1, plot = True, fsize=None):
        train_scores = 1 - train_scores
        valid_scores = 1 - valid_scores
        new_valid_scores = 1 - new_valid_scores
        train_scores_mean = np.mean(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)
        valid_scores_std = np.std(valid_scores, axis=1)
        plt.figure(figsize=fsize)
        plt.title(title)
        plt.grid()
        plt.xticks(param_range)
        plt.xlabel(x_label)
        plt.ylabel("Error")
        if (minx != -1 & maxx != -1):
            plt.xlim(minx, maxx)
        lw = 2
        plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
                         valid_scores_mean + valid_scores_std, alpha=0.2,
                         color="g", lw=lw)
        if (plot == False):
            plt.semilogx(param_range, train_scores_mean, label="Training Set Error",
                        color="r", lw=lw)
            plt.semilogx(param_range, valid_scores_mean, label="Cross-Validation Error",
                        color="g", lw=lw)
            if (self.has_valid):
                plt.semilogx(param_range, new_valid_scores, label="Validation Set Error",
                        color="b", lw=lw)
        else:
            plt.plot(param_range, train_scores_mean, label="Training Set Error",
                        color="r")
            plt.plot(param_range, valid_scores_mean, label="Cross-Validation Error",
                        color="g")
            if (self.has_valid):
                plt.plot(param_range, new_valid_scores, label="Validation Set Error",
                        color="b")
        plt.legend(loc="best")

    def get_validation_curve_data(self, param_name, cv = None, n_jobs = 1, param_range = np.linspace(1, 10, 10)):
        train_scores, valid_scores = validation_curve(
            self.estimator, self.train_x, self.train_y, param_name=param_name, param_range=param_range,
            cv=cv, scoring="accuracy", n_jobs=n_jobs)
        return train_scores, valid_scores

    def get_scores(self, estimator):
        import time
        start_time = time.time()
        estimator.fit(self.train_x, self.train_y)
        predict = estimator.predict(self.valid_x)
        tot_time = time.time() - start_time
        accuracy = metrics.accuracy_score(self.valid_y, predict)
        return accuracy, tot_time

    def normalize(self):
        scaler = StandardScaler()
        scaler.fit(self.train_x)
        self.train_x = scaler.transform(self.train_x)
        if self.has_valid:
            self.valid_x = scaler.transform(self.valid_x)

def get_DT_VC(data, dataset):
    print("Training Decision Tree on " + dataset)
    step = np.arange(1, 40, 2)
    new_valid_scores = np.zeros(len(step))
    index = 0
    for i in step:
        if data.has_valid:
            new_valid_scores[index], t = data.get_scores(DecisionTreeClassifier(max_depth=i))
        index = index+1
    data.set_estimator(DecisionTreeClassifier())
    train_scores, valid_scores = data.get_validation_curve_data(param_name="max_depth", cv = 6, n_jobs = -1, param_range=step)
    data.plot_validation_curve_data(dataset + "\nModel Complexity Curve (Decision Tree)", "Max depth",
                                    train_scores=train_scores, valid_scores=valid_scores, new_valid_scores=new_valid_scores, param_range=step,
                                    plot = True)
    plt.savefig("../Figures/"+"Model Complexity Curve (Decision Trees) - "+dataset+".png")
    return plt

def get_NN_VC(data, dataset):
    step = np.arange(1, 200, 5)
    print("Training NN on " + dataset)
    new_valid_scores = np.zeros(len(step))
    index = 0
    input_range = []
    data.normalize()
    for i in step:
        arr = (i, )
        input_range.append(arr)
        if data.has_valid:
            new_valid_scores[index], t = data.get_scores(MLPClassifier(hidden_layer_sizes=arr))
        index = index+1
    data.set_estimator(MLPClassifier())
    train_scores, valid_scores = data.get_validation_curve_data(param_name="hidden_layer_sizes", cv = 6, n_jobs = -1, param_range=input_range)
    data.plot_validation_curve_data(title=dataset+"\nModel Complexity Curve (Neural Network)", x_label="Hidden Layer Size",
                                    train_scores=train_scores, valid_scores=valid_scores, new_valid_scores=new_valid_scores, param_range=step,
                                    plot = True, fsize=(15, 10))
    plt.savefig("../Figures/"+"Model Complexity Curve (Neural Networks) - "+dataset+".png")
    return plt

def get_Boost_VC(data, dataset):
    print("Training AdaBoost on " + dataset)
    step = np.arange(10, 300, 10)
    new_valid_scores = np.zeros(len(step))
    index = 0
    for i in step:
        if data.has_valid:
            new_valid_scores[index], t = data.get_scores(AdaBoostClassifier(n_estimators=i))
        index = index+1
    data.set_estimator(AdaBoostClassifier())
    train_scores, valid_scores = data.get_validation_curve_data(param_name="n_estimators", cv = 6, n_jobs = -1, param_range=step)
    data.plot_validation_curve_data(title=dataset + "\nModel Complexity Curve (AdaBoost)", x_label="Number of estimators",
                                    train_scores=train_scores, valid_scores=valid_scores, new_valid_scores=new_valid_scores, param_range=step,
                                    plot = True, fsize=(15, 10))
    plt.savefig("../Figures/"+"Model Complexity Curve (AdaBoost) - "+dataset+".png")
    return plt

def get_KNN_VC(data, dataset):
    print("Training KNN on " + dataset)
    step = np.arange(1, 51, 2)
    new_valid_scores = np.zeros(len(step))
    index = 0
    for i in step:
        if data.has_valid:
            new_valid_scores[index], t = data.get_scores(KNeighborsClassifier(n_neighbors=i))
        index = index+1
    data.set_estimator(KNeighborsClassifier())
    train_scores, valid_scores = data.get_validation_curve_data(param_name="n_neighbors", cv = 6, n_jobs = -1, param_range=step)
    data.plot_validation_curve_data(title=dataset + "\nModel Complexity Curve (KNN)", x_label="k",
                                    train_scores=train_scores, valid_scores=valid_scores, new_valid_scores=new_valid_scores, param_range=step,
                                    plot = True)
    plt.savefig("../Figures/"+"Model Complexity Curve (KNN) - "+dataset+".png")
    return plt

def get_SVC_VC(data, dataset, ker):
    print("Training SVM on " + dataset)
    data.normalize()
    param_range = np.logspace(-6, 1.2, 20)
    new_valid_scores = np.zeros(len(param_range))
    index = 0
    best_gamma = 0
    for i in param_range:
        if data.has_valid:
            new_valid_scores[index], t = data.get_scores(SVC(gamma=i, kernel=ker))
        if new_valid_scores[index] > best_gamma:
            best_gamma = i
        index = index+1
    data.set_estimator(SVC(kernel=ker))
    train_scores, valid_scores = data.get_validation_curve_data(param_name="gamma", cv = 6, n_jobs = -1, param_range=param_range)
    data.plot_validation_curve_data(dataset + "\nModel Complexity Curve (SVM)", "Gamma of kernel",
                                    train_scores=train_scores, valid_scores=valid_scores, new_valid_scores=new_valid_scores, param_range=param_range,
                                    plot = False)
    plt.savefig("../Figures/"+"Model Complexity Curve (SVM) - "+dataset+".png")
    return best_gamma

def get_overview(data, dataset):
    print("Algorithms overview on " + dataset)
    accuracy = np.zeros(5)
    times = np.zeros(5)
    accuracy[0], times[0] = data.get_scores(DecisionTreeClassifier())
    accuracy[1], times[1] = data.get_scores(KNeighborsClassifier())
    accuracy[2], times[2] = data.get_scores(SVC())
    accuracy[3], times[3] = data.get_scores(AdaBoostClassifier())
    accuracy[4], times[4] = data.get_scores(MLPClassifier())
    accuracy = accuracy * 100
    name = ('Decision Tree', 'KNN', 'SVM', 'AdaBoost', 'Neural Network')
    plt.figure(figsize=(10,5))
    title = dataset + ": Overview of accuracy and performance."
    plt.title(title)
    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    opacity = 0.6
    ax1.set_ylabel('Accuracy', color = (0, 0, 1, opacity))
    ax2.set_ylabel('Running time(s)', color = (1, 0, 0, opacity))
    ax1.plot(color = (0, 0, 1, opacity))
    ax2.plot(color = (1, 0, 0, opacity))
    index = np.arange(5)+1
    bar_width = 0.25
    bar1 = ax1.bar(index-0.06, accuracy, bar_width, alpha = opacity, color = 'b', label = 'Accuracy')
    bar2 = ax2.bar(index + bar_width+0.06, times, bar_width, alpha = opacity, color = 'r', label = 'Running time')
    for rect in bar1:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., height,
                '%.1f%%' % height,
                ha='center', va='bottom')
    for rect in bar2:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height,
                '%.2fs' % height,
                ha='center', va='bottom')
    plt.xticks(index + bar_width/2, name)
    plt.tight_layout()
    plt.savefig("../Figures/"+dataset+" overview.png")
    return plt

def get_SVM_compare(datalist, dataset):
    print("SVM comparison on " + dataset)
    accuracy1 = np.zeros(4)
    times1 = np.zeros(4)
    train_x, train_y, valid_x, valid_y = datalist[2], datalist[3], datalist[4], datalist[5]
    data = Model_Validation(train_x, train_y, SVC(), len(train_x))
    data.normalize()
    data.set_valid_data(valid_x, valid_y)
    accuracy1[0], times1[0] = data.get_scores(SVC())
    print('rbf')
    accuracy1[1], times1[1] = data.get_scores(SVC(kernel='linear'))
    print('linear')
    accuracy1[2], times1[2] = data.get_scores(SVC(kernel='poly'))
    print('poly')
    accuracy1[3], times1[3] = data.get_scores(SVC(kernel='sigmoid'))
    print('sigmoid')
    accuracy1 = accuracy1 * 100
    name = ('rbf', 'linear', 'poly', 'sigmoid')
    plt.figure()
    title = "\nSVM kernels comparison on " + dataset
    plt.title(title)
    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    opacity = 0.6
    ax1.set_ylabel('Accuracy', color = (0, 0, 1, opacity))
    ax2.set_ylabel('Running time(s)', color = (1, 0, 0, opacity))
    ax1.plot(color = (0, 0, 1, opacity))
    ax2.plot(color = (1, 0, 0, opacity))
    index = np.arange(4)+1
    bar_width = 0.25
    bar11 = ax1.bar(index, accuracy1, bar_width, alpha = opacity, color = 'b')
    bar21 = ax2.bar(index + bar_width, times1, bar_width, alpha = opacity, color = 'r')
    for rect in bar11:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., height,
                '%.1f%%' % height,
                ha='center', va='bottom')
    for rect in bar21:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height,
                '%.2fs' % height,
                ha='center', va='bottom')
    plt.xticks(index + bar_width/2, name)
    plt.legend((bar11, bar21), ('Accuracy', 'Running time'))
    plt.tight_layout()
    plt.savefig("../Figures/"+dataset + " - SVM kernels comparison.png")
    return plt

def get_ROC_curve(data, dataset):
    print("ROC for " + dataset)
    train_x, train_y, valid_x, valid_y = data[2], data[3], data[4], data[5]
    colors = ('r', 'g', 'b', 'c', 'darkorange')
    estimators = (DecisionTreeClassifier(),
                  KNeighborsClassifier(),
                  SVC(gamma=0.001, probability=True),
                  AdaBoostClassifier(),
                  MLPClassifier())
    names = ('Decision tree', 'KNN', 'SVM', 'AdaBoost', 'Neural network')
    plt.figure()
    lw = 2
    for estimator, color, name in zip(estimators, colors, names):
        classifier = estimator.fit(train_x, train_y)
        y_score = classifier.predict_proba(valid_x)
        fpr, tpr, _ = roc_curve(valid_y, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color,
                 lw=lw, label=(name + (' (Area = %0.2f)' % roc_auc)) )
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('\nROC curves for '+dataset)
    plt.legend(loc="lower right")
    plt.savefig("../Figures/"+'ROC curves for '+dataset+".png")