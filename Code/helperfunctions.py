# Importing required Python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
import PIL
from mlxtend.data import loadlocal_mnist

def ReadDataSet_Credit():
    dataset = pd.read_csv("../Datasets/credit_card_clients.csv")
    dataset_name = 'Credit Card Default dataset'
    x_data = dataset.drop('default_payment', axis=1)
    y_data = dataset['default_payment']
    x_train, x_validate, y_train, y_validate = train_test_split(x_data, y_data, test_size=0.4, random_state=1)
    x_validate, x_test, y_validate, y_test= train_test_split(x_validate, y_validate, test_size=0.5, random_state=1)
    return x_data, y_data, x_train, y_train, x_validate, y_validate, x_test, y_test, dataset_name

def Convert_MNIST():
    X, y = loadlocal_mnist(images_path='../Datasets/train-images.idx3-ubyte', 
                           labels_path='../Datasets/train-labels.idx1-ubyte')
    np.savetxt(fname='../Datasets/MNIST_images.csv', X=X, delimiter=',', fmt='%d')
    np.savetxt(fname='../Datasets/MNIST_labels.csv', X=y, delimiter=',', fmt='%d')
    X_test, y_test = loadlocal_mnist(images_path='../Datasets/t10k-images.idx3-ubyte', 
                                     labels_path='../Datasets/t10k-labels.idx1-ubyte')
    np.savetxt(fname='../Datasets/MNIST_testimages.csv', X=X_test, delimiter=',', fmt='%d')
    np.savetxt(fname='../Datasets/MNIST_testlabels.csv', X=y_test, delimiter=',', fmt='%d')

def ReadDataSet_MNIST():
    training_images = pd.read_csv('../Datasets/MNIST_images.csv')
    training_labels = pd.read_csv('../Datasets/MNIST_labels.csv')
    test_images = pd.read_csv('../Datasets/MNIST_testimages.csv')
    test_labels = pd.read_csv('../Datasets/MNIST_testlabels.csv')
    dataset_name = 'MNIST Database of handwritten digits'
    x_data, dummy_x, y_data, dummy_y = train_test_split(training_images, training_labels, test_size=0.8, random_state=1)
    x_test, dummytest_x, y_test, dummytest_y = train_test_split(test_images, test_labels, test_size=0.5, random_state=1)
    x_train, x_validate, y_train, y_validate = train_test_split(x_data, y_data, test_size=0.3, random_state=1)
    return x_data, y_data, x_train, y_train, x_validate, y_validate, x_test, y_test, dataset_name

def plot_Correlation_Matrix(data, name, fsize=None):
    f, ax = plt.subplots(figsize=fsize)
    sns.heatmap(data.corr(), 
                xticklabels=data.columns.values,
                yticklabels=data.columns.values,
                cmap="Reds")
    f.savefig("../Figures/"+name)

def histogram_plot(dataset):
    hist, _ = np.histogram(dataset, bins=10)
    _, ax = plt.subplots()
    ax.bar(range(10), hist, width=0.8, align='center')
    ax.set(xticks=range(10), xlim=[-1, 10])
    plt.grid(axis='y')
    plt.xlabel("Classes")
    plt.ylabel("Number of exampels")
    plt.savefig("../Figures/MNIST Database Histogram.png")

def plot_classifier_graph(clf, title):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("../Figures/"+title+".png")

def combine_images(list_im, outfile):
    imgs      = [PIL.Image.open(i) for i in list_im]
    min_shape = sorted([(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save("../Figures/"+outfile)
