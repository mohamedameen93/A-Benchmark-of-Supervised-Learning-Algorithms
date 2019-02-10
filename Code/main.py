#################################################
# A Benchmark of Supervised Learning Algorithms #
#################################################

# Importing required Python libraries
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from helperfunctions import *
from learningmodels import *

####################################################################################################################################

if __name__ == '__main__':
    # Reading Credit Card Default dataset
    print("\nReading Credit Card Default dataset\n")
    credit_dataset = list()
    credit_dataset = ReadDataSet_Credit()

    # Dataset analysis
    print("Size of Training set (Credit):", len(credit_dataset[2]))
    print("Size of Validation set (Credit):", len(credit_dataset[4]))
    print("Size of Test set (Credit):", len(credit_dataset[6]))

    # Correlation Matrix
    plot_Correlation_Matrix(credit_dataset[0], 'Credit_CorrMat.png')
 
    # Reading MNIST Database of handwritten digits
    MNIST_dataset = list()
    MNIST_dataset = ReadDataSet_MNIST()
    print("\nReading MNIST Database of handwritten digits\n")

    # Dataset analysis
    print("Size of Training set (MNIST):", len(MNIST_dataset[2]))
    print("Size of Validation set (MNIST):", len(MNIST_dataset[4]))
    print("Size of Test set (MNIST):", len(MNIST_dataset[6]))

    # Dataset histogram
    histogram_plot(MNIST_dataset[3])

    ################################################################################################################################

    #################
    # Decision Tree #
    #################
    print("\nDecision Trees\n")
    # Credit Card Default dataset
    model_credit_DT = Model_Validation(credit_dataset[2], credit_dataset[3], DecisionTreeClassifier(), len(credit_dataset[2]))
    model_credit_DT.set_valid_data(credit_dataset[4], credit_dataset[5])
    get_DT_VC(model_credit_DT, credit_dataset[8])
    model_credit_DT.plot_learning_curve(title = "Learning Curve (Decision Tree) - " + credit_dataset[8], cv = 5, n_jobs=-1)
    # Decision Tree with the best max_depth applied to Credit Card Default dataset
    classifier_credit_DT = DecisionTreeClassifier(max_depth=5)
    classifier_credit_DT.fit(credit_dataset[2], credit_dataset[3])
    # Predicting new results
    y_pred = classifier_credit_DT.predict(credit_dataset[6])
    acc = "Accuracy = %.2f" % ((accuracy_score(credit_dataset[7], y_pred)) * 100)+"%"
    print(acc)
    # Plot cloassifier graph
    try:
        plot_classifier_graph(classifier_credit_DT, credit_dataset[8]+" Decision Tree graph")
    except:
        print("Error: Graphviz is not installed!\n")
        print("Classieifer graph cannot be produced.\n")

    # MNIST Database of handwritten digits
    model_MNIST = Model_Validation(MNIST_dataset[2], MNIST_dataset[3], DecisionTreeClassifier(), len(MNIST_dataset[2]))
    model_MNIST.set_valid_data(MNIST_dataset[4], MNIST_dataset[5])
    get_DT_VC(model_MNIST, MNIST_dataset[8])
    model_MNIST.plot_learning_curve(title = "Learning Curve (Decision Tree) - " + MNIST_dataset[8], cv = 5, n_jobs=-1)
    # Decision Tree with the best max_depth applied to MNIST Database of handwritten digits
    classifier_MNIST_DT = DecisionTreeClassifier(max_depth=9)
    classifier_MNIST_DT.fit(MNIST_dataset[2], MNIST_dataset[3])
    # Predicting new results
    y_pred = classifier_MNIST_DT.predict(MNIST_dataset[6])
    acc = "Accuracy = %.2f" % ((accuracy_score(MNIST_dataset[7], y_pred)) * 100)+"%"
    print(acc)
    # Plot cloassifier graph
    try:
        plot_classifier_graph(classifier_MNIST_DT, MNIST_dataset[8]+" Decision Tree graph")
    except:
        print("Error: Graphviz is not installed!\n")
        print("Classieifer graph cannot be produced.\n")

    combine_images(['../Figures/Learning Curve (Decision Tree) - Credit Card Default dataset.png',
                    '../Figures/Learning Curve (Decision Tree) - MNIST Database of handwritten digits.png'],
                    '../Figures/Learning Curve (Decision Tree).png')
    combine_images(['../Figures/Model Complexity Curve (Decision Trees) - Credit Card Default dataset.png',
                    '../Figures/Model Complexity Curve (Decision Trees) - MNIST Database of handwritten digits.png'],
                    '../Figures/Model Complexity Curve (Decision Trees).png')

    ################################################################################################################################

    ###################
    # Neural Networks #
    ###################
    print("\nNeural Networks\n")
    # Credit Card Default dataset
    model_credit_NN = Model_Validation(credit_dataset[2], credit_dataset[3], MLPClassifier(), len(credit_dataset[2]))
    model_credit_NN.set_valid_data(credit_dataset[4], credit_dataset[5])
    get_NN_VC(model_credit_NN, credit_dataset[8])
    model_credit_NN.plot_learning_curve(title = "Learning Curve (Neural Networks) - " + credit_dataset[8], cv = 5, n_jobs=-1)
    # Neural Network with the best hidden_layer_sizes applied to Credit Card Default dataset
    classifier_credit_NN = MLPClassifier(hidden_layer_sizes=(10,))
    classifier_credit_NN.fit(credit_dataset[2], credit_dataset[3])
    # Predicting new results
    y_pred = classifier_credit_NN.predict(credit_dataset[6])
    acc = "Accuracy = %.2f" % ((accuracy_score(credit_dataset[7], y_pred)) * 100)+"%"
    print(acc)

    # MNIST Database of handwritten digits
    model_MNIST_NN = Model_Validation(MNIST_dataset[2], MNIST_dataset[3], MLPClassifier(), len(MNIST_dataset[2]))
    model_MNIST_NN.set_valid_data(MNIST_dataset[4], MNIST_dataset[5])
    get_NN_VC(model_MNIST_NN, MNIST_dataset[8])
    model_MNIST_NN.plot_learning_curve(title = "Learning Curve (Neural Networks) - " + MNIST_dataset[8], cv = 5, n_jobs=-1)
    # Neural Network with the best hidden_layer_sizes applied to MNIST Database of handwritten digits
    classifier_MNIST_NN = MLPClassifier(hidden_layer_sizes=(21,))
    classifier_MNIST_NN.fit(MNIST_dataset[2], MNIST_dataset[3])
    # Predicting new results
    y_pred = classifier_MNIST_NN.predict(MNIST_dataset[6])
    acc = "Accuracy = %.2f" % ((accuracy_score(MNIST_dataset[7], y_pred)) * 100)+"%"
    print(acc)

    combine_images(['../Figures/Learning Curve (Neural Networks) - Credit Card Default dataset.png',
                    '../Figures/Learning Curve (Neural Networks) - MNIST Database of handwritten digits.png'],
                    '../Figures/Learning Curve (Neural Network).png')
    combine_images(['../Figures/Model Complexity Curve (Neural Networks) - Credit Card Default dataset.png',
                    '../Figures/Model Complexity Curve (Neural Networks) - MNIST Database of handwritten digits.png'],
                    '../Figures/Model Complexity Curve (Neural Network).png')

    ################################################################################################################################

    ############
    # Boosting #
    ############
    print("\nBoosting\n")
    # Credit Card Default dataset
    model_credit_AdaB = Model_Validation(credit_dataset[2], credit_dataset[3], AdaBoostClassifier(), len(credit_dataset[2]))
    model_credit_AdaB.set_valid_data(credit_dataset[4], credit_dataset[5])
    get_Boost_VC(model_credit_AdaB, credit_dataset[8])
    model_credit_AdaB.plot_learning_curve(title = "Learning Curve (AdaBoost) - " + credit_dataset[8], cv = 5, n_jobs=-1)
    # AdaBoost with the best n_estimators applied to Credit Card Default dataset
    classifier_credit_AdaB = AdaBoostClassifier(n_estimators=20)
    classifier_credit_AdaB.fit(credit_dataset[2], credit_dataset[3])
    # Predicting new results
    y_pred = classifier_credit_AdaB.predict(credit_dataset[6])
    acc = "Accuracy = %.2f" % ((accuracy_score(credit_dataset[7], y_pred)) * 100)+"%"
    print(acc)

    # MNIST Database of handwritten digits
    model_MNIST_AdaB = Model_Validation(MNIST_dataset[2], MNIST_dataset[3], AdaBoostClassifier(), len(MNIST_dataset[2]))
    model_MNIST_AdaB.set_valid_data(MNIST_dataset[4], MNIST_dataset[5])
    get_Boost_VC(model_MNIST_AdaB, MNIST_dataset[8])
    model_MNIST_AdaB.plot_learning_curve(title = "Learning Curve (AdaBoost) - " + MNIST_dataset[8], cv = 5, n_jobs=-1)
    # AdaBoost with the best n_estimators applied to MNIST Database of handwritten digits
    classifier_MNIST_AdaB = AdaBoostClassifier(n_estimators=40)
    classifier_MNIST_AdaB.fit(MNIST_dataset[2], MNIST_dataset[3])
    # Predicting new results
    y_pred = classifier_MNIST_AdaB.predict(MNIST_dataset[6])
    acc = "Accuracy = %.2f" % ((accuracy_score(MNIST_dataset[7], y_pred)) * 100)+"%"
    print(acc)

    combine_images(['../Figures/Learning Curve (AdaBoost) - Credit Card Default dataset.png',
                    '../Figures/Learning Curve (AdaBoost) - MNIST Database of handwritten digits.png'],
                    '../Figures/Learning Curve (AdaBoost).png')
    combine_images(['../Figures/Model Complexity Curve (AdaBoost) - Credit Card Default dataset.png',
                    '../Figures/Model Complexity Curve (AdaBoost) - MNIST Database of handwritten digits.png'],
                    '../Figures/Model Complexity Curve (AdaBoost).png')

    ################################################################################################################################

    ###########################
    # Support Vector Machines #
    ###########################
    print("\nSupport Vector Machines\n")
    # SVC Kernels Comparison
    get_SVM_compare(credit_dataset, 'Credit Card Default dataset')
    get_SVM_compare(MNIST_dataset, 'MNIST Database of handwritten digits')

    combine_images(['../Figures/Credit Card Default dataset - SVM kernels comparison.png',
                    '../Figures/MNIST Database of handwritten digits - SVM kernels comparison.png'],
                    '../Figures/SVM kernels comparison.png')

    # Credit Card Default dataset
    model_credit_SVM = Model_Validation(credit_dataset[2], credit_dataset[3], SVC(), len(credit_dataset[2]))
    model_credit_SVM.set_valid_data(credit_dataset[4], credit_dataset[5])
    bestgamma_credit_SVM = get_SVC_VC(model_credit_SVM, credit_dataset[8], ker='rbf')
    model_credit_SVM.plot_learning_curve(title = "Learning Curve (SVM) - " + credit_dataset[8], cv = 5, n_jobs=-1)
    # SVC with the best gamma applied to Credit Card Default dataset
    classifier_credit_SVM = SVC(gamma=bestgamma_credit_SVM, kernel='rbf')
    classifier_credit_SVM.fit(credit_dataset[2], credit_dataset[3])
    # Predicting new results
    y_pred = classifier_credit_SVM.predict(credit_dataset[6])
    acc = "Accuracy = %.2f" % ((accuracy_score(credit_dataset[7], y_pred)) * 100)+"%"
    print(acc)

    # MNIST Database of handwritten digits
    model_MNIST_SVM = Model_Validation(MNIST_dataset[2], MNIST_dataset[3], SVC(), len(MNIST_dataset[2]))
    model_MNIST_SVM.set_valid_data(MNIST_dataset[4], MNIST_dataset[5])
    model_MNIST_SVM.plot_learning_curve(title = "Learning Curve (SVM) - " + MNIST_dataset[8], cv = 5, n_jobs=-1)
    # SVC with the best gamma applied to MNIST Database of handwritten digits
    classifier_MNIST_SVM = SVC(kernel='linear')
    classifier_MNIST_SVM.fit(MNIST_dataset[2], MNIST_dataset[3])
    # Predicting new results
    y_pred = classifier_MNIST_SVM.predict(MNIST_dataset[6])
    acc = "Accuracy = %.2f" % ((accuracy_score(MNIST_dataset[7], y_pred)) * 100)+"%"
    print(acc)

    combine_images(['../Figures/Model Complexity Curve (SVM) - Credit Card Default dataset.png',
                    '../Figures/Model Complexity Curve (SVM) - MNIST Database of handwritten digits.png'],
                    '../Figures/Model Complexity Curve (SVM).png')

    ################################################################################################################################

    #######################
    # K-Nearest Neighbors #
    #######################
    print("\nK-Nearest Neighbors\n")
    # Credit Card Default dataset
    model_credit_KNN = Model_Validation(credit_dataset[2], credit_dataset[3], KNeighborsClassifier(), len(credit_dataset[2]))
    model_credit_KNN.set_valid_data(credit_dataset[4], credit_dataset[5])
    get_KNN_VC(model_credit_KNN, credit_dataset[8])
    model_credit_KNN.plot_learning_curve(title = "Learning Curve (KNN) - " + credit_dataset[8], cv = 5, n_jobs=-1)
    # KNN with the best n_neighbors applied to Credit Card Default dataset
    classifier_credit_KNN = KNeighborsClassifier(n_neighbors=37)
    classifier_credit_KNN.fit(credit_dataset[2], credit_dataset[3])
    # Predicting new results
    y_pred = classifier_credit_KNN.predict(credit_dataset[6])
    acc = "Accuracy = %.2f" % ((accuracy_score(credit_dataset[7], y_pred)) * 100)+"%"
    print(acc)

    # MNIST Database of handwritten digits
    model_MNIST_KNN = Model_Validation(MNIST_dataset[2], MNIST_dataset[3], KNeighborsClassifier(), len(MNIST_dataset[2]))
    model_MNIST_KNN.set_valid_data(MNIST_dataset[4], MNIST_dataset[5])
    get_KNN_VC(model_MNIST_KNN, MNIST_dataset[8])
    model_MNIST_KNN.plot_learning_curve(title = "Learning Curve (KNN) - " + MNIST_dataset[8], cv = 5, n_jobs=-1)
    # KNN with the best n_neighbors applied to Credit Card Default dataset
    classifier_MNIST_KNN = KNeighborsClassifier(n_neighbors=5)
    classifier_MNIST_KNN.fit(MNIST_dataset[2], MNIST_dataset[3])
    # Predicting new results
    y_pred = classifier_MNIST_KNN.predict(MNIST_dataset[6])
    acc = "Accuracy = %.2f" % ((accuracy_score(MNIST_dataset[7], y_pred)) * 100)+"%"
    print(acc)

    combine_images(['../Figures/Learning Curve (KNN) - Credit Card Default dataset.png',
                    '../Figures/Learning Curve (KNN) - MNIST Database of handwritten digits.png'],
                    '../Figures/Learning Curve (KNN).png')
    combine_images(['../Figures/Model Complexity Curve (KNN) - Credit Card Default dataset.png',
                    '../Figures/Model Complexity Curve (KNN) - MNIST Database of handwritten digits.png'],
                    '../Figures/Model Complexity Curve (KNN).png')

    ################################################################################################################################

    ############
    # Overview #
    ############
    print("\nOverview\n")
    # Credit Card Default dataset
    model_credit_Overview = Model_Validation(credit_dataset[2], credit_dataset[3], DecisionTreeClassifier(), len(credit_dataset[2]))
    model_credit_Overview.set_valid_data(credit_dataset[4], credit_dataset[5])
    get_overview(model_credit_Overview, 'Credit Card Default dataset')

    # MNIST Database of handwritten digits
    model_MNIST_Overview = Model_Validation(MNIST_dataset[2], MNIST_dataset[3], DecisionTreeClassifier(), len(MNIST_dataset[2]))
    model_MNIST_Overview.set_valid_data(MNIST_dataset[4], MNIST_dataset[5])
    get_overview(model_MNIST_Overview, 'MNIST Database of handwritten digits')

    combine_images(['../Figures/Credit Card Default dataset overview.png',
                    '../Figures/MNIST Database of handwritten digits overview.png'],
                    '../Figures/Datasets overview.png')

    # Receiver Operating Characteristic (ROC) Curve
    get_ROC_curve(credit_dataset, 'Credit Card Default dataset')

    print("\nProcessing completed successfuly!\n")
