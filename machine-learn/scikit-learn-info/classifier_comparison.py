import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mpl_toolkits.mplot3d import Axes3D

datasets = [make_circles(n_samples=200,  noise=0.05, factor=0.5),
            make_classification(n_samples=200,  n_features=2,n_informative=2,   n_redundant=0,     n_repeated=0,  n_classes=2,random_state=11),
            make_moons(n_samples=200, noise=0.05,)]

classifiers = [KNeighborsClassifier(),#assegna la classe del punto in base alla maggioranza dei k punti più vicini.
               SVC(kernel="linear"),#trova una linea (o iperpiano) che separa le classi massimizzando il margine.
               SVC(gamma=2),#SVM con kernel RBF (radiale) → trasforma lo spazio per separare dati non lineari.
               GaussianProcessClassifier(),#modello probabilistico, usa un kernel per stimare la funzione decisionale.
               DecisionTreeClassifier(),#divide lo spazio iterativamente con soglie sulle feature.
               RandomForestClassifier(),#insieme di alberi decisionali con bagging → media dei risultati.
               MLPClassifier(),#rete con layer fully connected, apprende pattern non lineari.
               AdaBoostClassifier(),#combina molti “weak learners” (di solito alberi piccoli) in sequenza.
               GaussianNB(),#probabilistico, assume feature indipendenti condizionatamente alla classe.
               QuadraticDiscriminantAnalysis()#assume distribuzioni gaussiane per classe con covarianze diverse → confini quadratici.
]

names = [
    "Nearest Neighbors", 
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

i = 1
colors = ['red', 'blue', 'green']
markers = ['o', '^', 's']
plt.figure(figsize=(10, 10))
for ds_cnt, ds in enumerate(datasets):
    X, y = ds

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=67)
    third_train = np.random.rand(len(X_train))
    third_test = np.random.rand(len(X_test))
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i, projection="3d")
    if ds_cnt == 0:
        ax.set_title("DATI DI INPUT")
    ax.scatter(X_train[:, 0], X_train[:, 1], third_train, c=y_train, cmap='Set1', edgecolors="k")
    ax.scatter(X_test[:, 0], X_test[:, 1],third_test, c=y_test, cmap='Set1')
    i += 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    for name, clf in zip(names, classifiers):

        ax = plt.subplot(len(datasets), len(classifiers)+1, i, )
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(clf, X, alpha=0.8, ax=ax, eps=0.5)
        ax.scatter(X_train[:, 0], X_train[:, 1],  c=y_train, cmap='plasma', edgecolors="k")
        ax.scatter(X_test[:,0],X_test[:, 1],  c=y_test, cmap="plasma", edgecolors="k")
        if ds_cnt == 0:
            ax.set_title(name)
        i += 1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        
plt.tight_layout()
plt.show()


