# here import the model to be called
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# model types
models = {
    'LR': LogisticRegression(solver='lbfgs', multi_class='multinomial', penalty='l1'),
    'RF': RandomForestClassifier(n_estimators=40),
    'SVM': LinearSVC(dual=False)
}

model_names = [
    {'value': 'LR', 'label': 'Logistic Regression'},
    {'value': 'RF', 'label': 'Random Forest'},
    {'value': 'SVM', 'label': 'Support Vector Machine'}
]

# tuning_param

tuning_params = {
    'LR': {'penalty': ['l2','l1'], 'C': [0.0001, 0.001, 0.01,.1]},
    'RF': {'n_estimators': [1, 20, 30, 40, 50]},
    'SVM': {'C': [0.001, 0.01, 0.1, 1]}
}

