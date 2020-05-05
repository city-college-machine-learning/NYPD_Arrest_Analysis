# here import the model to be called
from sklearn.linear_model import LogisticRegression
# model types
models = {
    'LR': LogisticRegression(solver='lbfgs', multi_class='multinomial', penalty='l1')
}

model_names = [
    {'value': 'LR', 'label': 'Logistic Regression'}
]

# tuning_param

tuning_params = {
    'LR': {'penalty': ['l2'], 'C': [0.0001, 0.001, 0.01]}
}

