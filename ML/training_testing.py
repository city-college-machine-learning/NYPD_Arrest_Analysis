# libraries

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from ML import model_constants

# tuning function


def tune_model(model_name, model, X_train, y_train, tuning_parameters):
    """
    :param model_name: model Abbreviation such as LR for LogisticRegression
    :param model: name of the model
    :param X_train: training features
    :param y_train: target column
    :param tuning_parameters: tuning parameters
    :return: return a cross validation
    """
    cross_vals = GridSearchCV(estimator=model, param_grid=tuning_parameters[model_name], cv=5)
    cross_vals.fit(X_train, y_train)
    return (cross_vals)

# function to train the model


def train_a_model(model, X_train, y_train, cross_vals):
    """
    :param model: a model with fixed parameter
    :param X_train: features
    :param y_train: target columns to predict
    :param cross_vals: a cross validation model model
    :return: return a model trained
    """
    model.set_params(**cross_vals.best_params_)
    model.fit(X_train, y_train)
    return (model)


# split the data into train and test


def data_split(df, target_col, test_plit):
    """
    :param df: an array or dataframe of features data
    :param target_col: 1D array or column of the target labels
    :param test_plit: percent of data to keep for test set
    :return: training and testing set
    """
    X_train, X_test, y_train, y_test = train_test_split(df, target_col, test_size=test_plit)
    return X_train, X_test, y_train, y_test

# function to uses a model to predict


def model_predict(df, model):
    """
    :param df: data to predict
    :param model:
    :return:
    """
    y_predict = model.predict(df)
    return y_predict

# function to check the performance of the model


def get_model_perform(y_preds, y_test, labels):
    """
    :param y_preds: predicted labels
    :param y_test: true labels
    :param labels: labels
    :return: precision, recall, f1
    """

    precision = precision_score(y_test, y_preds, average='weighted')
    recall = recall_score(y_test, y_preds, average='weighted')
    f1 = f1_score(y_test, y_preds, average='weighted')
    results = [precision, recall, f1]

    return results

# function to train and test the model


def train_and_test(df, target_col, model_name, pct_test):
    """
    :param df: features data
    :param target_col: target column
    :param model_name: model name
    :param pct_test: percent to keep for the testing
    :return: return the best model and its performance
    """
    model_abr = ""
    for model_info in model_constants.model_names:
        if model_info["label"] == model_name
            model_abr = model_info["value"]
            break

    model = model_constants.models.get(model_abr)
    X_train, X_test, y_train, y_test = data_split(df, target_col, test_plit=pct_test)
    cross_val = tune_model(model_abr, model, X_train, y_train, model_constants.tuning_params)
    model = train_a_model(model, X_train, y_train, cross_val)
    model_preds = model_predict(X_test, model)
    classes = list(model.classes_)
    model_results = get_model_perform(model_preds, y_test, classes)

    return model, model_results


