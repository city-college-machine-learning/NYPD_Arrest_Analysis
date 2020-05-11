
from ML import training_testing, data_preprocess


df, y_column = data_preprocess.data_preprocess1()

model, model_perf = training_testing.train_and_test(df, y_column, 'Support Vector Machine', 0.3)


