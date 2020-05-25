import training_testing, data_preprocess

df, y_column = data_preprocess.data_preprocess1()

model, model_perf = training_testing.train_and_test(df, y_column, 'Logistic Regression', 0.3)

print(model_perf)
