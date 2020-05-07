from sklearn.preprocessing import LabelEncoder
import pandas as pd
lab_enc = LabelEncoder()


def data_preprocess1():
    data = pd.read_csv('./NYPD_Arrest_Data__Year_to_Date_.csv')
    data.drop(['ARREST_KEY', 'PD_CD', 'KY_CD', 'LAW_CODE', ], axis=1, inplace=True)
    data.dropna(inplace=True)
    data.rename(columns={'PD_DESC':'offenses', 'LAW_CAT_CD':'offense_type'}, inplace=True)

    X = data[['offenses', 'offense_type', 'ARREST_BORO', 'AGE_GROUP']].values

    X[:,0] = lab_enc.fit_transform(X[:,0])
    X[:, 1] = lab_enc.fit_transform(X[:, 1])
    X[:, 2] = lab_enc.fit_transform(X[:, 2])
    X[:, 3] = lab_enc.fit_transform(X[:, 3])

    y = lab_enc.fit_transform(data[['PERP_RACE']].values)

    return X, y
