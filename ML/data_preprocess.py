from sklearn.preprocessing import LabelEncoder
import pandas as pd
lab_enc = LabelEncoder()


def data_preprocess1():
    data = pd.read_csv('/Users/jja/GitHub/NYPD_Arrest_Data__Year_to_Date_.csv')

    data.drop(['ARREST_DATE','ARREST_KEY', 'PD_CD', 'PD_DESC', 'KY_CD', 'LAW_CODE',\
               'OFNS_DESC', 'X_COORD_CD','Y_COORD_CD','Latitude','Longitude'], axis=1, inplace=True)
    data.dropna(inplace=True)
    crimes = {'M': 'Misdemeanor', 'F': 'Felony', 'V': 'Violation', 'I': 'Infraction'}
    data['LAW_CAT_CD'] = data['LAW_CAT_CD'].replace(crimes)

    data = data[data.LAW_CAT_CD != 'I']
    data = data[data.LAW_CAT_CD != 'V']
    target_removed = data.loc[:, data.columns != 'LAW_CAT_CD']

    X = pd.get_dummies(target_removed, drop_first=True)

    y = data['LAW_CAT_CD'].values

    return X, y
