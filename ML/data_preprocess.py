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


def jiffar_preprocess():

    data = pd.read_csv('C:/Users/daoud/NYPD_Arrest_Data__Year_to_Date_.csv')
    # change borough columns to make it clear
    borough = {'K': 'Brooklyn', 'M': 'Manhattan', 'B': 'Bronx', 'Q': "Queens", 'S': 'Staten Island'}
    # change the crime labels to make it clear
    crimes = {'M': 'Misdemeanor', 'F': 'Felony', 'V': 'Violation', 'I': 'Infraction'}
    # change the gender labels
    gender = {'M': 'Male', 'F': 'Female'}
    # Replace borough label
    data['ARREST_BORO'] = data['ARREST_BORO'].replace(borough)
    # Replace Crime label
    data['LAW_CAT_CD'] = data['LAW_CAT_CD'].replace(crimes)
    # Replace gender label
    data['PERP_SEX'] = data['PERP_SEX'].replace(gender)
    # Change the column names
    data.rename(columns={'PD_DESC': 'OFFENSE_DESC_1', 'OFNS_DESC': 'OFFENSE_DESC_2', 'LAW_CAT_CD': 'LEVEL_OF_OFFENSE',
                         'PD_CD': 'INTERNAL_CODE_1', 'KY_CD': 'INTERNAL_CODE_2'}, inplace=True)
    coordinates = data[['X_COORD_CD', 'Y_COORD_CD']]
    data.drop(['X_COORD_CD', 'Y_COORD_CD'], axis=1, inplace=True)
    data = data.astype('object')

    data.drop(['OFFENSE_DESC_1', 'OFFENSE_DESC_2', 'INTERNAL_CODE_1', 'INTERNAL_CODE_2', 'LAW_CODE'], axis=1,
              inplace=True)
    target_removed = data.loc[:, data.columns != 'LEVEL_OF_OFFENSE']
    X = pd.concat([coordinates, pd.get_dummies(target_removed, drop_first=True)], axis=1)
    target = data.loc[:, 'LEVEL_OF_OFFENSE']

    return X, target

