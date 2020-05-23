import pandas as pd
    
class Clean(object):
    def read(self, data):
        new_data = pd.read_csv(data)
        return new_data
    
    
    def fig_generate_month(self, data):
        new_data = pd.read_csv(data)
        new_data.drop(['ARREST_KEY', 'PD_CD', 'KY_CD', 'LAW_CODE', ], axis=1, inplace=True)
        new_data.dropna(inplace=True)
        new_data.rename(columns={'PD_DESC':'offenses', 'LAW_CAT_CD':'offense_type'}, inplace=True)
        return new_data
        
        