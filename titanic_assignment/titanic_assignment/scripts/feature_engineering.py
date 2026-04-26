import pandas as pd
import numpy as np

def apply_engineering(df):
    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Title extraction
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # ... (Include your Title replacement logic here) ...
    
    # Fare transformation
    df['Fare_Log'] = np.log1p(df['Fare'])
    return df