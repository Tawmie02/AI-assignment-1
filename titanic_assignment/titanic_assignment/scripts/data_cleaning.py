import pandas as pd

def clean_data(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    if 'Cabin' in df.columns:
        df.drop(columns=['Cabin'], inplace=True)
    return df