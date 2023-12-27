import pandas as pd
import seaborn as sns

def explore_data(df):
    print(df.shape)
    print(df.head())
    print(df.info())
    print(df.columns)
    print(df.isna().sum())
    df["Label"].value_counts()
    sns.countplot(x=df["Label"], data=df)
    df['Label'] = df['Label'].replace({'b': 0, 's': 1})

def prepare_data(df):
    drop = ['EventId', 'Weight']
    df = df.drop(columns=drop)
    features = df.columns[1:-1]
    X = df[features]
    y = df['Label']
    return X, y