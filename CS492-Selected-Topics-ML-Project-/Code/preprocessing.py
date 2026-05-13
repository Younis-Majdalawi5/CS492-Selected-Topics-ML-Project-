import pandas as pd
from sklearn.preprocessing import StandardScaler

def loadAndCleanData(filePath):

    df = pd.read_csv(filePath, delimiter=';')

    df['Date'] = pd.to_datetime(df['Date'])

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    df = df.drop(columns=['Date'])

    df.dropna(inplace=True)

    return df

def splitFutureData(df):
    futureData = df.tail(15)

    modelData = df.iloc[0:-15]

    return modelData, futureData

def applyScaling(X_train, X_test, X_future):
    scaler = StandardScaler()

    X_trainScaled = scaler.fit_transform(X_train)

    X_testScaled = scaler.transform(X_test)
    X_futureScaled = scaler.transform(X_future)

    return X_trainScaled, X_testScaled, X_futureScaled, scaler
