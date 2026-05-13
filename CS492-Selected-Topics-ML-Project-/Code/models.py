from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def splitData(df):

    X = df.drop(columns=['Close'])

    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    return X_train, X_test, y_train, y_test

def getRegressionModels():

    models = {}

    models["Linear Regression"] = LinearRegression()

    models["Random Forest"] = RandomForestRegressor(n_estimators=100, random_state=42)

    models["SVR"] = SVR(kernel='rbf')

    return models