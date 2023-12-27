from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    max_it = 1000
    lr = LogisticRegression(random_state=0, max_iter=max_it)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return y_test, y_pred