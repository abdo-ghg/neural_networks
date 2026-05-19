import numpy as np

def preprocessing(df):
    df['Species'] = df['Species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
    df['OriginLocation'] = df['OriginLocation'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})

    df = df.dropna()

    X = df.drop('Species', axis=1).values
    y = df['Species'].values

    return X, y


def split(X, y):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]

        X_cls = X[idx]
        y_cls = y[idx]

        X_train.append(X_cls[:30])
        y_train.append(y_cls[:30])

        X_test.append(X_cls[30:50])
        y_test.append(y_cls[30:50])

    return (np.vstack(X_train), np.hstack(y_train), np.vstack(X_test), np.hstack(y_test),)

