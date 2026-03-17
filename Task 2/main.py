import numpy as np
from micrograd.engine import Tensor
from model import MLP
import micrograd.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix
import seaborn as sns

EPOCHS = 2000
learning_rate = 0.01
bias = True

features = 5
classes = 3
tanh_hidden = True

num_hidden_neurons = 4 * 16
num_of_hidden_layers = 2

losses = []

np.random.seed(37) # for reproducibility

# dataset
df = pd.read_csv('penguins.csv')

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

X, y = preprocessing(df)

X_train, y_train, X_test, y_test = split(X, y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = MLP(bias, features, classes, tanh_hidden, num_of_hidden_layers, num_hidden_neurons)
model.train()

print(f'Number of parameters: {sum(p.data.size for p in model.parameters())}')

optimizer = optim.Gradient(model.parameters(), lr=learning_rate)


for epoch in range(EPOCHS):

    optimizer.zero_grad()

    x = Tensor(X_train)

    y_onehot = np.eye(3)[y_train]  
    target = Tensor(y_onehot)

    pred = model(x)


    loss = ((pred - target)**2).mean()


    losses.append(loss.data)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data}, Accuracy: {(np.argmax(pred.data, axis=1) == y_train).mean():.4f}')



logits_test = model(Tensor(X_test))
preds_test = np.argmax(logits_test.data, axis=1)

accuracy = np.mean(preds_test == y_test)
print(f"Test Accuracy: {accuracy:.4f}")


plt.scatter(range(len(y_test)), y_test, label="True", alpha=0.6)
plt.scatter(range(len(preds_test)), preds_test, label="Predicted", alpha=0.6)
plt.legend()
plt.title("Predictions vs True Labels")
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.show()



cm = confusion_matrix(y_test, preds_test)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()