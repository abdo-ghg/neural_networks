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

EPOCHS = 5000
learning_rate = 0.01
bias = True


losses = []

np.random.seed(37) # for reproducibility

# dataset
df = pd.read_csv('penguins.csv')

def preprocessing(df):
    df['Species'] = df['Species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
    df['OriginLocation'] = df['OriginLocation'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
    df.dropna(inplace=True)
    X = df.drop('Species', axis=1).values
    y = df['Species'].values
    #normalize the data
    scaler = StandardScaler()
    x = scaler.fit_transform(X)

    return x, y

X, y = preprocessing(df)
print(X.shape)
print(y.shape) 


model = MLP(bias)
model.train()

print(f'Number of parameters: {sum(p.data.size for p in model.parameters())}')

optimizer = optim.Gradient(model.parameters(), lr=learning_rate)


for epoch in range(EPOCHS):

    optimizer.zero_grad()

    x = Tensor(X)
    y_onehot = np.eye(3)[y]  # (148, 3)
    target = Tensor(y_onehot)

    pred = model(x)


    loss = ((pred - target)**2).mean()


    losses.append(loss.data)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(epoch, loss.data)



print("predictions:")
print(model(Tensor(X)).data)

logits = model(Tensor(X))
preds = np.argmax(logits.data, axis=1)

accuracy = np.mean(preds == y)
print(f"Accuracy: {accuracy:.4f}")


plt.scatter(range(len(y)), y, label="True", alpha=0.6)
plt.scatter(range(len(preds)), preds, label="Predicted", alpha=0.6)
plt.legend()
plt.title("Predictions vs True Labels")
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.show()



cm = confusion_matrix(y, preds)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

