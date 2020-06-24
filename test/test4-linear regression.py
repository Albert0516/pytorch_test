import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Linear, MSELoss
from torch.optim import SGD

# x = np.linspace(0, 20, 500)
# y = 5*x + 7
# plt.plot(x, y)

x = np.random.rand(256)
noise = np.random.random(256)/4
y = 5*x + 7 + noise
# df = pd.DataFrame()
# df['x'] = x
# df['y'] = y
# sns.lmplot(x='x', y='y', data=df)
# plt.show()


model = Linear(1, 1)
criterion = MSELoss()
optim = SGD(model.parameters(), lr=0.01)
epochs = 3000

x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

for i in range(epochs):
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    outputs = model(inputs)

    optim.zero_grad()

    loss = criterion(outputs, labels)

    loss.backward()

    optim.step()

    if i % 100 == 0:
        print('epoch {}, loss {:1.4f}'.format(i, loss.data.item()))

[w, b] = model.parameters()
print(w.item(), b.item())

predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'go', label='data', alpha=0.3)
plt.plot(x_train, predicted, label='predicted', alpha=1)
plt.show()
