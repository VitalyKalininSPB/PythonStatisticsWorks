import torch

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
# Is it necessary?
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

# Derivatives
def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c)
    return dsq_diffs

def dmodel_dw(t_u, w, b):
    return t_u

def dmodel_db(t_u, w, b):
    return 1.0

def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dw = dloss_fn(t_p, t_c) * dmodel_dw(t_u, w, b)
    dloss_db = dloss_fn(t_p, t_c) * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        # This is the forward pass.
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        # And this is the backward pass.
        grad = grad_fn(t_u, t_c, t_p, w, b)
        params = params - learning_rate * grad
        # This logging line can be verbose.
        print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params


# First attempt
params = training_loop(
    n_epochs = 100,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_u,
    t_c = t_c)

print(params)
# NaN - we need to make learning rate less

print("Second attempt")
params = training_loop(
    n_epochs = 100,
    learning_rate = 1e-4,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_u,
    t_c = t_c)

print(params)
# Loss decreased slowly. We should use normalization (?)and increase n epochs

print("Third attempt")
t_un = 0.1 * t_u
params = training_loop(
    n_epochs = 100,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_un,
    t_c = t_c)

print(params)
# We should increase n epochs for get loss to less value

print("Fourth attempt")
#t_un = 0.1 * t_u
t_un = t_u
params = training_loop(
    n_epochs = 5000,
    learning_rate = 1e-4,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_un,
    t_c = t_c)

print(params)

# Plot our data
from matplotlib import pyplot as plt

t_p = model(t_un, *params)
# Remember that you’re training on the normalized unknown units
fig = plt.figure(dpi=200)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.show()
