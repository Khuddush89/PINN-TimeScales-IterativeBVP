import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
np.random.seed(1234)

# Time scale: [0,1]
t = torch.linspace(0, 1, 200).view(-1, 1).to(device)

# Exact coefficients
def a_exact(t):
    return 1.0 + t

def b_exact(t):
    return torch.exp(-t)

# Exact solutions
def y1(t):
    return 0.3 + 0.2 * torch.sin(np.pi * t)

def y2(t):
    return 0.4 + 0.15 * torch.cos(np.pi * t)

# Exact second derivatives
def y1_dd(t):
    return -(0.2 * np.pi ** 2) * torch.sin(np.pi * t)

def y2_dd(t):
    return -(0.15 * np.pi ** 2) * torch.cos(np.pi * t)

# Iterative terms
def y1_iter(x):
    return y1(y1(x))

def y2_iter(x):
    return y2(y2(x))

# Manufactured forcing
with torch.no_grad():
    f1_val = y1_dd(t) - a_exact(t) * y1(t) - b_exact(t) * y1_iter(t)
    f2_val = y2_dd(t) - a_exact(t) * y2(t) - b_exact(t) * y2_iter(t)

# Neural networks
class CoeffNet(nn.Module):
    def __init__(self, width=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1)
        )
    def forward(self, x):
        return self.net(x)

A_net = CoeffNet().to(device)
B_net = CoeffNet().to(device)

# Optimisation
params = list(A_net.parameters()) + list(B_net.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)
loss_history = []
epochs = 10000

for epoch in range(epochs):
    optimizer.zero_grad()
    A = A_net(t)
    B = B_net(t)
    R1 = y1_dd(t) - A * y1(t) - B * y1_iter(t) - f1_val
    R2 = y2_dd(t) - A * y2(t) - B * y2_iter(t) - f2_val
    loss = torch.mean(R1**2) + torch.mean(R2**2)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss = {loss.item():.6e}")

# L-BFGS refinement
optimizer_lbfgs = torch.optim.LBFGS(params, max_iter=500,
                                    tolerance_grad=1e-10,
                                    tolerance_change=1e-12)
def closure():
    optimizer_lbfgs.zero_grad()
    A = A_net(t)
    B = B_net(t)
    R1 = y1_dd(t) - A * y1(t) - B * y1_iter(t) - f1_val
    R2 = y2_dd(t) - A * y2(t) - B * y2_iter(t) - f2_val
    loss = torch.mean(R1**2) + torch.mean(R2**2)
    loss.backward()
    return loss
optimizer_lbfgs.step(closure)

# Evaluation
with torch.no_grad():
    A_pred = A_net(t).cpu().numpy()
    B_pred = B_net(t).cpu().numpy()
    A_true = a_exact(t).cpu().numpy()
    B_true = b_exact(t).cpu().numpy()

errA = np.linalg.norm(A_pred - A_true) / np.linalg.norm(A_true)
errB = np.linalg.norm(B_pred - B_true) / np.linalg.norm(B_true)
maxErrA = np.max(np.abs(A_pred - A_true))
maxErrB = np.max(np.abs(B_pred - B_true))
rmseA = np.sqrt(np.mean((A_pred - A_true)**2))
rmseB = np.sqrt(np.mean((B_pred - B_true)**2))

print("\n=== Continuous time scale ===")
print(f"Relative L2 error a(t) = {errA:.6e}")
print(f"Relative L2 error b(t) = {errB:.6e}")
print(f"Max error a(t) = {maxErrA:.6e}")
print(f"Max error b(t) = {maxErrB:.6e}")
print(f"RMSE a(t) = {rmseA:.6e}")
print(f"RMSE b(t) = {rmseB:.6e}")

# Plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(t.cpu(), A_true, label='Exact a(t)')
plt.plot(t.cpu(), A_pred, '--', label='Recovered a(t)')
plt.legend(); plt.grid(); plt.title('Coefficient a(t)')
plt.subplot(1,2,2)
plt.plot(t.cpu(), B_true, label='Exact b(t)')
plt.plot(t.cpu(), B_pred, '--', label='Recovered b(t)')
plt.legend(); plt.grid(); plt.title('Coefficient b(t)')
plt.tight_layout(); plt.show()

plt.figure()
plt.semilogy(loss_history)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(); plt.title('Loss history')
plt.show()