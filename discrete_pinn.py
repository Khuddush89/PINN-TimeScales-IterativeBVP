import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
np.random.seed(1234)

# Time scale: {0,1,...,10}
n = torch.arange(0, 11, dtype=torch.float32).view(-1,1).to(device)
n_int = torch.arange(0, 9, dtype=torch.float32).view(-1,1).to(device)  # interior for Δ²

# Exact coefficients
def a_exact(x):
    return 0.5 + 0.1 * x

def b_exact(x):
    return 1.0 + 0.3 * torch.sin(np.pi * x / 10.0)

# Exact solutions
def y1(x):
    return 0.3 + 0.2 * torch.sin(np.pi * x / 10.0)

def y2(x):
    return 0.4 + 0.15 * torch.cos(np.pi * x / 10.0)

# Second delta derivatives
def y1_ddelta(n):
    return y1(n+2) - 2*y1(n+1) + y1(n)

def y2_ddelta(n):
    return y2(n+2) - 2*y2(n+1) + y2(n)

# Iterative terms
def y1_iter(n):
    return y1(y1(n+1))

def y2_iter(n):
    return y2(y2(n+1))

# Manufactured forcing
with torch.no_grad():
    f1_disc = y1_ddelta(n_int) - a_exact(n_int)*y1(n_int+1) - b_exact(n_int)*y1_iter(n_int)
    f2_disc = y2_ddelta(n_int) - a_exact(n_int)*y2(n_int+1) - b_exact(n_int)*y2_iter(n_int)

# Neural networks
class CoeffNet(nn.Module):
    def __init__(self, width=60):
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
epochs = 15000

for epoch in range(epochs):
    optimizer.zero_grad()
    A = A_net(n_int)
    B = B_net(n_int)
    R1 = y1_ddelta(n_int) - A*y1(n_int+1) - B*y1_iter(n_int) - f1_disc
    R2 = y2_ddelta(n_int) - A*y2(n_int+1) - B*y2_iter(n_int) - f2_disc
    loss = torch.mean(R1**2) + torch.mean(R2**2)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss = {loss.item():.6e}")

# L-BFGS refinement
optimizer_lbfgs = torch.optim.LBFGS(params, max_iter=500)
def closure():
    optimizer_lbfgs.zero_grad()
    A = A_net(n_int)
    B = B_net(n_int)
    R1 = y1_ddelta(n_int) - A*y1(n_int+1) - B*y1_iter(n_int) - f1_disc
    R2 = y2_ddelta(n_int) - A*y2(n_int+1) - B*y2_iter(n_int) - f2_disc
    loss = torch.mean(R1**2) + torch.mean(R2**2)
    loss.backward()
    return loss
optimizer_lbfgs.step(closure)

# Evaluation
with torch.no_grad():
    A_pred = A_net(n).cpu().numpy()
    B_pred = B_net(n).cpu().numpy()
    A_true = a_exact(n).cpu().numpy()
    B_true = b_exact(n).cpu().numpy()

errA = np.linalg.norm(A_pred - A_true) / np.linalg.norm(A_true)
errB = np.linalg.norm(B_pred - B_true) / np.linalg.norm(B_true)
maxErrA = np.max(np.abs(A_pred - A_true))
maxErrB = np.max(np.abs(B_pred - B_true))
rmseA = np.sqrt(np.mean((A_pred - A_true)**2))
rmseB = np.sqrt(np.mean((B_pred - B_true)**2))

print("\n=== Discrete time scale ===")
print(f"Relative L2 error a(n) = {errA:.6e}")
print(f"Relative L2 error b(n) = {errB:.6e}")
print(f"Max error a(n) = {maxErrA:.6e}")
print(f"Max error b(n) = {maxErrB:.6e}")
print(f"RMSE a(n) = {rmseA:.6e}")
print(f"RMSE b(n) = {rmseB:.6e}")

# Plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(n.cpu(), A_true, 'o-', label='Exact a(n)')
plt.plot(n.cpu(), A_pred, 's--', label='Recovered a(n)')
plt.legend(); plt.grid(); plt.title('Coefficient a(n)')
plt.subplot(1,2,2)
plt.plot(n.cpu(), B_true, 'o-', label='Exact b(n)')
plt.plot(n.cpu(), B_pred, 's--', label='Recovered b(n)')
plt.legend(); plt.grid(); plt.title('Coefficient b(n)')
plt.tight_layout(); plt.show()

plt.figure()
plt.semilogy(loss_history)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(); plt.title('Loss history')
plt.show()