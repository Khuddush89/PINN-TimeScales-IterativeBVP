import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
np.random.seed(1234)

# Hybrid time scale: [0,1) ∪ {2,3,...,10}
Nc = 200
t_c = torch.linspace(0, 1, Nc + 1)[:-1].view(-1, 1).to(device)
t_d = torch.arange(2, 11, dtype=torch.float32).view(-1, 1).to(device)
t_d_int = torch.arange(2, 9, dtype=torch.float32).view(-1, 1).to(device)

# Exact coefficients
def a_exact(t):
    a_cont = 1.0 + t
    a_disc = 1.0 + 0.1 * (t - 1.0)
    return torch.where(t <= 1.0, a_cont, a_disc)

def b_exact(t):
    b_cont = torch.exp(-t)
    b_disc = np.exp(-1.0) + 0.1 * torch.sin(np.pi * (t - 2.0) / 8.0)
    return torch.where(t <= 1.0, b_cont, b_disc)

# Exact solutions
def y1(t):
    y_cont = 0.3 + 0.2 * torch.sin(np.pi * t)
    y_disc = 0.3 + 0.2 * torch.sin(np.pi * (t - 2.0) / 8.0)
    return torch.where(t <= 1.0, y_cont, y_disc)

def y2(t):
    y_cont = 0.4 + 0.15 * torch.cos(np.pi * t)
    y_disc = 0.4 + 0.15 * torch.cos(3.0 * np.pi * (t - 2.0) / 8.0)
    return torch.where(t <= 1.0, y_cont, y_disc)

# Classical second derivatives on continuous part
def y1_dd_cont(t):
    return -(0.2 * np.pi ** 2) * torch.sin(np.pi * t)

def y2_dd_cont(t):
    return -(0.15 * np.pi ** 2) * torch.cos(np.pi * t)

# Iterative terms
def y1_iter(x):
    return y1(y1(x))

def y2_iter(x):
    return y2(y2(x))

# Manufactured forcing (continuous)
with torch.no_grad():
    f1_cont = (y1_dd_cont(t_c) - a_exact(t_c) * y1(t_c) - b_exact(t_c) * y1_iter(t_c))
    f2_cont = (y2_dd_cont(t_c) - a_exact(t_c) * y2(t_c) - b_exact(t_c) * y2_iter(t_c))

# Discrete second delta-derivative
def delta2(f, n):
    return f(n + 2) - 2 * f(n + 1) + f(n)

with torch.no_grad():
    f1_disc = (delta2(y1, t_d_int) - a_exact(t_d_int) * y1(t_d_int + 1) - b_exact(t_d_int) * y1_iter(t_d_int + 1))
    f2_disc = (delta2(y2, t_d_int) - a_exact(t_d_int) * y2(t_d_int + 1) - b_exact(t_d_int) * y2_iter(t_d_int + 1))

# Pre-training identifiability check
with torch.no_grad():
    t_test = torch.cat([t_c, t_d], dim=0)
    U1 = y1(t_test); U2 = y2(t_test); V1 = y1_iter(t_test); V2 = y2_iter(t_test)
    Delta = U1 * V2 - U2 * V1
    min_abs = torch.min(torch.abs(Delta)).item()
    max_abs = torch.max(torch.abs(Delta)).item()
    print(f"Pre-training: min |Δ| = {min_abs:.4e}, max |Δ| = {max_abs:.4e}")

# Neural networks
class CoeffNet(nn.Module):
    def __init__(self, width=100):
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
epochs = 20000

for epoch in range(epochs):
    optimizer.zero_grad()
    # Continuous part
    A_c = A_net(t_c); B_c = B_net(t_c)
    Rc1 = y1_dd_cont(t_c) - A_c * y1(t_c) - B_c * y1_iter(t_c) - f1_cont
    Rc2 = y2_dd_cont(t_c) - A_c * y2(t_c) - B_c * y2_iter(t_c) - f2_cont
    loss_cont = torch.mean(Rc1**2) + torch.mean(Rc2**2)
    # Discrete part
    A_d = A_net(t_d_int); B_d = B_net(t_d_int)
    Rd1 = delta2(y1, t_d_int) - A_d * y1(t_d_int+1) - B_d * y1_iter(t_d_int+1) - f1_disc
    Rd2 = delta2(y2, t_d_int) - A_d * y2(t_d_int+1) - B_d * y2_iter(t_d_int+1) - f2_disc
    loss_disc = torch.mean(Rd1**2) + torch.mean(Rd2**2)
    loss = loss_cont + loss_disc
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
    A_c = A_net(t_c); B_c = B_net(t_c)
    Rc1 = y1_dd_cont(t_c) - A_c*y1(t_c) - B_c*y1_iter(t_c) - f1_cont
    Rc2 = y2_dd_cont(t_c) - A_c*y2(t_c) - B_c*y2_iter(t_c) - f2_cont
    loss_cont = torch.mean(Rc1**2) + torch.mean(Rc2**2)
    A_d = A_net(t_d_int); B_d = B_net(t_d_int)
    Rd1 = delta2(y1, t_d_int) - A_d*y1(t_d_int+1) - B_d*y1_iter(t_d_int+1) - f1_disc
    Rd2 = delta2(y2, t_d_int) - A_d*y2(t_d_int+1) - B_d*y2_iter(t_d_int+1) - f2_disc
    loss_disc = torch.mean(Rd1**2) + torch.mean(Rd2**2)
    loss = loss_cont + loss_disc
    loss.backward()
    return loss
optimizer_lbfgs.step(closure)

# Evaluation
with torch.no_grad():
    t_plot = torch.cat([t_c, t_d], dim=0)
    A_pred = A_net(t_plot).cpu().numpy()
    B_pred = B_net(t_plot).cpu().numpy()
    A_true = a_exact(t_plot).cpu().numpy()
    B_true = b_exact(t_plot).cpu().numpy()

errA = np.linalg.norm(A_pred - A_true) / np.linalg.norm(A_true)
errB = np.linalg.norm(B_pred - B_true) / np.linalg.norm(B_true)
maxErrA = np.max(np.abs(A_pred - A_true))
maxErrB = np.max(np.abs(B_pred - B_true))
rmseA = np.sqrt(np.mean((A_pred - A_true)**2))
rmseB = np.sqrt(np.mean((B_pred - B_true)**2))

print("\n=== Hybrid time scale ===")
print(f"Relative L2 error a(t) = {errA:.6e}")
print(f"Relative L2 error b(t) = {errB:.6e}")
print(f"Max error a(t) = {maxErrA:.6e}")
print(f"Max error b(t) = {maxErrB:.6e}")
print(f"RMSE a(t) = {rmseA:.6e}")
print(f"RMSE b(t) = {rmseB:.6e}")

# Plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(t_plot.cpu(), A_true, label='Exact a(t)')
plt.plot(t_plot.cpu(), A_pred, '--', label='Recovered a(t)')
plt.legend(); plt.grid(); plt.title('Coefficient a(t)')
plt.subplot(1,2,2)
plt.plot(t_plot.cpu(), B_true, label='Exact b(t)')
plt.plot(t_plot.cpu(), B_pred, '--', label='Recovered b(t)')
plt.legend(); plt.grid(); plt.title('Coefficient b(t)')
plt.tight_layout(); plt.show()

plt.figure()
plt.semilogy(loss_history)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(); plt.title('Loss history')
plt.show()

# Identifiability determinant
with torch.no_grad():
    U1 = y1(t_plot); U2 = y2(t_plot); V1 = y1_iter(t_plot); V2 = y2_iter(t_plot)
    Delta = U1 * V2 - U2 * V1
    plt.figure()
    plt.plot(t_plot.cpu(), Delta.cpu().numpy())
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('t'); plt.ylabel('Δ(t)'); plt.grid(); plt.title('Identifiability determinant')
    plt.show()