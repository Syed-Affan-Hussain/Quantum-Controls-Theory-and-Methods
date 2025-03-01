import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
#Code Author: Syed Affan Hussain
# Book Credits : Cong Shuang
# ========================
# Core Functions
# ========================

def construct_U(t, omega0, Omega, phi):
    """Time evolution operator (Equation 2.21)"""
    mat1 = np.array([
        [np.exp(1j * omega0 * t / 2), 0],
        [0, np.exp(-1j * (omega0 * t + 2 * phi) / 2)]
    ], dtype=complex)
    
    mat2 = np.array([
        [np.cos(Omega*t/2), 1j*np.sin(Omega*t/2)],
        [1j*np.sin(Omega*t/2), np.cos(Omega*t/2)]
    ], dtype=complex)
    
    mat3 = np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=complex)
    
    return mat1 @ mat2 @ mat3

def plot_bloch(ax, x, y, z, title):
    """3D Bloch sphere plotting"""
    # Sphere wireframe
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = np.cos(u)*np.sin(v)
    ys = np.sin(u)*np.sin(v)
    zs = np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color="k", alpha=0.1)
    
    # Trajectory
    ax.plot(x, y, z, 'b-', lw=1.5)
    ax.scatter(x[0], y[0], z[0], color='g', s=50, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='r', s=50, label='End')
    ax.set_title(title, pad=20)
    ax.legend()

# ========================
# Example Simulations
# ========================

def example1():
    """|0⟩ → |1⟩ with minimum Ωt (Figure 2.8)"""
    omega0 = 5.0
    Omega = 2.5
    phi = 0.0
    t_total = np.pi/Omega  # 1.2566
    
    # Simulate trajectory
    t_points = np.linspace(0, t_total, 200)
    x, y, z = [], [], []
    psi0 = np.array([1, 0], dtype=complex)
    
    for t in t_points:
        U = construct_U(t, omega0, Omega, phi)
        psi_t = U @ psi0
        alpha, beta = psi_t
        x.append(2*np.real(alpha*np.conj(beta)))
        y.append(2*np.imag(alpha*np.conj(beta)))
        z.append(np.abs(alpha)**2 - np.abs(beta)**2)
    
    # Plotting
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    plot_bloch(ax, x, y, z, "Example 1: |0⟩ → |1⟩\nMinimum Ωt = π")
    
    ax2 = fig.add_subplot(122)
    ax2.plot(t_points, Omega * np.cos(omega0 * t_points + phi), 'r-')
    ax2.set_title("Control Field γB_x(t)")
    ax2.set_xlabel("Time")

def example2():
    """(|0⟩ + |1⟩)/√2 → 0.8|0⟩ + 0.6|1⟩ (Figure 2.9)"""
    omega0 = 5.0
    Omega = 0.284 / (2*np.pi/omega0)  # ≈ 0.284/1.2566
    phi = 0.0
    t_total = 2*np.pi/omega0
    
    # Initial state
    psi0 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    
    # Simulate trajectory
    t_points = np.linspace(0, t_total, 400)
    x, y, z = [], [], []
    
    for t in t_points:
        U = construct_U(t, omega0, Omega, phi)
        psi_t = U @ psi0
        alpha, beta = psi_t
        x.append(2*np.real(alpha*np.conj(beta)))
        y.append(2*np.imag(alpha*np.conj(beta)))
        z.append(np.abs(alpha)**2 - np.abs(beta)**2)
    
    # Plotting
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    plot_bloch(ax, x, y, z, "Example 2: Superposition State\nMinimum Ωt ≈ 0.284")
    
    ax2 = fig.add_subplot(122)
    ax2.plot(t_points, Omega * np.cos(omega0 * t_points + phi), 'r-')
    ax2.set_title("Control Field γB_x(t)")
    ax2.set_xlabel("Time")

def example3():
    """|0⟩ → |1⟩ with Fixed T=0.02 (Figure 2.10)"""
    omega0 = 5.0
    T = 0.02
    Omega = np.pi / T  # ≈ 157.08
    phi = 0.0
    
    # Simulate trajectory
    t_points = np.linspace(0, T, 100)
    x, y, z = [], [], []
    psi0 = np.array([1, 0], dtype=complex)
    
    for t in t_points:
        U = construct_U(t, omega0, Omega, phi)
        psi_t = U @ psi0
        alpha, beta = psi_t
        x.append(2*np.real(alpha*np.conj(beta)))
        y.append(2*np.imag(alpha*np.conj(beta)))
        z.append(np.abs(alpha)**2 - np.abs(beta)**2)
    
    # Plotting
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    plot_bloch(ax, x, y, z, "Example 3: |0⟩ → |1⟩\nFixed T=0.02")
    
    ax2 = fig.add_subplot(122)
    ax2.plot(t_points, Omega * np.cos(omega0 * t_points + phi), 'r-')
    ax2.set_title("Control Field γB_x(t)")
    ax2.set_xlabel("Time")

def example4():
    """Superposition State with Fixed T=0.02 (Figure 2.11)"""
    omega0 = 5.0
    T = 0.02
    
    # State parameters
    theta1 = np.pi/2
    theta2 = 2 * np.arccos(0.8)
    phi1 = phi2 = 0.0
    
    # Solve for Ω and φ with improved convergence
    def equations(vars):
        phi, psi = vars
        # Add small epsilon to prevent division by zero
        eps = 1e-12
        
        numerator = (np.cos(theta1)*np.sin(theta2)*np.sin(omega0*T + phi) 
                    - np.sin(theta1)*np.cos(theta2)*np.sin(phi + eps))
        
        denominator = (np.sin(theta1)*np.sin(theta2)*np.sin(phi + eps)*np.sin(omega0*T + phi) 
                     + np.cos(theta1)*np.cos(theta2) + eps)
        
        eq1 = np.tan(psi) - numerator/(denominator + eps)
        eq2 = phi - np.arctan2(
            np.sin(theta1)*np.cos(phi1) - np.sin(theta2)*np.cos(omega0*T + eps),
            np.sin(theta1)*np.sin(phi1) - np.sin(theta2)*np.sin(omega0*T + eps) + eps
        )
        return [eq1, eq2]
    
    # Better initial guess and solver parameters
    phi_initial = -0.4  # Estimated from Figure 2.11
    psi_initial = 36.125 * T  # From text value Ω=36.125
    phi, psi = fsolve(
        equations, 
        (phi_initial, psi_initial),
        xtol=1e-8,  # Tighten tolerance
        maxfev=2000  # Increase max iterations
    )
    
    Omega = psi / T
    
    # Simulate trajectory
    t_points = np.linspace(0, T, 100)
    x, y, z = [], [], []
    psi0 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    
    for t in t_points:
        U = construct_U(t, omega0, Omega, phi)
        psi_t = U @ psi0
        alpha, beta = psi_t
        x.append(2*np.real(alpha*np.conj(beta)))
        y.append(2*np.imag(alpha*np.conj(beta)))
        z.append(np.abs(alpha)**2 - np.abs(beta)**2)
    
    # Plotting
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    plot_bloch(ax, x, y, z, "Example 4: Superposition State\nFixed T=0.02")
    
    ax2 = fig.add_subplot(122)
    ax2.plot(t_points, Omega * np.cos(omega0 * t_points + phi), 'r-')
    ax2.set_title("Control Field γB_x(t)")
    ax2.set_xlabel("Time")

def example5():
    """Ω vs. T Variation (Figure 2.12)"""
    omega0 = 5.0
    T_values = np.linspace(0.01, 1.0, 50)
    Omega_values = []
    
    # Simplified calculation for demonstration
    for T in T_values:
        psi = np.pi  # Approximation
        Omega = psi / T
        Omega_values.append(Omega)
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(T_values, Omega_values, 'b-', lw=2)
    plt.xlabel("Time T", fontsize=12)
    plt.ylabel("Rabi Frequency Ω", fontsize=12)
    plt.title("Example 5: Ω vs. T Relationship", pad=20)
    plt.grid(True)

# ========================
# Execute All Examples
# ========================
if __name__ == "__main__":
    #example1()
    #example2()
    example4()
    #example5()
    plt.tight_layout()
    plt.show()
