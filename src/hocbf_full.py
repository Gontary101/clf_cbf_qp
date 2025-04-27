# hocbf_full.py
# Compatible with Python 2.7
import sympy as sp
import numpy as np

_symbolic_initialized = False
_Lf_psi2_lambda = None
_Lg_psi2_lambda = None
_psi0_lambda = None
_psi1_lambda = None
_psi2_lambda = None

def _initialize_symbolic():
    global _symbolic_initialized, _Lf_psi2_lambda, _Lg_psi2_lambda
    global _psi0_lambda, _psi1_lambda, _psi2_lambda

    if _symbolic_initialized:
        return

    # Define symbols
    p_sym = sp.symbols('p0:3')      # Position
    v_sym = sp.symbols('v0:3')      # Velocity (inertial frame)
    R_sym = sp.symbols('R0:9')      # Rotation matrix elements (flattened, row-major)
    omega_sym = sp.symbols('omega0:3') # Angular velocity (body frame)
    m_sym, g_sym, Ix_sym, Iy_sym, Iz_sym = sp.symbols('m g Ix Iy Iz')
    k0_sym, k1_sym, k2_sym = sp.symbols('k0 k1 k2')
    po_sym = sp.symbols('po0:3')    # Obstacle center
    rs_sym = sp.symbols('rs')        # Obstacle safe radius

    # State vector parts
    p = sp.Matrix(p_sym)
    v = sp.Matrix(v_sym)
    omega = sp.Matrix(omega_sym)
    R = sp.Matrix(3, 3, R_sym)
    po = sp.Matrix(po_sym)

    # Known constants / matrices
    e3 = sp.Matrix([0, 0, 1])
    I_mat = sp.diag(Ix_sym, Iy_sym, Iz_sym)
    I_inv = sp.diag(1/Ix_sym, 1/Iy_sym, 1/Iz_sym)

    # Full state vector (order matters!)
    # Combine state parts into a single list for jacobian
    x_sym_list = list(p) + list(v) + list(R_sym) + list(omega)
    x_sym = sp.Matrix(x_sym_list)

    # Dynamics f(x) (drift terms)
    f_p = v
    f_v = -g_sym * e3
    # Skew-symmetric matrix for cross product
    omega_skew = sp.Matrix([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    f_R_flat = (R * omega_skew).reshape(9, 1)
    f_omega = I_inv * (-omega.cross(I_mat * omega))
    f_sym = sp.Matrix(list(f_p) + list(f_v) + list(f_R_flat) + list(f_omega))

    # Dynamics g(x) (control mapping)
    g_p = sp.zeros(3, 4)
    g_v = sp.Matrix.hstack((1/m_sym) * R * e3, sp.zeros(3, 3)) # Only U1 affects acceleration
    g_R_flat = sp.zeros(9, 4)
    g_omega = sp.Matrix.hstack(sp.zeros(3, 1), I_inv) # U2, U3, U4 affect angular acceleration
    # Combine g parts correctly
    g_sym = sp.Matrix([list(g_p), list(g_v), list(g_R_flat), list(g_omega)]) # Check dimensions N x M (state_dim x input_dim)

    # HOCBF sequence psi_k = Lf psi_{k-1} + k_{k-1} * psi_{k-1}
    # psi_0 = h0
    h0 = (p - po).dot(p - po) - rs_sym**2
    psi_0 = h0

    # psi_1 = Lf h0 + k0 * psi_0
    grad_psi_0 = psi_0.jacobian(x_sym) # Should be 1 x N
    Lf_psi_0 = grad_psi_0 * f_sym      # (1xN) * (Nx1) -> 1x1
    psi_1 = Lf_psi_0[0] + k0_sym * psi_0 # Lf_psi_0 is 1x1 matrix

    # psi_2 = Lf psi_1 + k1 * psi_1
    grad_psi_1 = psi_1.jacobian(x_sym) # 1 x N
    Lf_psi_1 = grad_psi_1 * f_sym      # (1xN) * (Nx1) -> 1x1
    psi_2 = Lf_psi_1[0] + k1_sym * psi_1

    # Final constraint terms: Lf psi_2 + k2 * psi_2 >= - Lg psi_2 * u
    # G = -Lg psi_2
    # h = Lf psi_2 + k2 * psi_2
    grad_psi_2 = psi_2.jacobian(x_sym) # 1 x N
    Lf_psi_2 = grad_psi_2 * f_sym      # (1xN) * (Nx1) -> 1x1
    Lg_psi_2 = grad_psi_2 * g_sym      # (1xN) * (NxM) -> 1xM (1x4)

    G_expr = -Lg_psi_2
    h_expr = Lf_psi_2[0] + k2_sym * psi_2

    # Lambdify expressions for numerical evaluation
    # Input args must match the order they are used in the functions
    # Combine symbols into a single list/tuple for lambdify
    _input_symbols = list(p_sym) + list(v_sym) + list(R_sym) + list(omega_sym) + \
                     list(po_sym) + \
                     [rs_sym, m_sym, g_sym, Ix_sym, Iy_sym, Iz_sym, k0_sym, k1_sym, k2_sym]

    _Lf_psi2_lambda = sp.lambdify(_input_symbols, Lf_psi_2[0], 'numpy')
    _Lg_psi2_lambda = sp.lambdify(_input_symbols, Lg_psi_2, 'numpy')

    # Also lambdify intermediate psi functions if needed for debugging/analysis
    _psi0_lambda = sp.lambdify(_input_symbols, psi_0, 'numpy')
    _psi1_lambda = sp.lambdify(_input_symbols, psi_1, 'numpy')
    _psi2_lambda = sp.lambdify(_input_symbols, psi_2, 'numpy')


    _symbolic_initialized = True
    print("Symbolic HOCBF functions initialized (Python 2.7 compatible).")


def compute_hocbf_constraint(x_state_np, obs_centre_np, r_obs_safe, k0, k1, k2, m, g, Ix, Iy, Iz):
    """
    Calculates G and h for the HOCBF constraint G*u <= h for one obstacle.

    Args:
        x_state_np: Full state [p(3), v(3), R_flat(9), omega(3)] as numpy array
        obs_centre_np: Obstacle center [x,y,z] as numpy array
        r_obs_safe: Safe radius (obstacle radius + drone radius)
        k0, k1, k2: HOCBF gain parameters
        m, g, Ix, Iy, Iz: Drone physical parameters

    Returns:
        G: 1x4 numpy array
        h: scalar float
    """
    if not _symbolic_initialized:
        _initialize_symbolic()

    # Prepare arguments for lambdified functions (Python 2.7 compatible)
    # Convert numpy arrays to lists or individual elements as needed by lambdify
    state_args = list(x_state_np[0:3]) + list(x_state_np[3:6]) + \
                 list(x_state_np[6:15]) + list(x_state_np[15:18])
    obs_args = list(obs_centre_np)
    param_args = [r_obs_safe, m, g, Ix, Iy, Iz, k0, k1, k2]
    args = tuple(state_args + obs_args + param_args) # Combine all arguments into a tuple

    # Evaluate expressions numerically
    try:
        # Evaluate psi2 first as it's needed for the h calculation's k2 term
        psi2_val = _psi2_lambda(*args)

        Lf_psi_2_val = _Lf_psi2_lambda(*args)
        Lg_psi_2_val = _Lg_psi2_lambda(*args).reshape(1, 4) # Ensure shape

        G = -Lg_psi_2_val
        h = Lf_psi_2_val + k2 * psi2_val # Use the evaluated psi2_val here

        # Check for NaN/Inf results which indicate symbolic/numerical issues
        if np.isnan(h) or np.any(np.isnan(G)) or np.isinf(h) or np.any(np.isinf(G)):
             print("Warning: NaN or Inf detected in HOCBF constraint calculation.")
             # Return a constraint that is likely infeasible or allows nominal
             return np.zeros((1, 4)), -1e9 # Example: make it highly restrictive

        return G, h

    except Exception as e:
        # Use print for Python 2.7 compatibility
        print("Error during HOCBF constraint evaluation: {}".format(e))
        # Return a safe default (e.g., allow nominal control) or raise error
        return np.zeros((1, 4)), 1e9 # Example: make it non-restrictive