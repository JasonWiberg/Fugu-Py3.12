from fugu.backends.slca_backend import slca_Backend as slca
from fugu.bricks import LCABrick
import numpy as np
import pytest
from fugu import Scaffold
"""
Integration test for using a general neuron with a compartment for an algorithm.
Compartment used: compartments.py/RecurrentInhibition

S-LCA implements this using a network of spiking neurons with lateral inhibition.

The paper that inspired this Fugu implementation is the following:

[1] P. T. P. Tang, T.-H. Lin, and M. Davies, “Sparse coding by spiking neural networks: 
Convergence theory and computational results,” arXiv preprint arXiv:1705.05475, May 2017
"""

def normalize_columns(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
    return A / norms

def classo_fista_nonneg(Phi, s, lam, max_iter=5000, tol=1e-9):
    """FISTA solver for nonnegative CLASSO: min ||s - Phi a||^2/2 + lam||a||_1"""
    Phi = normalize_columns(Phi)
    PhiT = Phi.T
    PhiTPhi = PhiT @ Phi
    PhiTs = PhiT @ s
    L = np.linalg.eigvalsh(PhiTPhi).max() + 1e-12
    tstep = 1.0 / L

    def prox_nonneg_l1(x, threshold):
        return np.maximum(0.0, x - threshold)

    N = Phi.shape[1]
    a = np.zeros(N)
    y_iter = a.copy()
    theta = 1.0

    for _ in range(max_iter):
        grad = PhiTPhi @ y_iter - PhiTs
        a_next = prox_nonneg_l1(y_iter - tstep * grad, lam * tstep)
        theta_next = 0.5 * (1 + np.sqrt(1 + 4 * theta * theta))
        y_iter = a_next + (theta - 1) / theta_next * (a_next - a)
        if np.linalg.norm(a_next - a) < tol * (np.linalg.norm(a) + 1e-12):
            a = a_next
            break
        a, theta = a_next, theta_next

    return a



#### For repeating results from [1] ####
Phi = np.array([
    [0.3313, 0.8148, 0.4364],
    [0.8835, 0.3621, 0.2182],
    [0.3313, 0.4527, 0.8729],
], dtype=float)

# Input signal y (3-dimensional)
y = np.array([0.5, 1.0, 1.5], dtype=float)

# Sparsity parameter lambda
lam = 0.1

a_ground_truth = classo_fista_nonneg(Phi, y, lam)
print("Ground truth CLASSO solution (a*):")
print(np.round(a_ground_truth, 6))


def slca_lay_bricks():
    # Create a scaffold (container for neural circuits)
    scaffold = Scaffold()

    # Add the LCA brick with our problem parameters
    scaffold.add_brick(
            LCABrick(Phi=Phi, input_signal=y, dt=1e-3, lam=lam),
        output=True
    )

    # Construct the actual neural graph
    scaffold.lay_bricks()

    assert isinstance(scaffold.bricks[0], LCABrick)
    assert scaffold.bricks[0].Phi is Phi
    assert scaffold.bricks[0].input_signal is y
    assert scaffold.bricks[0].lam == lam
    assert scaffold.bricks[0].dt == 1e-3



def test_slca_scaffold():
    backend = slca()
    # Create a scaffold (container for neural circuits)
    scaffold = Scaffold()

    # Add the LCA brick with our problem parameters
    scaffold.add_brick(
            LCABrick(Phi=Phi, input_signal=y, dt=1e-3, lam=lam, name='LCABrick'),
        output=True
    )

    # Construct the actual neural graph
    scaffold.lay_bricks()
    
    # assert (scaffold.tag_to_name.get('LCABrick-0') is 'LCABrick')
    assert scaffold.bricks[0].Phi is Phi
    assert scaffold.bricks[0].input_signal is y
    assert scaffold.bricks[0].lam == lam
    assert scaffold.bricks[0].dt == 1e-3


# def test_slca_compile():
#     backend = slca()

#     compile_args = {
#         'Phi': Phi,
#         'y': y,
#         'lam': lam,
#         'T_steps': 1000,
#         't0_steps': 100,
#         'unit_area': True
#     }
#     backend.compile(scaffold=None, compile_args=compile_args)
#     assert backend.Phi == Phi
#     assert backend.y_obs == y
#     assert backend.lam == lam
#     assert backend.T_steps == 1000 
#     assert backend.t0_steps == 100 
#     assert backend.unit_area is True


def test_slca_run():
    backend = slca()
    # Create a scaffold (container for neural circuits)
    scaffold = Scaffold()

    # Add the LCA brick with our problem parameters
    scaffold.add_brick(
            LCABrick(Phi=Phi, input_signal=y, dt=1e-3, lam=lam),
        output=True
    )

    # Construct the actual neural graph
    scaffold.lay_bricks()


    compile_args = {
        'Phi': Phi,
        'y': y,
        'lam': lam,
        'T_steps': 100000,
        't0_steps': 100,
    }
    backend.compile(scaffold=scaffold, compile_args=compile_args)
    results = backend.run(rescale=True, dt=1e-3)
    assert results is not None
    assert results['a_tail'].shape == (Phi.shape[1],) 
    # check to see if the results match 
    assert np.allclose(results['a_tail'], a_ground_truth, atol=1e-2)