"""Contains functions to extract/ prepare states for rev/frem annealing."""

# imports
import random
import numpy as np
import qutip as qt

# internal import
import qanic as qa

# reverse annealing measurement schemes
def ml_measurement(H, probs, subset=None):
    """
    Finds the most likely measurement outcome predicted from discrete probability
    distribution of n-qubits. (Relies on probs having canonical tensor product ordering).

    Inputs
    ---------------------------------------------------------------------
    H: IsingH Hamiltonian
    probs: list of probability amplitudes for each composite qubit state
    subset: list of qubits of interest

    Output
    ---------------------------------------------------------------------
    ml_state: reconstructed most likely state as a dictionary of {q0: ket(q0), q1: ket(q1), ...}
    """
    qubits = H.qubits
    # if we are not considered a subset, extract all qubits
    if subset is None:
        subset = qubits
    # gets the most likely state
    max_idx = np.argmax(probs)

    # return classical state with tensor product ordering assumption
    return infer_classical_state(qubits, max_idx, subset)

def c_diag_H(H):
    """
    Diag H, find a random classical ground-state and return it

    Inputs
    ---------------------------------------------------------------------
    H: IsingH Hamiltonian

    Output
    ---------------------------------------------------------------------
    state: dict--contains qubit to qutip state mapping {q0: ket(q0), ...}
    """
    # create IsingH representation of dictHR
    qubits = H.qubits
    # get non-zero indices of gs (which encode state)
    gs = H.Hz_gs_info()['gs']
    nz_idx = np.nonzero(gs)

    # select a random entry from non-zero superposition indices
    gs_idx = random.choice(nz_idx[0])

    # return classical state with tensor product ordering assumption
    return infer_classical_state(qubits, gs_idx, qubits)

def coin_toss(H):
    """
    Guesses ground-state of H by a coin-toss for each qubit (up/down 50/50).

    Inputs
    ---------------------------------------------------------------------
    H: IsingH Hamiltonian

    Output
    ---------------------------------------------------------------------
    state: dict--contains qubit to qutip state mapping {q0: ket(q0), ...}
    """
    state = {}
    # create IsingH representation of dictHR
    qubits = H.qubits
    for q in qubits:
        ket = random.choice(['0', '1'])
        state[q] = qt.ket(ket)

    return state

def measurement(H, probs, subset=None):
    """
    Given [H] and [probs], "measures" the state with probability of each
    state given by [probs]. Assumes standard tensor product ordering.

    Inputs
    ---------------------------------------------------------------------
    H: IsingH Hamiltonian
    probs: list of probability amplitudes for each composite qubit state
    subset: list of qubits of interest

    Output
    ---------------------------------------------------------------------
    m_state: measured state as a dictionary, {q0: ket(q0), q1: ket(q1), ...}
    """
    qubits = H.qubits
    # if we are not considered a subset, extract all qubits
    if subset is None:
        subset = qubits
    # "measures" from the pmf
    indices = [i for i in range(len(probs))]
    meas_i = random.choices(indices, probs)[0]

    # return classical state with tensor product ordering assumption
    return infer_classical_state(qubits, meas_i, subset)

# ------------------------------------------------------------
# frem annealing R partition state init schemes
# ------------------------------------------------------------
def c_diag_HR(HRdict, Rqubits):
    """
    Diag HR, find a random classical ground-state and return it

    Input
    --------------------------------------------------
    HRdict: dict--encoding of HR
    Rqubits: list--qubits in R partition

    Output
    --------------------------------------------------
    state: dict--contains qubit to qutip state mapping {q0: ket(q0), ...}
    """
    # create IsingH representation of dictHR
    HR = qa.probrep.IsingH(HRdict)
    qubits = HR.qubits
    # get non-zero indices of gs (which encode state)
    gs = HR.Hz_gs_info()['gs']
    nz_idx = np.nonzero(gs)

    # select a random entry from non-zero superposition indices
    gs_idx = random.choice(nz_idx[0])

    # return classical state with tensor product ordering assumption
    return infer_classical_state(qubits, gs_idx, Rqubits)

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def infer_classical_state(qubits, idx, subset=None):
    """
    Finds classical state of n qubits given index of basis state in H2^n.

    Input
    --------------------------------------------------
    qubits: list--qubits in numeric order
    idx: int--index of '1' in basis state of 2^n dimensional Hilbert space
    subset: optional list--only output state of qubits in subset list

    Output
    --------------------------------------------------
    state: dict--{q0: qutip ket 0 or ket 1, ...}
    """
    state = {}
    num_qubits = len(qubits)
    if subset is None:
        subset = qubits
    # find classical state of qubits with tensor product ordering assumption
    for (n, qubit) in enumerate(qubits):
        if qubit in subset:
            power = num_qubits - n
            mod = 2**power
            cut_off = mod / 2
            if idx % mod >= cut_off:
                state[qubit] = qt.ket('1')
            else:
                state[qubit] = qt.ket('0')

    return state

