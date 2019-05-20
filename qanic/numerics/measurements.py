# imports
import numpy as np


def ml_measurement(qubits, probs, subset=None):
    """
    Finds the most likely measurement outcome predicted from discrete probability
    distribution of n-qubits. (Relies on probs having canonical tensor product ordering).

    Inputs
    ---------------------------------------------------------------------
    qubits: list of all qubits in problem (increasing order)
    probs: list of probability amplitudes for each composite qubit state
    subset: list of qubits of interest

    Output
    ---------------------------------------------------------------------
    ml_state: reconstructed most likely state as a dictionary of {q0: '1', q1: '0', ...}
    """
    state = {}
    # if we are not considered a subset, extract all qubits
    if subset is None:
        subset = qubits
    # gets the most likely state
    max_idx = np.argmax(probs)
    # gets number of qubits
    num_qubits = len(qubits)

    # finds the correct state of each qubit with a tensor product ordering assumption
    for (n, qubit) in enumerate(qubits):
        if qubit in subset:
            power = num_qubits - n
            mod = 2**power
            cut_off = mod / 2
            if (max_idx) % mod >= cut_off:
                state[qubit] = '1'
            else:
                state[qubit] = '0'
                
    return state
