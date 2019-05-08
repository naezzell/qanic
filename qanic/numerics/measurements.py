# imports
import numpy as np


def ml_measurement(probs, num_qubits, qubits=None):
    """
    Finds the most likely measurement outcome predicted from discrete probability
    distribution of n-qubits. (Relies on probs being canonicaly tensor product ordering).

    Inputs
    ---------------------------------------------------------------------
    probs: list of probability amplitudes for each composite qubit state

    Output
    ---------------------------------------------------------------------
    ml_state: reconstructed most likely state as a string
    """
    state = []

    # gets the most likely state
    max_idx = np.argmax(probs)

    # finds the correct state of each qubit with a tensor product ordering assumption
    for n in range(num_qubits):
        power = num_qubits - n
        mod = 2**power
        cut_off = mod / 2
        if (max_idx) % mod >= cut_off:
            state.append(1)
        else:
            state.append(0)

    # returns only the qubits of interest if a subset of qubits is specified
    sub_system_state = []
    if qubits:
        for (idx, qs) in enumerate(state):
            if idx in qubits:
                sub_system_state.append(qs)
        state = sub_system_state

    return state
