import sys
import random
from itertools import (chain, combinations)

def all_parts(H, scheme, critq='Unknown'):
    """
    This generates all possible R-favored partitions of a Hamiltonian
    in preparation for an exhaustive FREM simulation.

    Inputs:
    H - an IsingH Hamiltonian
    scheme - 'all_R', 'all_F' or 'random' to assign mixed couplers
    critq - index of a known 'critical' qubit (can be str)
    * if it's unclear which should be critical, leave as default 
    value 'Unknown'
    * if all qubits are equally as important, then the correct answer 
    is None or False

    Outputs:
    (partition, critq_with_R, perRteam)
    * partition is a dictionary containing the Hamiltonian partition
        --> Rqubits is list containing which qubits belong to R parition
        --> HR is dictionary of reverse annealing (R team) Hamiltonian
        --> HF is dictionary of forward annealing (F team) Hamiltonian
    * critq_with_R stores whether user-specified "critical" qubit of
    H is an element of HR
    * perRteam gives percentage of mixed couplers assigned to R
    """
    # iterate over the powerset of possible partitions
    for qsubset in powerset(H.qubits):
        Rqubits = list(qsubset)
        Fqubits = list(set(H.qubits) - set(qsubset))
        # is critical qubit part part of R partition?
        if critq != 'Unknown' and critq is not None:
            critq_with_R = (critq in qsubset)
        else:
            critq_with_R = critq
        # do not include the empty set or entire set as a viable partition
        if len(qsubset) == 0 or len(qsubset) == len(H.qubits):
            continue
        else:
            # assign couplers
            coupler_info = assign_couplers(H, Rqubits, Fqubits, scheme)
            dictHR, dictHF, perRteam = coupler_info
            # assign qubits
            for q in H.qubits:
                # R team assignments
                if q in Rqubits:
                    dictHR[(q, q)] = H[(q, q)]
                    dictHF[(q, q)] = 0
                else:
                    # F team assignments
                    dictHR[(q, q)] = 0
                    dictHF[(q, q)] = H[(q, q)]

        partition = {'HR': dictHR, 'HF': dictHF, 'Rqubits': Rqubits}
        yield (partition, perRteam, critq_with_R)

def R1qubit(H, scheme, critq='Unknown'):
    return nqubit_part(H, scheme, 1, critq)

def R2qubits(H, scheme, critq='Unknown'):
    return nqubit_part(H, scheme, 2, critq)

def R3qubits(H, scheme, critq='Unknown'):
    return nqubit_part(H, scheme, 3, critq)

def R4qubits(H, scheme, critq='Unknown'):
    return nqubit_part(H, scheme, 4, critq)

def R5qubits(H, scheme, critq='Unknown'):
    return nqubit_part(H, scheme, 5, critq)

def R6qubits(H, scheme, critq='Unknown'):
    return nqubit_part(H, scheme, 6, critq)

def R7qubits(H, scheme, critq='Unknown'):
    return nqubit_part(H, scheme, 7, critq)

def R8qubits(H, scheme, critq='Unknown'):
    return nqubit_part(H, scheme, 8, critq)

def R9qubits(H, scheme, critq='Unknown'):
    return nqubit_part(H, scheme, 9, critq)

        
# ***************************************************************************
# Internal Helper Functions
# ***************************************************************************
def assign_couplers(H, Rqubits, Fqubits, scheme):
    """
    Assigns a team (R or F) to each coupler

    Inputs:
    ------------------------------------------------------------
    H.couplers - a dictionary containing all the couplings
    Rqubits - a list containing all qubits on team R
    Fqubits - a list containing all qubits on team F
    scheme - specifies 'all_R', 'all_F', or 'random'

    Output:
    ------------------------------------------------------------
    Outputs HR and HF coupler dictionaries in accordance with scheme.
    HRcouplers = {(qi, qj): wij, ...}
    HFcouplers = {(qi, qj): wij, ...}
    perRteam - percernt of mixed partitions assigned to team R
    """
    HRcouplers = {}
    HFcouplers = {}
    MtoRTeam = 0
    m_couplers = 0
    for coupler, bias in H.Hz.items():
        # if both qubits are in one team, assignment is trivial
        if coupler[0] in Rqubits and coupler[1] in Rqubits:
            HRcouplers[coupler] = bias
            HFcouplers[coupler] = 0
        elif coupler[0] in Fqubits and coupler[1] in Fqubits:
            HRcouplers[coupler] = 0
            HFcouplers[coupler] = bias
        # otherwise, mixed coupler assigned by scheme
        else:
            m_couplers += 1
            if scheme == 'all_R':
                HRcouplers[coupler] = bias
                HFcouplers[coupler] = 0
                MtoRTeam += 1
            elif scheme == 'all_F':
                HRcouplers[coupler] = 0
                HFcouplers[coupler] = bias
            elif scheme == 'random':
                if random.uniform(0, 1) > 0.5:
                    HRcouplers[coupler] = bias
                    HFcouplers[coupler] = 0
                    MtoRTeam += 1
                else:
                    HRcouplers[coupler] = 0
                    HFcouplers[coupler] = bias
            else:
                raise ValueError("scheme {} is not supported".format(scheme))

    perRteam = (MtoRTeam / m_couplers) * 100
    return (HRcouplers, HFcouplers, perRteam)

def nqubit_part(H, scheme, n, critq='Unknown'):
    """
    Returns HR partitions of size n.
    """
    # generate them all (not efficient)
    allparts = all_parts(H, scheme, critq)

    nparts = []
    for p in allparts:
        HRsize = len(p[0]['Rqubits'])
        if HRsize == n:
            nparts.append(p)

    return nparts

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
