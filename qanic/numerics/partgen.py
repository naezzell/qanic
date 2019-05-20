import sys
from itertools import (chain, combinations)
#from qanic.probrep.dictrep import DictRep


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def all_R(H, critq='Unknown'):
    """
    This generates all possible R-favored partitions of a Hamiltonian in preparation for an
    exhaustive FREM simulation.

    Inputs:
    H - a DictRep Hamiltonian
    critq - index (can be however you indexed your qubit in DictRep, so string or int or w.e.) of
    a known 'critical' qubit
    * if it's unclear which should be critical, leave as default value 'Unknown'
    * if all qubits are equally as important, then the correct answer is None or False

    Outputs:
    (partition, critq_with_R, perRteam)
    * partition is a dictionary containing the Hamiltonian partition
        --> Rqubits is list containing which qubits belong to R parition
        --> HR is dictionary of reverse annealing (R team) Hamiltonian
        --> HF is dictionary of forward annealing (F team) Hamiltonian
    * critq_with_R stores whether user-specified "critical" qubit of H is an element of HR
    * perRteam gives percentage of mixed couplers (couplings between R and F) assigned to R
    """
    # the percent of mixed (M) couplers assigned to R team is 100 for R_all geneartor
    perRteam = 100

    # iterate over the powerset of possible qubit combinations (i.e. all partitions)
    for qsubset in powerset(H.qubits):
        Rqubits = list(qsubset)
        # determine whether critical qubit (if applicable) is part of R partition
        if critq != 'Unknown' and critq is not None:
            critq_with_R = (critq in qsubset)
        else:
            critq_with_R = critq
        # do not include the empty set or entire set as a viable partition
        if len(qsubset) == 0 or len(qsubset) == len(H.qubits):
            continue
        else:
            dictHR = {}
            dictHF = {}
            HFcouplers = H.couplers[::]
            # iterate over the qubits
            for q in H.qubits:
                if q in qsubset:
                    dictHR[(q, q)] = H[(q, q)]
                    dictHF[(q, q)] = 0
                    # assign all couplers that contain q to 'R team'
                    for c in H.couplers:
                        if q in c:
                            dictHR[c] = H[c]
                            dictHF[c] = 0
                            try:
                                HFcouplers.remove(c)
                            except ValueError:
                                pass
                else:
                    # assign everything else to 'F team'
                    dictHF[(q, q)] = H[(q, q)]
                    dictHR[(q, q)] = 0
                    for c in HFcouplers:
                        dictHF[c] = H[c]
                        dictHR[c] = 0

        partition = {'HR': dictHR, 'HF': dictHF, 'Rqubits': Rqubits}

        yield (partition, perRteam, critq_with_R)
