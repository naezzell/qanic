"""Collection of utilities to make isingh run smoothly."""

# default python imports
from functools import reduce
import itertools
from collections import OrderedDict
import random

# commonly used external python packages
import numpy as np
from scipy.stats import entropy
import pandas as pd
import networkx as nx

# OCEAN and QuTip imports
from dwave.system.samplers import DWaveSampler
import qutip as qt
import qutip.operators as qto

# *****************************************************************
# *****************************************************************
#           Top-level (for isingH) utilities
# *****************************************************************
# *****************************************************************

# *****************************************************************
#           Hamiltonian conversion utilities
# *****************************************************************
def get_dwave_H(isingH):
    """wrapper to convert input H (of valid type) to d-wave processable H"""
    if isinstance(isingH, dict):
        return dictH_to_dwaveH(isingH)

def get_numeric_H(isingH):
    """
    Wrapper to turn input H of valid type to QuTip H.
    Input: isingH hamiltonian 
    Output: [Hz, Hx] where Hz is QuTip rep of isingH
    and Hx is QuTip rep of D-Wave Hx
    """
    if isinstance(isingH, dict):
        return dict_to_qutip(isingH)

def get_networkx_H(isingH):
    """
    Wrapper to turn input Hz of valid type to networkx H.
    Input: isingH hamiltonian 
    Output: Hz as networkx graph
    """
    if isinstance(isingH, dict):
        return dict_to_networkx(isingH)

# *****************************************************************
#           Anneal Schedule Utilities
# *****************************************************************

def make_dwave_schedule(t1 = 20, direction = 'f', sp = 1, tp = 0, t2 = 0):
    """
    Creates an annealing schedule that D-Wave can interpret.  

    Input Arguments (all floats)
    --------------------
    t1: anneal length (in micro seconds) from s_init to sp
    direction: 'f' or 'r' for forward (s_init = 0) or reverse (s_init = 1)
    sp: intermediate s value (s-prime) reached after annealing for t1
    tp: duration of pause after reaching sp
    t2: anneal length from sp to s_final = 1

    Output
    --------------------
    anneal_schedule: a list of lists of the form [[ti, si], [t1, s1], ... [tf, 1]]
    """
    # get DWave sampler as set-up in dwave config file and save relevant properites
    sampler = DWaveSampler()
    mint, maxt = sampler.properties["annealing_time_range"]

    # first, ensure that quench slope is within chip bounds
    #if t2 != 0 and t2 < mint:
    #    raise ValueError("Minimum value of t2 possible by chip is: {mint}".format(mint=mint))
    # now, check that anneal time is not too long (quench has equivalent anneal time)
    if (t1 + tp + t2) > maxt:
        raise ValueError("Maximum allowed anneal time is: {maxt}.".format(maxt=maxt))
    #make sure s is valid
    if sp > 1.0:
        raise ValueError("s cannot exceed 1.")

    if t1 < mint or t2-sp < mint:
        raise ValueError("minimum anneal time is: {mint}.".format(mint=mint))
    
    #if s = 1, stop the anneal after t1 micro seconds
    if sp == 1:
        return [[0, 0], [t1, 1]]

    #otherwise, create anneal schedule according to times/ s
    if direction.lower()[0] == 'f':
        sch = [[0, 0], [t1, sp], [t1+tp, sp], [t1+tp+t2, 1]]
    elif direction.lower()[0] == 'r':
        sch = [[0, 1], [t1, sp], [t1+tp, sp], [t1+tp+t2, 1]]

    #remove duplicates while preserving order (for example if tp = 0)
    ann_sch = list(map(list, OrderedDict.fromkeys(map(tuple, sch))))

    return ann_sch

def make_numeric_schedule(ann_params={'t1': 1}):
    """
    Creates an anneal_schdule to be used for numerical calculatins with QuTip.
    Returns times and svals associated with each time [times, svals]
    for each (discrete) unit of time during the anneal. 

    Input Arguments (all floats)
    --------------------
    *ann_params = {'t1': t1, ...}
    t1: anneal length (in micro seconds) from s_init to sp
    direction: 'f' or 'r' for forward (s_init = 0) or reverse (s_init = 1)
    sp: intermediate s value (s-prime) reached after annealing for t1
    tp: duration of pause after reaching sp
    t2: anneal length from sp to s_final = 1
    disc: discretization used between times 

    Output
    --------------------
    anneal_schedule: a list of lists of the form [[t0, s0], [t1, s1], ... [tn, 1]]
    """
    # pre-process arguments, setting to default value if not present
    t1 = ann_params.get('t1', 1)
    direction = ann_params.get('direction', 'f')
    sp = ann_params.get('sp', 1)
    tp = ann_params.get('tp', 0)
    t2 = ann_params.get('t2', 0)
    disc = ann_params.get('disc', 0.01)
    
    # turn discretization into samplerate multiplier and 'buffer' between changes
    buf = .001 * disc

    if direction.lower()[0] == 'f':
        # if sp = 1, create a standard forward anneal for t1
        if sp == 1:
            # determine slope of anneal
            ma = sp / t1

            # create a list of times 0, disc, 2*disc, ..., t1
            t = np.linspace(0, t1, int(t1 / disc) + 1)

            # create linear s(t) function
            sfunc = ma * t

        # if no pause present, anneal forward for t1 to sa then quench for t2 to s=1
        elif tp == 0:
            # determine slopes and y-intercept (bq) to create piece-wise function
            ma = sp / t1
            mq = (1 - sp) / t2
            bq = (sp * (t1 + t2) - t1) / t2

            # create a list of times where sampling for anneal/ quench proportional to time there
            t = reduce(np.union1d, (np.linspace(0, t1, int(t1 / disc) + 1),
                                    np.linspace(t1 + buf, t1 + t2, int((t1 + t2) / disc) + 1)))

            # create a piece-wise-linear (PWL) s(t) function defined over t values
            sfunc = np.piecewise(t, [t <= t1, (t1 < t) & (t <= t1 + t2)],
                                [lambda t: ma * t, lambda t: bq + mq * t])

        # otherwise, forward anneal, pause, then quench
        else:
            # determine slopes and y-intercept (bq) to create piece-wise function
            ma = sp / t1
            mq = (1 - sp) / t2
            bq = (sp * (t1 + tp + t2) - (t1 + tp)) / t2

            # create a list of times with samplerate elements from 0 and T = t1 + tp + t2
            t = reduce(np.union1d, (np.linspace(0, t1, int(t1 / disc) + 1),
                                    np.linspace(t1 + buf, t1 + tp, int((t1 + tp) / disc) + 1),
                                    np.linspace(t1 + tp + buf, t1 + tp + t2, int((t1 +tp + t2) / disc) + 1)))

            # create a piece-wise-linear (PWL) s(t) function defined over t values
            sfunc = np.piecewise(t, [t <= t1, (t1 < t) & (t <= t1 + tp),(t1 + tp < t) & (t <= t1 +tp + t2)],
                                 [lambda t: ma * t, lambda t: sp, lambda t: bq + mq * t])

    elif direction.lower()[0] == 'r':
        # if no pause, do standard 'reverse' anneal
        if tp == 0:
            # determine slopes and y-intercept (bq) to create piece-wise function
            ma = (sp - 1) / t1
            ba = 1
            mq = (1 - sp) / t2
            bq = (sp * (t1 + t2) - t1) / t2

            # create a list of times where sampling for anneal/ quench proportional to time there
            t = reduce(np.union1d, (np.linspace(0, t1, int(t1/disc) + 1),
                                    np.linspace(t1 + buf, t1 + t2, int((t1+t2)/ disc) + 1)))

            # create a piece-wise-linear (PWL) s(t) function defined over t values
            sfunc = np.piecewise(t, [t <= t1, (t1 < t) & (t <= t1 + t2)],
                                [lambda t: ba + ma * t, lambda t: bq + mq * t])

        # otherwise, include pause
        else:
            # determine slopes and y-intercept (bq) to create piece-wise function
            ma = (sp - 1) / t1
            ba = 1
            mq = (1 - sp) / t2
            bq = (sp * (t1 + tp + t2) - (t1 + tp)) / t2

            # create a list of times with samplerate elements from 0 and T = t1 + tp + t2
            t = reduce(np.union1d, (np.linspace(0, t1, int(t1 / disc) + 1),
                                    np.linspace(t1 + buf, t1 + tp, int((t1 + tp) / disc) + 1),
                                    np.linspace(t1 + tp + buf, t1 + tp + t2, int((t1 + tp + t2) / disc) + 1)))

            # create a piece-wise-linear (PWL) s(t) function defined over t values
            sfunc = np.piecewise(t, [t <= t1, (t1 < t) & (t <= t1 + tp),(t1 + tp < t) & (t <= t1 + tp + t2)],
                                 [lambda t:ba + ma * t, lambda t: sp, lambda t: bq + mq * t])

    return [t, sfunc]

def loadAandB(file="/home/nic/inputdata/processor_annealing_schedule_DW_2000Q_2_June2018.csv"):
    """
    Loads in A(s) and B(s) data from chip and interpolates using QuTip's
    cubic-spline function. Useful for numerical simulations.

    Returns (as list in this order):
    svals: numpy array of discrete s values for which A(s)/B(s) are defined
    Afunc: interpolated A(s) function
    Bfunc: interpolated B(s) function
    """

    Hdata = pd.read_csv(file)
    # pd as in pandas Series form of data
    pdA = Hdata['A(s) (GHz)']
    pdB = Hdata['B(s) (GHz)']
    pds = Hdata['s']
    Avals = np.array(pdA)
    Bvals = np.array(pdB)
    svals = np.array(pds)

    processor_data = {'svals': svals, 'Avals': Avals, 'Bvals': Bvals}

    return processor_data

def time_interpolation(schedule, processor_data):
    """
    Interpolates the A(s) and B(s) functions in terms of time in accordance with an
    annealing schedule s(t). Returns cubic-splines amenable to use with QuTip.
    """

    svals = processor_data['svals']
    Avals = processor_data['Avals']
    Bvals = processor_data['Bvals']

    # interpolate Avals and Bvals into a cubic spline function
    Afunc = qt.interpolate.Cubic_Spline(svals[0], svals[-1], Avals)
    Bfunc = qt.interpolate.Cubic_Spline(svals[0], svals[-1], Bvals)

    # now, extract s(t)
    times = schedule[0]
    sprogression = schedule[1]

    # interpolate A/B funcs with respect to time with s(t) relationship implicitly carried through
    sch_Afunc = qt.interpolate.Cubic_Spline(times[0], times[-1], Afunc(sprogression))
    sch_Bfunc = qt.interpolate.Cubic_Spline(times[0], times[-1], Bfunc(sprogression))

    sch_ABfuncs = [sch_Afunc, sch_Bfunc]

    return sch_ABfuncs

# *****************************************************************
#            Useful QuTip Self-written Utilities
# *****************************************************************
def gs_calculator(H, etol=1e-8, stol=1e-12):
    """
    Returns the ground-state energy/eigenstates/degeneracy.
    Inputs
    --------------------
    H: the input Hamiltonian of the QuTip variety
    etol: energy tolerance, cut-off for 'numerical equivalence'
    stol: small tolerance, anything less is set to 0 during normalization

    Output
    --------------------
    (super-position gs state, energy, degeneracy, [list of groundstates])
    """
    energies, states = H.eigenstates()
    degeneracy = 1
    lowest_E = energies[0]
    for n in range(1,len(energies)):
        if abs(energies[n]-lowest_E) < etol:
            degeneracy += 1

    # this works because listed in order of inc energy
    gs = states[0]
    statelist = [gs]
    for n in range(1, degeneracy):
        gs = gs + states[n]
        statelist.append(states[n])

    gs = gs.tidyup(stol)
    gs = gs.unit()

    gs_info = {'gs': gs, 'E': lowest_E, 'degen': degeneracy, 'statelist': statelist}

    return gs_info

def qto_to_npa(state):
    """
    Converts QuTip object that encodes a quantum state
    into a numpy array.
    """
    return np.array([amp[0][0] for amp in state])

# *****************************************************************
# *****************************************************************
#           helper functions for top-level utilities
# *****************************************************************
# *****************************************************************

# *****************************************************************
#           Hamiltonian conversion helper functions
# *****************************************************************
def dictH_to_dwaveH(dictH):
    """Converts dict H to D-wave H"""
    h_vals = {}
    j_vals = {}
    for key, value in dictH.items():
        if key[0] == key[1]:
            h_vals[key[0]] = value
        else:
            j_vals[key] = value

    return [h_vals, j_vals]

def dict_to_qutip(dictH):
    """
    Converts ising H to QuTip Hz and makes QuTip Hx of D-Wave
    in the process to avoid redundant function calls.
    """
    # make useful operators
    sigmaz = qto.sigmaz()
    nqbits = len([key for key in dictH.keys() if key[0] == key[1]])
    Hx = sum([nqubit_1pauli(qto.sigmax(), m, nqbits) for m in range(nqbits)])
    
    zeros = [qto.qzero(2) for m in range(nqbits)]
    Hz = qt.tensor(*zeros)

    for key, value in dictH.items():
        if key[0] == key[1]:
            Hz += value*nqubit_1pauli(sigmaz, key[0], nqbits)
        else:
            Hz += value*nqubit_2pauli(sigmaz, sigmaz, key[0], key[1], nqbits)

    return [Hz, Hx]

def dict_to_networkx(dictH):
    """
    Converts dict ising Hz into networkx Hz.
    """
    G = nx.Graph() # init graph
    for key, value in dictH.items():
        if key[0] != key[1]:
            G.add_edge(key[0], key[1], weight = value)
        else:
            G.add_node(key[0], weight = value)

    return G

# *****************************************************************
#           Numeric Hamiltonian helper functions
# *****************************************************************

def nqubit_1pauli(pauli, i, n):
    """
    Creates a single-qubit pauli operator on qubit i (0-based)
    that acts on n qubits. (padded with identities).

    For example, pauli = Z, i = 1 and n = 3 gives:
    Z x I x I, an 8 x 8 matrix
    """
    #create identity padding
    iden1 = [qto.identity(2) for j in range(i)]
    iden2 = [qto.identity(2) for j in range(n-i-1)]

    #combine into total operator list that is in proper order
    oplist = iden1 + [pauli] + iden2

    #create final operator by using tensor product on unpacked operator list
    operator = qt.tensor(*oplist)

    return operator

def nqubit_2pauli(ipauli, jpauli, i, j, n):
    """
    Creates a 2 qubit x/y/z pauli operator on qubits i,j
    with i < j that acts on n qubits in total.

    For example, ipauli = Y, jpauli = Z, i = 1, j = 2 and n = 3 gives:
    Y x Z x I, an 8 x 8 matrix
    """
    #create identity padding
    iden1 = [qto.identity(2) for m in range(i)]
    iden2 = [qto.identity(2) for m in range(j-i-1)]
    iden3 = [qto.identity(2) for m in range(n-j-1)]

    #combine into total operator list
    oplist = iden1 + [ipauli] + iden2 + [jpauli] + iden3

    # apply tensor product on unpacked oplist
    operator = qt.tensor(*oplist)

    return operator
