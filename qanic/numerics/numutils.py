


def KL_div(diag, others):
    """
    Compares the Kullback Liebler divergence of different probability density
    functions w.r.t. direct diagonlization.

    Inputs
    -------------------------------------------------------------------------------
    diag: list of probs of each state obtained via direct diagonlization of final H
    others: dict of the form {'name': probs}

    Outputs
    -------------------------------------------------------------------------------
    KL_div = {'name': KL value}
    """

    return {name: entropy(diag, others[name]).flatten() for name in others}

def random_partition(dictrep_H):
    """
    Creates a random partition of the H from 1 to n-1 qubits.

    Input
    ---------------------------------------------------------
    dictrep_H: a Hamiltonian represented in the dictrep class

    Output
    ---------------------------------------------------------
    Returns a dictionary with the following elements:
    *dictHR: random partition of H (paritions qubits and THEN
    looks at random set of couplers that involve these qubits)

    *Rqubits: qubits that 'belog' to HR partition

    *dictHF: the complement of dictHR w.r.t. H

    Note: not very efficient implementation, but I don't want to
    perform random choice of qubits and couplers at the same time,
    as this leads to issues of getting random HR couplings between
    only HF qubits...
    """
    H = dictrep_H.H
    Hgraph = dictrep_H.graph
    qubit_list = dictrep_H.qubits
    nqubits = dictrep_H.nqubits

    # find HR, a random partition of H
    rand_int = random.randint(1, nqubits-1)
    rand_qubits = random.sample(qubit_list, rand_int)
    dictHR = {}
    for qubit in rand_qubits:
        dictHR.update({(qubit, qubit): H[(qubit, qubit)]})
        neighbors = list(Hgraph.neighbors(qubit))
        rand_int = random.randint(1, len(neighbors))
        rand_neighbors = random.sample(neighbors, rand_int)
        for rn in rand_neighbors:
            rand_coupler = tuple(sorted((qubit, rn)))
            #dictHR.update({rand_coupler: H[rand_coupler]})
            dictHR[rand_coupler] = H[rand_coupler]

    # create the complementary dictionary of dictHR
    dictHF = {}
    for key, value in H.items():
        if key not in dictHR:
            dictHF[key] = value
            if key[0] == key[1]:
                dictHR[key] = 0
        else:
            if key[0] == key[1]:
                dictHF[key] = 0

    return {'HR': dictHR, 'Rqubits': rand_qubits, 'HF': dictHF}

def gs_calculator(H, etol=1e-8, stol=1e-12):
    """
    Computes the (possibly degenerate) ground state of an input
    Hamiltonian H.

    H: a QuTIP defined Hamitltonian
    gs: ground-state in QuTIP style
    """
    energies, states = H.eigenstates()
    degeneracy = 1
    lowest_E = energies[0]
    for n in range(1,len(energies)):
        if abs(energies[n]-lowest_E) < etol:
            degeneracy += 1

    gs = states[0]
    for n in range(1, degeneracy):
        gs = gs + states[n]
    gs = gs.tidyup(stol)
    gs = gs.unit()

    return gs

def make_numeric_schedule(discretization, **kwargs):
    """
    Creates an anneal_schdule to be used for numerical calculatins with QuTip.
    Returns times and svals associated with each time that [times, svals]
    that together define an anneal schedule.

    Inputs:
    discretization: determines what step size to use between points
    kwargs: dictionary that contains optional key-value args
    optional args: sa, ta, tp, tq
        sa: s value to anneal to
        ta: time it takes to anneal to sa
        tp: time to pause after reaching sa
        tq: time to quench from sa to s = 1
    """
    # Parse the kwargs input which encodes anneal schedule parameters
    try:
        ta = kwargs['ta']
    except KeyError:
        raise KeyError("An anneal schedule must at least include an anneal time, 'ta'")

    # extracts anneal parameter if present; otherwise, returns an empty string
    direction = kwargs.get('direction', '')
    sa = kwargs.get('sa', '')
    tp = kwargs.get('tp', '')
    tq = kwargs.get('tq', '')

    # turn discretization into samplerate multiplier
    samplerate = 1 / discretization

    if direction == 'forward' or direction == '':

        # if no sa present, create a standard forward anneal for ta seconds
        if not sa:
            # determine slope of anneal
            sa = 1; ta = kwargs['ta'];
            ma = sa / ta

            # create a list of times with (ta+1)*samplerate elements
            t = np.linspace(0, ta, int((ta+1)*samplerate))

            # create linear s(t) function
            sfunc = ma*t

        # if no pause present, anneal forward for ta to sa then quench for tq to s=1
        elif not tp:
            # determine slopes and y-intercept (bq) to create piece-wise function
            ma = sa / ta
            mq = (1 - sa) / tq
            bq = (sa*(ta + tq) - ta)/tq

            # create a list of times where sampling for anneal/ quench proportional to time there
            t = reduce(np.union1d, (np.linspace(0, ta, int((ta+1)*samplerate)),
                                    np.linspace(ta+.00001, ta+tq, int((ta+tq+1)*samplerate))))

            # create a piece-wise-linear (PWL) s(t) function defined over t values
            sfunc = np.piecewise(t, [t <= ta, (ta < t) & (t <= ta+tq)],
                                [lambda t: ma*t, lambda t: bq + mq*t])

        # otherwise, forward anneal, pause, then quench
        else:
            # determine slopes and y-intercept (bq) to create piece-wise function
            ma = sa / ta
            mp = 0
            mq = (1 - sa) / tq
            bq = (sa*(ta + tp + tq) - (ta + tp))/tq

            # create a list of times with samplerate elements from 0 and T = ta + tp + tq
            t = reduce(np.union1d, (np.linspace(0, ta, int((ta+1)*samplerate)),
                                    np.linspace(ta+.00001, ta+tp, int((ta+tp+1)*samplerate)),
                                    np.linspace(ta+tp+.00001, ta+tp+tq, int((ta+tp+tq+1)*samplerate))))

            # create a piece-wise-linear (PWL) s(t) function defined over t values
            sfunc = np.piecewise(t, [t <= ta, (ta < t) & (t <= ta+tp),(ta+tp < t) & (t <= ta+tp+tq)],
                                 [lambda t: ma*t, lambda t: sa, lambda t: bq + mq*t])

    elif direction == 'reverse':
        # if no pause, do standard 'reverse' anneal
        if not tp:
            # determine slopes and y-intercept (bq) to create piece-wise function
            ma = (sa - 1) / ta
            ba = 1
            mq = (1 - sa) / tq
            bq = (sa*(ta + tq) - ta)/tq

            # create a list of times where sampling for anneal/ quench proportional to time there
            t = reduce(np.union1d, (np.linspace(0, ta, int((ta+1)*samplerate)),
                                    np.linspace(ta+.00001, ta+tq, int((ta+tq+1)*samplerate))))

            # create a piece-wise-linear (PWL) s(t) function defined over t values
            sfunc = np.piecewise(t, [t <= ta, (ta < t) & (t <= ta+tq)],
                                [lambda t: ba + ma*t, lambda t: bq + mq*t])

        # otherwise, include pause
        else:
            # determine slopes and y-intercept (bq) to create piece-wise function
            ma = (sa - 1) / ta
            ba = 1
            mp = 0
            mq = (1 - sa) / tq
            bq = (sa*(ta + tp + tq) - (ta + tp))/tq

            # create a list of times with samplerate elements from 0 and T = ta + tp + tq
            t = reduce(np.union1d, (np.linspace(0, ta, int((ta+1)*samplerate)),
                                    np.linspace(ta+.00001, ta+tp, int((ta+tp+1)*samplerate)),
                                    np.linspace(ta+tp+.00001, ta+tp+tq, int((ta+tp+tq+1)*samplerate))))

            # create a piece-wise-linear (PWL) s(t) function defined over t values
            sfunc = np.piecewise(t, [t <= ta, (ta < t) & (t <= ta+tp),(ta+tp < t) & (t <= ta+tp+tq)],
                                 [lambda t:ba + ma*t, lambda t: sa, lambda t: bq + mq*t])



    return [t, sfunc]


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

def dict_to_qutip(dictrep, encoded_params=None):
    """
    Takes a DictRep Ising Hamiltonian and converts it to a QuTip Ising Hamiltonian.
    Encoded params must be passed if dictrep weights are variables (abstract) and
    not actual numbers.
    """
    # make useful operators
    sigmaz = qto.sigmaz()
    nqbits = len(dictrep.qubits)
    zeros = [qto.qzero(2) for m in range(nqbits)]
    finalH = qt.tensor(*zeros)

    for key, value in dictrep.H.items():
        if key[0] == key[1]:
            if encoded_params:
                finalH += encoded_params[value]*nqubit_1pauli(sigmaz, key[0], nqbits)
            else:
                finalH += value*nqubit_1pauli(sigmaz, key[0], nqbits)
        else:
            if encoded_params:
                finalH += encoded_params[value]*nqubit_2pauli(sigmaz, sigmaz, key[0], key[1], nqbits)
            else:
                finalH += value*nqubit_2pauli(sigmaz, sigmaz, key[0], key[1], nqbits)

    return finalH

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

    sch_ABfuncs = {'A(t)': sch_Afunc, 'B(t)': sch_Bfunc}

    return sch_ABfuncs

def loadAandB(file="processor_annealing_schedule_DW_2000Q_2_June2018.csv"):
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

def get_numeric_H(dictrep):
    HZ = dict_to_qutip(dictrep)
    nqbits = len(dictrep.qubits)
    HX = sum([nqubit_1pauli(qto.sigmax(), m, nqbits) for m in range(nqbits)])

    H = {'HZ': HZ, 'HX': HX}

    return H
