"""FREM simulation protocol."""
# standard libarary imports
import sys
import time

# common library imports 
import numpy as np
import pandas as pd

# qanic library functions
sys.path.append("..")
import qanic as qa

def frem_sim(H, annparams, m_scheme, part_gen):
    """
    Tests the efficacy of FREM (forward-reverse error mitigation) annealing as compared to standard forward and reverse annealing.

    Arguments:
    * H - an IsingH hamiltonian
    * annparams - dictionary containing annealing parameters
        foTf - (float) forward anneal annealing time
        reTf - (float) reverse anneal forward annealing time
        reTr - (float) reverse anneal reverse annealing time
        frFTf - (float) frem F-part f annealing time
        frRTf - (float) frem R-part f annealing time
        frRTr - (float) frem R-part r annealing time
        s - (float) depth of reverse anneal
    * m_scheme - a function that specifies measurement scheme
        takes prob dist as input and resturns dict of {q0: state...}
    * part_gen - a generator that yields partitions of H; should yield data of the form
        (partition, critq_with_R, perRteam)
    --partition is a dictionary containing the Hamiltonian partition
        --> Rqubits is list containing which qubits belong to R parition
        --> HR is dictionary of reverse annealing (R team) Hamiltonian
        --> HF is dictionary of forward annealing (F team) Hamiltonian
    --critq_with_R stores whether user-specified "critical" qubit of H is an element of HR
    --perRteam gives percentage of mixed couplers (couplings between R and F) assigned to R

    Outputs:
    * rawdata: contains inputs, correct ground-state, forward result, reverse result, and frem results
    * run summary: contains inputs, correct gs, forward result, reverse result, and frem summary/ comparisons
    """
    # get annealing parameters and verify type
    foTf = annparams['foTf']
    reparams = ['reTf', 'reTr', 's']
    reTf, reTr, s = [annparams[key] for key in reparams]
    fremparams = ['frFTf', 'frRTf', 'frRTr']
    frFTf, frRTf, frRTr = [annparams[key] for key in fremparams]
    
    # initialize (pre pandas) data list
    listdata = []

    # try to diagonlize unless input wrong
    try:
        gsinfo = H.get_Hz_gs()
        gsprobs = (gsinfo['gs'].conj()*gsinfo['gs']).real
        ndegen = gsinfo['degen']
        # get indices of non-zero probs
        nzidxs = np.nonzero(gsprobs)
        # get non-zero gs entries
        nz_gsprobs = gsprobs[nzidxs]

        # append gs from diagonlization to data
        dictdatum = {'method': 'd', 'pgs': 1, 'gs_dist': nz_gsprobs, 'nRq': None, 'critq_R': None, 'pM_R': None}
        listdata.append(dictdatum)

    except TypeError:
        print("H is not an instance of DictRep.")
        raise
    except MemoryError:
        print("Hamiltonian too big to fit into memory.")
        raise

    # perform numerical forward anneal
    f_ann_params = {'t1': foTf, 'direction': 'f'}
    fstate = H.numeric_anneal(f_ann_params)
    fprobs = (fstate.conj()*fstate).real
    gs_fprobs = fprobs[nzidxs]
    # save result
    dictdatum = {'method': 'f', 'pgs': sum(gs_fprobs), 'gs_dist': gs_fprobs, 'nRq': None, 'critq_R': None, 'pM_R': None}
    listdata.append(dictdatum)

    # perform reverse anneal with init state given by forward anneal
    init_state = m_scheme(H.qubits, fprobs)
    r_ann_params = {'t1': reTf, 'direction': 'r', 't2': reTr, 's': s,
                    'init_state': init_state}
    rstate = H.numeric_anneal(r_ann_params)
    rprobs = (rstate.conj()*rstate).real
    gs_rprobs = rprobs[nzidxs]
    # save result
    dictdatum = {'method': 'r', 'pgs': sum(gs_rprobs), 'gs_dist': gs_rprobs, 'nRq': None, 'critq_R': None, 'pM_R': None}
    listdata.append(dictdatum)

    # perform FREM annealing over partitions specified by part_gen
    numpar = 0
    crit_exists = False
    for partinfo in part_gen:
        part, perRteam, critqinR = partinfo
        if critqinR is True or critqinR is False:
            crit_exists = True

        # perform the FREM anneal
        f_ann_params = {'t1': frFTf}
        r_ann_params = {'t1': frRTf, 'direction': 'r', 's': s,
                        't2': frRTf, 'init_state': init_state}
        fremstate = H.frem_anneal(f_ann_params, r_ann_params, part)
        frem_probs = (fremstate.conj()*fremstate).real 
        gs_fremprobs = frem_probs[nzidxs]

        # save data
        nRq = len(part['Rqubits'])
        dictdatum = {'method': 'frem', 'pgs': sum(gs_fremprobs), 'gs_dist': gs_fremprobs, 'nRq': nRq, 'critq_R': critqinR, 'pM_R': perRteam}
        listdata.append(dictdatum)

        numpar += 1

    # turn data into pandas DataFrame
    df = pd.DataFrame(listdata)

    # save runs separately
    f_run = df.loc[df['method'] == 'f']
    r_run = df.loc[df['method'] == 'r']
    frem_runs = df.loc[df['method'] == 'frem']

    # get bulk statistics to compare methods
    # find best gs prob of each method
    bf_pgs = f_run['pgs'].max()
    br_pgs = r_run['pgs'].max()
    bfrem_pgs = frem_runs['pgs'].max()
    # get % frem better than forward/reverse
    btf_runs = frem_runs[frem_runs.pgs > bf_pgs]
    btr_runs = frem_runs[frem_runs.pgs > br_pgs]
    p_btf = (len(btf_runs) / numpar) * 100
    p_btr = (len(btr_runs) / numpar) * 100
    # get avg gs_prob of ALL frem runs
    avgfrem_pgs = frem_runs['pgs'].mean()
    # of those that are better, find distribution of partition size
    btf_partdist = dict((btf_runs['nRq'].value_counts() / len(btf_runs)) * 100)
    btr_partdist = dict((btr_runs['nRq'].value_counts() / len(btr_runs)) * 100)
    # get stats on critical qubit
    if crit_exists:
        # % best frem result that is critical
        best_runs = frem_runs[np.isclose(frem_runs['pgs'], bfrem_pgs)]
        ncrit = sum([1 for x in best_runs if x is True])
        p_bc = (ncrit / len(best_runs)) * 100
        # avg pgs without critical
        avgwoc_pgs = frem_runs[frem_runs.critq_R == False]['pgs'].mean()
        # avg pgs with critical
        avgwc_pgs = frem_runs[frem_runs.critq_R == True]['pgs'].mean()
        # % of "better than" that are critical
        try:
            nfcrit = len(btf_runs[btf_runs.critq_R == True])
            p_cbtf = (nfcrit / len(btf_runs)) * 100
        except ZeroDivisionError:
            p_cbtf = 0
        try:
            nrcrit = len(btr_runs[btr_runs.critq_R == True])
            p_cbtr = (nrcrit / len(btr_runs)) * 100
        except ZeroDivisionError:
            p_cbtr = 0
    # if there isn't one listed, just put None to make outputfile consistent
    else:
        p_bc = None
        avgwoc_pgs = None
        avgwc_pgs = None
        p_cbtf = None
        p_cbtr = None

    # set-up file names for raw data and summary data
    date = time.strftime("%dd%mm%Yy-%Hh%Mm%Ss")
    infostr = "H-{H}_date-{date}".format(H=H.kind, date=date)
    summ_file = "summ_{}.txt".format(infostr)
    raw_file = "raw_{}.csv".format(infostr)

    # dump raw pandas df to raw file
    df.to_csv(raw_file, index=False)

    # print bulk/sumary data to summ_file
    with open(summ_file, 'w') as f:
        f.write("Input Data\n")
        f.write("---------------------------------------------------------\n")
        f.write("H kind: {}\n".format(H.kind))
        f.write("degeneracy: {}\n".format(ndegen))
        f.write("foTf: {}\n".format(foTf))
        f.write("reTf: {}\n".format(reTf))
        f.write("reTr: {}\n".format(reTr))
        f.write("frRTf: {}\n".format(frRTf))
        f.write("frRTr: {}\n".format(frRTr))
        f.write("s: {}\n".format(s))
        f.write("measurement scheme: {}\n".format(m_scheme.__name__))
        f.write("partition scheme: {}\n".format(part_gen.__name__))
        f.write("\n")
        f.write("Bulk Results\n")
        f.write("---------------------------------------------------------\n")
        f.write("forward gs prob: {}\n".format(bf_pgs))
        f.write("reverse gs prob: {}\n".format(br_pgs))
        f.write("best frem gs prob: {}\n".format(bfrem_pgs))
        f.write("number of frem partitions tried: {}\n".format(numpar))
        f.write("avg frem gs prob: {}\n".format(avgfrem_pgs))
        f.write("percent better than forward: {}\n".format(p_btf))
        f.write("part size dist of those better than forward: {}\n".format(btf_partdist))
        f.write("percent better than reverse: {}\n".format(p_btr))
        f.write("part size dist of those better than reverse: {}\n".format(btr_partdist))
        f.write("\n")
        f.write("Critical Qubit Results\n")
        f.write("---------------------------------------------------------\n")
        f.write("avg frem gs prob with critical qubit: {}\n".format(avgwc_pgs))
        f.write("avg frem gs prob without critical qubit: {}\n".format(avgwoc_pgs))
        f.write("Of best frem result, percent where R part contains critical qubit: {}\n".format(p_bc))
        f.write("Of better than forward, per containing critical: {}\n".format(p_cbtf))
        f.write("Of better than reverse, per containing critical: {}\n".format(p_cbtr))

    return df
