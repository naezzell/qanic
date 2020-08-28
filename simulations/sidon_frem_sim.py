"""FREM simulation protocol."""
# internal import
import time
import itertools

# data-handling imports
import h5py
import numpy as np
import pandas as pd

#internal imports
import qanic as qa
from qanic.numerics import hamgen

# number of trials needed for 95% confidence on 5% interval
ss95_5 = {3: 23, 4: 43, 5: 66, 6: 92, 7: 117,
               8: 142, 9: 165, 10: 186}

def sidon_frem_sim(Hsizes, hbool, Tvals, svals, discs, r_init, frem_init, part_scheme, include_zero, hx_ver, filename, datadir = '', Htrials=ss95_5, itrials=1, store_raw=False, profile=False):
    """
    Tests the efficacy of FREM (forward-reverse error mitigation) annealing
    compared to forward and reverse annealing.
    """
    ntrials = 0
    for n in Hsizes:
        ntrials += (len(Tvals) * len(svals) * len(discs) * Htrials[n] * itrials)

    # add input/output datasets to hdf5 file
    with h5py.File(filename, 'w') as f:
        # store meta-data for this entire simulation run
        ig = f.create_group('Inputs')
        ig.attrs['Mixed_Couplings'] = 'Forward'
        ig.attrs['F_Sch_Form'] = '[[0, 0], [T, 1]]'
        ig.attrs['R_Sch_Form'] = '[[0, 1], [T/2, s], [T, 1]]'
        if hbool is True:
            ig.attrs['H_kind'] = 'SKn couplings and h values'
        else:
            ig.attrs['H_kind'] = 'SKn couplings and h = 0'
        ig.attrs['part_scheme'] = part_scheme.__name__
        ig.attrs['r_init_scheme'] = r_init.__name__
        ig.attrs['frem_init_scheme'] = frem_init.__name__
        # store the inputs we iterate over_
        ig.create_dataset('Hsizes', (ntrials, ), dtype='i')
        ig.create_dataset('Tvals', (ntrials, ), dtype='f')
        ig.create_dataset('svals', (ntrials, ), dtype='f')
        ig.create_dataset('discs', (ntrials, ), dtype='f')
        # store datasets for the outputs
        og = f.create_group('Outputs')
        og.create_dataset('f_probs', (ntrials,), dtype='f')
        og.create_dataset('r_probs', (ntrials,), dtype='f')
        og.create_dataset('bfrem_probs', (ntrials,), dtype='f')
        og.create_dataset('bfrem_psizes', (ntrials,), dtype='f')
        og.create_dataset('avgfrem_probs', (ntrials,), dtype='f')
        og.create_dataset('p_btf', (ntrials,), dtype='f')
        og.create_dataset('p_btr', (ntrials,), dtype='f')
        if profile:
            og.create_dataset('rtime', (ntrials,), dtype='f')
            #og.create_dataset('pmem', (ntrials,), dtype='f')
    #TODO: add critical qubit datasets that take variable type
    k_loop = 0
    for n in Hsizes:
        for j in range(Htrials[n]):
            if include_zero is True:
                dictH, _ = hamgen.sidon0Kn(n, hbool)
            else:
                dictH, _ = hamgen.sidonKn(n, hbool)
            H = qa.probrep.IsingH(dictH, hx_ver)
            # generate the partitions of H
            part_list = list(part_scheme(H, 'all_F'))
            # cartesian product over annealing parameters
            for annparams in itertools.product(*[Tvals, svals, discs]):
                if profile:
                    tic = time.perf_counter()

                T, sp, disc = annparams
                # create the anneal schedules
                fsch = [[0, 0], [T, 1]]
                rsch = [[0, 1], [T / 2, sp], [T, 1]]

                # init data container for each sim
                listdata = []

                # diagonlize unless input wrong or uses too much memory
                try:
                    gsinfo = H.Hz_gs_info()
                    gsprobs = (gsinfo['gs'].conj()*gsinfo['gs']).real
                    ndegen = gsinfo['degen']
                    # get indices of non-zero probs
                    nzidxs = np.nonzero(gsprobs)
                    # get non-zero gs entries
                    nz_gsprobs = gsprobs[nzidxs]

                    # append gs from diagonlization to data
                    dictdatum = {'method': 'd', 'pgs': sum(nz_gsprobs),
                                 'gs_dist': nz_gsprobs, 'nRq': None,
                                 'critq_R': None, 'pM_R': None}
                    listdata.append(dictdatum)

                except TypeError:
                    print("H is not an instance of IsingH.")
                    raise
                except MemoryError:
                    print("Hamiltonian too big to fit into memory.")
                    raise

                # perform numerical forward anneal
                fprobs = H.numeric_anneal(fsch, disc)
                gs_fprobs = fprobs[nzidxs]
                # save result
                dictdatum = {'method': 'f', 'pgs': sum(gs_fprobs), 'gs_dist': gs_fprobs, 'nRq': None, 'critq_R': None, 'pM_R': None}
                listdata.append(dictdatum)

                for jj in range(itrials):
                # perform reverse anneal using r_init for state
                    init_state = r_init(H, fprobs)
                    rprobs = H.numeric_anneal(rsch, disc, init_state)
                    gs_rprobs = rprobs[nzidxs]
                    # save result
                    dictdatum = {'method': 'r', 'pgs': sum(gs_rprobs),
                                 'gs_dist': gs_rprobs, 'nRq': None,
                                 'critq_R': None, 'pM_R': None}
                    listdata.append(dictdatum)

                    # perform FREM annealing over partitions
                    numpar = 0
                    crit_exists = False
                    for partinfo in part_list:
                        part, perRteam, critqinR = partinfo
                        # could be None or 'Unknown', so we check if it is a bool
                        if critqinR is True or critqinR is False:
                            crit_exists = True

                        # perform the FREM anneal
                        R_init = frem_init(part['HR'], part['Rqubits'])
                        frem_probs = H.frem_anneal(fsch, rsch, R_init, part, disc)
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
                    # num qubits in best partition
                    nq_bp = df.iloc[frem_runs['pgs'].idxmax()]['nRq']
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
                    raw_file = "{ddir}raw_{info}.csv".format(ddir=datadir, info=infostr)

                    # dump raw pandas df to raw file
                    if store_raw is True:
                        df.to_csv(raw_file, index=False)

                    # print bulk/sumary data to summ_file
                    with h5py.File(filename, 'a') as f:
                        f['Inputs/Hsizes'][k_loop] = n
                        f['Inputs/Tvals'][k_loop] = T
                        f['Inputs/svals'][k_loop] = sp
                        f['Inputs/discs'][k_loop] = disc
                        f['Outputs/f_probs'][k_loop] = bf_pgs
                        f['Outputs/r_probs'][k_loop] = br_pgs
                        f['Outputs/bfrem_probs'][k_loop] = bfrem_pgs
                        f['Outputs/bfrem_psizes'][k_loop] = nq_bp
                        f['Outputs/avgfrem_probs'][k_loop] = avgfrem_pgs
                        f['Outputs/p_btf'][k_loop] = p_btf
                        f['Outputs/p_btr'][k_loop] = p_btr
                        if profile:
                            toc = time.perf_counter()
                            rtime = toc - tic
                            f['Outputs/rtime'][k_loop] = rtime

                    k_loop += 1

    return df
