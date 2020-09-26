import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import datetime
import pandas as pd
from qanic.numerics import partgen
from qanic.numerics import partgen, revstate
from sidon_frem_sim import sidon_frem_sim

Hsizes = [3]
hbias = False
Tvals = [.01]
svals = [0.27]
discs = [.001]
r_init = revstate.ml_measurement
frem_init = revstate.c_diag_HR
part_scheme = partgen.all_parts
include_zero = False
store_raw = True
profile = False
now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
datadir = "./personal_tests/data/"
filename = f'{datadir}sidon_frem_sim_data_{now}_rinit_mlm_freminit_diag_parts_all_hbias_{hbias}_with0_{include_zero}.hdf5'
#filename = f'{currdir}/qubits_{Hsizes}_Tvals_{Tvals}_zero_{include_zero}_.hdf5'

sidon_frem_sim(Hsizes, hbias, Tvals, svals, discs, r_init, frem_init, part_scheme, include_zero, filename, datadir, store_raw=store_raw, profile=profile)


