import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from qanic.numerics import partgen
from qanic.numerics import partgen, revstate
from sidon_frem_sim import sidon_frem_sim

Hsizes = [3]
hbool = False
Tvals = [1]
svals = [0.27]
discs = [.1]
r_init = revstate.ml_measurement
frem_init = revstate.c_diag_HR
part_scheme = partgen.all_parts
include_zero = False
hx_ver = 1
filename = f'qubits_{Hsizes}_Tvals_{Tvals}_zero_{include_zero}_hxver_{hx_ver}_.hdf5'
currdir = "."

sidon_frem_sim(Hsizes, hbool, Tvals, svals, discs, r_init, frem_init, part_scheme, include_zero, hx_ver, filename, currdir, profile=True)


