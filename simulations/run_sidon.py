import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import sys
import pandas as pd
sys.path.append("/home/nic/Dropbox/qanic-dev/")
sys.path.append("/home/nic/Dropbox/qanic-dev/simulations/")
from qanic.numerics.partgen import all_parts
from qanic.numerics.revstate import ml_measurement, c_diag_HR
from sidon_frem_sim import sidon_frem_sim

sample_size = {3: 23, 4: 43, 5: 66, 6: 92, 7: 117,
               8: 142, 9: 165, 10: 186}

Hsizes = [3]
hbool = False
Tvals = [10]
svals = [0.27]
discs = [.1]
r_init = ml_measurement
frem_init = c_diag_HR
part_scheme = all_parts
filename = '3qubit_10T.hdf5'
currdir = "/home/nic/Dropbox/qanic-dev/simualtions"

args = [Hsizes, hbool, Tvals, svals, discs, r_init, frem_init, part_scheme, filename, currdir, sample_size[Hsizes[0]]]
sidon_frem_sim(*args)


