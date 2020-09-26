import math
import numpy as np

Hsizes = [6]
Tvalues = np.linspace(0, 10, 50)

# this is the number of random trials of SKn needed for
# 95% confidence with 5% confidence interval
sample_size = {3: 23, 4: 43, 5: 66, 6: 92, 7: 117,
               8: 142, 9: 165, 10: 186}


def time_estimate(n, T, unit='m'):
    """
    Uses the fitted function from empirical data on scout
    MSU HPC^2 to estimate time to run simulation.
    """
    #t_sec = (.123 * np.exp(.434 * n)) + (.165 + .311 * T)
    t_sec = 10 * (2**(n/1.6))
    if unit == 's':
        return t_sec
    elif unit == 'm':
        return (t_sec / 60)
    elif unit == 'h':
        return (t_sec / (60 * 60))
    elif unit == 'd':
        return (t_sec / (3600 * 24))

def mem_estimate(n, T, unit='MB'):
    mem_MB = 0.315 * 2**n + (n * T / 20) + 177
    if unit == 'MB':
        return mem_MB
    elif unit == 'GB':
        return (mem_MB / 1000)

total_time = 0
max_mem = 0
for n in Hsizes:
    for T in Tvalues:
        ss = sample_size[n]
        total_time += ss * time_estimate(n, T, unit='h')
        mem = mem_estimate(n, T, unit='GB')
        if mem > max_mem:
            max_mem = mem

print(total_time)
print(max_mem)
