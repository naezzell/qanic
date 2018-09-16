import sys
sys.path.append("..")

# abstract base class
from probrep import ProbRep
from dwavetools import get_dwaveH

# D-Wave Ocean Dependencies
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.composites import TilingComposite
import dimod

# useful general python libraries
import pandas as pd
import numpy as np
import sys
import itertools
import re
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml
import json
import copy
from operator import add


class DictRep(ProbRep):
    """
    A concrete class that represents problems
    on the D-Wave as a dictionary of values.
    """

    def __init__(self, H, qpu, vartype, encoding):
        """
        Class that takes a dictionary representation of an ising-hamiltonian and submits problem to a quantum annealer

        qpu: string
            specifies quantum processing unit to be used during calculation--referring to name specifying this information in dwave config file

        vartype: string
            QUBO or Ising (all case variants acceptable)

        encoding: string
            logical or direct. If logical, embeds onto chip under the hood. If direct, assumes manual embedding and tries to put directly onto chip as is.

        H: dict
            The hamiltonian represented as a dict of the form {(0, 0): h0, (0, 1): J01, ...} OR {(0, 0): 'h0', (0, 1): 'J0', (1, 1): 'h1', ...}
        """
        # poplate run information
        super().__init__(qpu, vartype, encoding)
        
        # create a set of regex rules to parse h/J string keys in H
        # also creates a "rules" dictionary to relate qubit weights to relevant factor
        self.hvrule = re.compile('h[0-9]*')
        self.Jvrule = re.compile('J[0-9]*')
        self.weight_rules = {}

        # create list of qubits/ couplers and
        # dicts that map indepndent params to
        # all qubits/ couplers that have that value
        self.qubits = []
        self.params_to_qubits = {}
        self.couplers = []
        self.params_to_couplers = {}
        for key, value in H.items():
            if key[0] == key[1]:
                self.qubits.append(key[0])
                if type(value) != str:
                    div_idx = -1
                else:
                    div_idx = value.find('/')
                if div_idx == -1:
                    self.weight_rules[key[0]] = 1
                else:
                    self.weight_rules[key[0]] = float(value[div_idx+1:])
                    value = value[:div_idx]
                self.params_to_qubits.setdefault(value, []).append(key[0])
            else:
                self.couplers.append(key)
                if type(value) != str:
                    div_idx = -1
                else:
                    div_idx = value.find('/')
                if div_idx == -1:
                    self.weight_rules[key] = 1
                else:
                    self.weight_rules[key] = float(value[div_idx+1:])
                    value = value[:div_idx]
                self.params_to_couplers.setdefault(value, []).append(key)

        if qpu == 'dwave':
            try:
                # let OCEAN handle embedding
                if encoding == "logical":
                    # encode several times on graph
                    # based on qubits encoded
                    if len(self.qubits) <= 4:
                        self.sampler = EmbeddingComposite(TilingComposite(DWaveSampler(), 1, 1, 4))
                    else:
                        self.sampler = EmbeddingComposite(DWaveSampler())

                # otherwise, assume 1-1
                else:
                    self.sampler = DWaveSampler()

            except:
                raise ConnectionError("Cannot connect to DWave sampler. Have you created a DWave config file using 'dwave config create'?")

        elif qpu == 'test':
            self.sampler = dimod.SimulatedAnnealingSampler()

        # save values/ metadata
        self.H = copy.deepcopy(H)
        if encoding == 'direct':
            self.wqubits = self.sampler.properties['qubits']
            self.wcouplers = self.sampler.properties['couplers']
    

    def save_config(self, fname, config_data={}):
        """
        Saves Hamiltonian configuration for future use
        """
        # if config data not supplied, at least dump Hamiltonian
        if config_data == {}:
            config_data = {'H': self.H}

        with open(fname + ".yml", 'w') as yamloutput:
            yaml.dump(config_data, yamloutput)
        
    
    def tile_H(self):
        pqubits = self.qubits
        pcouplers = self.couplers
        wqubits = self.wqubits
        wcouplers = self.wcouplers
        H = copy.deepcopy(self.H)

        for unitcell in range(1, 16 * 16):
            # create list of qubits/couplers that should be working in this unit cell
            unit_qubits = [q for q in range(8 * unitcell, 8 * (unitcell + 1))]
            unit_couplers = [[q, q + (4 - q % 4) + i] for q in unit_qubits[:4] for i in range(4)]

            # ensure that all qubits and couplers are working
            if all(q in wqubits for q in unit_qubits) and all(c in wcouplers for c in unit_couplers):
                # init_state.extend(init_state[:])
                # copy the initial state
                # if so, create copies of H on other unit cells
                for q in unit_qubits:
                    if (q % 8) in pqubits:
                        H[(q, q)] = H[(q % 8, q % 8)]

                for c in unit_couplers:
                    if (c[0] % 8, c[1] % 8) in pcouplers:
                        H[tuple(c)] = H[(c[0] % 8, c[1] % 8)]

            # else:
                #init_state.extend([3 for q in range(len(unit_qubits))])

        #self.init_state = init_state
        self.H = H
        self.qubits = []
        self.params_to_qubits = {}
        self.couplers = []
        self.params_to_couplers = {}
        for key, value in H.items():
            if key[0] == key[1]:
                self.qubits.append(key[0])
                self.params_to_qubits.setdefault(value, []).append(key[0])
            else:
                self.couplers.append(key)
                self.params_to_couplers.setdefault(value, []).append(key)


    def populate_parameters(self, parameters):
        # generate all independent combinations of parameters
        self.params = []
        self.values = []
        for key, value in parameters.items():
            self.params.append(key)
            self.values.append(value)
        self.combos = list(itertools.product(*self.values))

        # format pandas DataFrame
        # columns = self.params[:]
        # columns.extend(['energy', 'state'])
        self.data = pd.DataFrame()
        self.data.H = str(self.H)
        self.data.vartype = self.vartype
        self.data.encoding = self.encoding

        return

    def call_annealer(self, **kwargs):
        """
        Calls qpu on problem encoded by H.
        cull: bool
            if true, only outputs lowest energy states,
            otherwise shows all results
        """

        # parse the input data
        # cull = kwargs.get('cull', False)  # only takes lowest energy data
        s_to_hx = kwargs.get('s_to_hx', '')  # relates s to transverse-field bias
        spoint = kwargs.get('spoint', 0)  # can start sampling data midway through if interuptted

        # if H.values() only contains floats,
        # run sinlge problem as is
        if all(type(value) == int or type(value) == float for value in self.H.values()):

            if self.vartype == 'ising':
                h, J = get_dwaveH(self.H, 'ising')
                response = self.sampler.sample_ising(h, J)
                return response

            elif self.vartype == 'qubo':
                H = get_dwaveH(self.H, 'vartype')
                response = self.sampler.sample_ising(H)
                return response

        # otherwise, run a parameter sweep
        for combo in self.combos[spoint::]:
            # init single run's data/inputs
            rundata = {}
            runh = {}
            runJ = {}
            optional_args = {}
            count = 0
            # map params to values in combo
            for param in self.params:
                rundata[param] = combo[count]
                # if param is qubit param
                if self.hvrule.match(param):
                    for qubit in self.params_to_qubits[param]:
                        runh[qubit] = combo[count]/self.weight_rules[qubit]

                elif self.Jvrule.match(param):
                    for coupler in self.params_to_couplers[param]:
                        runJ[coupler] = combo[count]/self.weight_rules[coupler]

                elif param == 'anneal_schedule':
                    anneal_schedule = combo[count]
                    optional_args['anneal_schedule'] = anneal_schedule
                    # if transverse field terms, get hx
                    if s_to_hx:
                        hx = s_to_hx[anneal_schedule[1][1]]
                        rundata['hx'] = hx

                elif param == 'num_reads':
                    num_reads = combo[count]
                    optional_args['num_reads'] = num_reads

                elif param == 'initial_state':
                    initial_state = combo[count]
                    optional_args['initial_state'] = initial_state

                elif param == 'reinitialize_state':
                    reinitialize_state = combo[count]
                    optional_args['reinitialize_state'] = reinitialize_state

                count += 1

            # run the sim and collect data
            if self.qpu == 'dwave':

                response = self.sampler.sample_ising(h=runh, J=runJ, **optional_args)

                for energy, state, num in response.data(fields=['energy', 'sample', 'num_occurrences']):
                    rundata['energy'] = energy
                    rundata['state'] = tuple(state[key] for key in sorted(state.keys()))
                    for n in range(num):
                        self.data = self.data.append(rundata, ignore_index=True)

            elif self.qpu == 'simulate':
                bqm = dimod.BinaryQuadraticModel.from_ising(h=runh, J=runJ)
                response = self.sampler.sample(bqm, num_reads=num_reads)
                for energy, state in response.data(fields=['energy', 'sample']):
                    rundata['energy'] = energy
                    rundata['state'] = tuple(state[key] for key in sorted(state.keys()))
                    self.data = self.data.append(rundata, ignore_index=True)

        return self.data

    def visualize_graph(self):
        G = nx.Graph()
        G.add_edges_from(self.couplers)
        nx.draw_networkx(G)
        return G

    def save_data(self, filename):
        self.data.to_csv(filename, index=False)

    def get_state_plot(self, figsize=(12, 8), filename=None, title='Distribution of Final States'):
        data = self.data
        ncount = len(data)

        plt.figure(figsize=figsize)
        ax = sns.countplot(x="state", data=data)
        plt.title(title)
        plt.xlabel('State')

        # Make twin axis
        ax2 = ax.twinx()

        # Switch so count axis is on right, frequency on left
        ax2.yaxis.tick_left()
        ax.yaxis.tick_right()

        # Also switch the labels over
        ax.yaxis.set_label_position('right')
        ax2.yaxis.set_label_position('left')

        ax2.set_ylabel('Frequency [%]')

        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y),
                        ha='center', va='bottom')  # set the alignment of the text

        # Use a LinearLocator to ensure the correct number of ticks
        ax.yaxis.set_major_locator(ticker.LinearLocator(11))

        # Fix the frequency range to 0-100
        ax2.set_ylim(0, 100)
        ax.set_ylim(0, ncount)

        # And use a MultipleLocator to ensure a tick spacing of 10
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

        # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
        # ax2.grid(None)

        if filename:
            plt.savefig(filename, dpi=300)

        return plt

    def get_ferro_diagram(self, xparam, yparam, divideby=None, title=''):
        """
        Plot the probability that output states are ferromagnetic on a contour plot
        with xaxis as xparam/divdeby and yaxis as yparam/divideby.
        """
        df = self.data
        # obtain denominator of expression (if constant value is used across-the-board)
        if divideby:
            denom = abs(df[divideby].unique()[0])
            xlabel = xparam + '/|' + divideby + '|'
            ylabel = yparam + '/|' + divideby + '|'
        else:
            denom = 1
            xlabel = xparam
            ylabel = yparam

        xs = df[xparam].unique()
        ys = df[yparam].unique()

        pfm_meshgrid = []
        # iterate over trials and obtain pFM for given xparam and yparam
        for y in ys:
            pfms = []
            for x in xs:
                # get length of unique elements in state bitstrings
                lubs = [len(set(state)) for state in df.loc[(df[yparam] == y) & (df[xparam] == x)]['state']]
                pfms.append(lubs.count(1) / len(lubs))
            pfm_meshgrid.append(pfms)

        X, Y = np.meshgrid(xs / denom, ys / denom)

        # plot the figure
        plt.figure()
        plt.title(title)
        plt.contourf(X, Y, pfm_meshgrid, np.arange(0, 1.2, .2), cmap='viridis', extent=(-4, 4, 0, 4))
        cbar = plt.colorbar(ticks=np.arange(0, 1.2, .2))
        cbar.ax.set_title('$P_{FM}$')
        plt.clim(0, 1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        return plt

    def sudden_anneal_test(self):
        pass

    def get_QUBO_rep(self):
        pass

    def get_Ising_rep(self):
        pass
