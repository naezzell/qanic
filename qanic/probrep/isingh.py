"""Ising Hamiltonian class implementation"""
# useful python libraries
import math

# useful external python packages
import networkx as nx
import qutip as qt

# internal utilities
from . import utils

class IsingH():
    """Class that performs common protocols on an Ising Hamiltonian."""

    def __init__(self, Hz, AandBfile='default'):
        """
        Input: hamiltonian in the form of a dictionary
        #TODO add support for other input types

        Allows analysis of Ising Hamiltonians via numerical diagonalization,
        numerical annealing, and D-Wave annealing.
        """
        if isinstance(Hz, dict):
            self.Hz = Hz
        else:
            s_types = "dict"
            c_type = str(type(Hz))
            print("{} is not in list of supported types: {}".format(s_types ,c_type))

        # create the dwave-friendly Ising problem representation
        self.dwave_Hz = utils.get_dwave_H(self.Hz)
        self.qubits = list(self.dwave_Hz[0].keys())
        self.couplers = list(self.dwave_Hz[1].keys())

        # create the networkx Ising problem representation for viewing
        self.graph_Hz = utils.get_networkx_H(self.Hz)

        # create the numeric Hamiltonian amenable to QuTip operations
        num_H = utils.get_numeric_H(self.Hz)
        self.num_Hz = num_H[0]
        self.num_Hx = num_H[1]

        # if processor data is loaded, load in A and B functions
        if AandBfile is not None:
            if AandBfile == 'default':
                self.processor_data = utils.loadAandB()
            else:
                self.processor_data = utils.loadAandB(AandBfile)
            
    def __str__(self):
        return str(self.Hz)

    def __getitem__(self, idx):
        return self.Hz[idx]

    def visualize(self):
        """returns networkx graph visualization of connectivity."""
        return nx.draw_networkx(self.graph_Hz)   

    def diagonalize(self):
        """Returns eigen-energies and eigenstates of Hz."""
        return self.num_Hz.eigenstates()

    def get_Hz_gs(self, etol=1e-8, stol=1e-12):
        """
        Returns the ground-state energy/eigenstates/degeneracy of Hz.
        Inputs
        --------------------
        etol: energy tolerance, cut-off for 'numerical equivalence'
        stol: small tolerance, anything less is set to 0 during normalization

        Output
        --------------------
        (super-position gs state, energy, degeneracy, [list of groundstates])
        """
        result = utils.gs_calculator(self.num_Hz, etol, stol)
        result['gs'] = utils.qto_to_npa(result['gs'])
        return result

    def get_dwaveH0_gs(self, etol=1e-8, stol=1e-12):
        """
        Returns the ground-state energy/eigenstates/degeneracy of H(s=0)
        Inputs
        --------------------
        etol: energy tolerance, cut-off for 'numerical equivalence'
        stol: small tolerance, anything less is set to 0 during normalization

        Output
        --------------------
        (super-position gs state, energy, degeneracy, [list of groundstates])
        
        """
        # construct H(0) = A(0)Hx + B(0)Hz
        Avals = self.processor_data['Avals']
        Bvals = self.processor_data['Bvals']
        H0 = Avals[0]*self.num_Hx + Bvals[0]*self.num_Hz
        # get ground-state and return it
        gs = utils.gs_calculator(H0, etol, stol)['gs']
        return gs
    
    def get_dwaveH1_gs(self, etol=1e-8, stol=1e-12):
        """
        Returns the ground-state energy/eigenstates/degeneracy of H(s=1)
        Inputs
        --------------------
        etol: energy tolerance, cut-off for 'numerical equivalence'
        stol: small tolerance, anything less is set to 0 during normalization

        Output
        --------------------
        (super-position gs state, energy, degeneracy, [list of groundstates])
        
        """
        # construct H(0) = A(0)Hx + B(0)Hz
        Avals = self.processor_data['Avals']
        Bvals = self.processor_data['Bvals']
        H1 = Avals[-1]*self.num_Hx + Bvals[-1]*self.num_Hz
        print(Avals[-1])
        print(Bvals[-1])
        # get ground-state and return it
        gs = utils.gs_calculator(H1, etol, stol)['gs']
        return gs
    
    def numeric_anneal(self, ann_params={}, history=False):
        """
        Performs in-house (QuTiP based) numeric anneal.

        Input
        --------------------
        --> ann_params--dict of annealing params as floats (default value)
        *t1 (1): anneal length (in micro seconds) from s_init to sp
        *direction (f): 'f' or 'r' for forward (s_init = 0) or reverse (s_init = 1)
        *sp (1): intermediate s value (s-prime) reached after annealing for t1
        *tp (0): duration of pause after reaching sp
        *t2 (0): anneal length from sp to s_final = 1
        *disc (0.01): discretization used between times
        *init_state (None): initial state for anneal (forward or reverse)
        None means calculate in here as ground-state of H(t=0).
        --> history: True or False
        returns full results if True and just final state if False

        Output
        --------------------
        if history == None:
            outputs-->(energy, state)
            energy: float, energy of final state reached
            state: numpy array of wave-function amplitudes
        else:
            outputs-->QuTip result
        """
        # set init state to None if not specified
        init_state = ann_params.get('init_state', None)
        
        # create the anneal schedule (numeric schedule hosts defaults)
        sch = utils.make_numeric_schedule(ann_params)
        # interpolate A and B according to schedule
        sch_A, sch_B = utils.time_interpolation(sch, self.processor_data)
        # list H for schrodinger equation solver
        listH = [[self.num_Hx, sch_A], [self.num_Hz, sch_B]]
        # calculate ground-state at H(t=0) if init_state not specified            
        if init_state is None:
            xstate = (qt.ket('0') - qt.ket('1')).unit()
            statelist = [xstate for i in range(len(self.qubits))]
            init_state = qt.tensor(*statelist)
        else:
            statelist = []
            xstate = (qt.ket('0') - qt.ket('1')).unit()
            for qubit in self.qubits:
                if qubit in init_state:
                    zstate = qt.ket(str(init_state[qubit]))
                    statelist.append(zstate)
                else:
                    statelist.append(xstate)
            init_state = qt.tensor(*statelist)
            
        # peform a numerical anneal on H (sch[0] is list of discrete times)
        results = qt.sesolve(listH, init_state, sch[0])

        # only output final result if history set to False
        if history is False:
            return utils.qto_to_npa(results.states[-1])
        return results

    def frem_anneal(self, f_ann_params, r_ann_params, partition, history=False):
        """
        Performs a numeric FREM anneal on H using QuTip.

        inputs:
        ---------
        f_ann_params - a dictionary containg forward annealing parameters (clear from numeric_anneal what these are)
        r_ann_params - a dictionary containing reverse annealing parameters
        partition - dictionary containing F-parition (HF), R-partition (HR), and Rqubits
        {'HF': {HF part as dict}, 'HR': {HR part as dict}}

        outputs:
        ---------
        final_state - numpy array containing amplitudes of wave-functions with canonical tensor-product order 
        """
        # read init states of f/r annealing parameters and default to None
        r_init_state = r_ann_params.get('init_state', None)

        # add pause to forward anneal if it is shorter than reverse
        # TODO: make it possible to do the same for reverse
        ft = f_ann_params.get('t1', 1) + f_ann_params.get('tp', 0) + f_ann_params.get('t2', 0)
        rt = r_ann_params.get('t1', 1) + r_ann_params.get('tp', 0) + r_ann_params.get('t2', 0)
        if ft < rt:
            f_ann_params['tp'] = (rt - ft)
        if ft > rt:
            raise ValueError("Currently, having a forward anneal last longer than reverse is not supported.")
        # prepare Hamiltonian/ weight function list for QuTip se solver
        f_sch = utils.make_numeric_schedule(f_ann_params)
        r_sch = utils.make_numeric_schedule(r_ann_params)
        f_sch_A, f_sch_B = utils.time_interpolation(f_sch, self.processor_data)
        r_sch_A, r_sch_B = utils.time_interpolation(r_sch, self.processor_data)        
        # list H for schrodinger equation solver
        f_Hz, f_Hx = utils.get_numeric_H(partition['HF'])
        r_Hz, r_Hx = utils.get_numeric_H(partition['HR'])
        listH = [[f_Hx, f_sch_A], [f_Hz, f_sch_B], [r_Hx, r_sch_A], [r_Hz, r_sch_B]]

        # create the initial state vector for the FREM anneal
        statelist = []
        xstate = (qt.ket('0') - qt.ket('1')).unit()
        for qubit in self.qubits:
            if qubit in r_init_state:
                zstate = qt.ket(str(r_init_state[qubit]))
                statelist.append(zstate)
            else:
                statelist.append(xstate)
        init_state = qt.tensor(*statelist)

        # run the numerical simulation and extract the final state
        results = qt.sesolve(listH, init_state, f_sch[0])

        # only output final result if history is set to False
        if history is False:
            return utils.qto_to_npa(results.states[-1])
        return results
        
    def dwave_anneal(self, annealing_params, testrun = False):
        """Runs problems on D-Wave."""
        ##TODO write this...
        return self.Hz
