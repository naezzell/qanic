"""Ising Hamiltonian class implementation"""
# useful external python packages
import networkx as nx
import qutip as qt

# internal utilities
from . import utils

class IsingH():
    """Class that performs common protocols on an Ising Hamiltonian."""

    def __init__(self, Hz, kind='unspecified', AandBfile='default'):
        """
        Instatiates an Ising Hamiltonian.

        Inputs
        Hz - a dictionary input (TODO: add more type support)
        kind - a short string description of Hamiltonian (e.g. 'K3')
        AandBfile - location of annealing A and B functions

        Allows analysis of Ising Hamiltonians via numerical diagonalization,
        numerical annealing, and D-Wave annealing.
        """
        if isinstance(Hz, dict):
            self.Hz = Hz
        else:
            s_types = "dict"
            c_type = str(type(Hz))
            print("{} is not in list of supported types: {}".format(s_types ,c_type)) 
        self.kind = kind
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

    def diag_Hz(self):
        """Returns eigen-energies and eigenstates of Hz."""
        return self.num_Hz.eigenstates()

    def Hz_gs_info(self, etol=1e-8, stol=1e-12):
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

    def dwaveH0_gs(self, etol=1e-8, stol=1e-12):
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
    
    def dwaveH1_gs(self, etol=1e-8, stol=1e-12):
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
        # get ground-state and return it
        gs = utils.gs_calculator(H1, etol, stol)['gs']
        return gs
    
    def numeric_anneal(self, sch, disc=0.0001, init_state=None, history=False):
        """
        Performs in-house (QuTiP based) numeric anneal.

        Input
        --------------------
        *usch: list -- [[t0, s0], [t1, s1], ..., [tn, sn]]
        *disc (0.01): float -- discretization used between times
        *init_state (None): dict -- maps qubits to up (1) or down (0)
        None means calculate here as gs of Hx (for forward)
        *history: bool, True means return all intermediate states
        False returns only final probability vector

        Output
        --------------------
        if history == None:
            outputs-->(energy, state)
            energy: float, energy of final state reached
            state: numpy array of wave-function amplitudes
        else:
            outputs-->QuTip result
        """
        # create numeric anneal schedule
        sch = utils.make_numeric_schedule(sch, disc)
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
            for qubit in self.qubits:
                if qubit in init_state:
                    statelist.append(init_state[qubit])
                else:
                    raise ValueError("init_state does not specify state of qubit {}".format(qubit))
            init_state = qt.tensor(*statelist)
            
        # peform a numerical anneal on H (sch[0] is list of discrete times)
        results = qt.sesolve(listH, init_state, sch[0])

        # only output final result if history set to False
        if history is False:
            state = utils.qto_to_npa(results.states[-1])
            probs = (state.conj()*state).real
            return probs
        return results

    def frem_anneal(self, fsch, rsch, rinit, partition, disc=0.0001, history=False):
        """
        Performs a numeric FREM anneal on H using QuTip.

        Inputs:
        ---------
        *fsch: list--forward annealing schedule [[t0, 0], ..., [tf, 1]]
        *rsch: list--reverse annealing schedule [[t0, 1], ..., [tf, 1]]
        *rinit: dict--initial state of HR
        *partition: dict--contains F-parition (HF), R-partition (HR),
        and Rqubits {'HF': {HF part}, 'HR': {HR part}, 'Rqubits': [list]}
        *disc: float--discretization between times in numeric anneal
        *history: bool--False final prob vector; True all intermediate states

        Outputs:
        ---------
        *final_state: numpy array--if history is True, then
        contains wave function amps with tensor product ordering
        else: contains sesolve output with all intermediate states
        """
        # slim down rinit to only those states relevant for R partition
        Rstate = {q: rinit[q] for q in rinit if q in partition['Rqubits']}
        # add pause to f/r schedule if it is shorter than the other
        Tf = fsch[-1][0]
        Tr = rsch[-1][0]
        rdiff = Tr - Tf
        if rdiff != 0:
            if rdiff > 0:
                fsch.append([Tf + rdiff, 1])
            else:
                rsch.append([Tr + (-1 * rdiff), 1])
        # prepare Hamiltonian/ weight function list for QuTip se solver
        fsch = utils.make_numeric_schedule(fsch, disc)
        rsch = utils.make_numeric_schedule(rsch, disc)
        fsch_A, fsch_B = utils.time_interpolation(fsch, self.processor_data)
        rsch_A, rsch_B = utils.time_interpolation(rsch, self.processor_data)        
        # list H for schrodinger equation solver
        f_Hx, f_Hz, r_Hx, r_Hz = utils.get_frem_Hs(self.qubits, partition)
        listH = [[f_Hx, fsch_A], [f_Hz, fsch_B], [r_Hx, rsch_A], [r_Hz, rsch_B]]

        # create the initial state vector for the FREM anneal
        statelist = []
        xstate = (qt.ket('0') - qt.ket('1')).unit()
        for qubit in self.qubits:
            if qubit in Rstate:
                statelist.append(Rstate[qubit])
            else:
                statelist.append(xstate)
        init_state = qt.tensor(*statelist)

        # run the numerical simulation and extract the final state
        results = qt.sesolve(listH, init_state, fsch[0])

        # only output final result if history is set to False
        if history is False:
            state = utils.qto_to_npa(results.states[-1])
            probs = (state.conj()*state).real
            return probs
        return results
        
    def dwave_anneal(self, annealing_params, testrun = False):
        """Runs problems on D-Wave."""
        ##TODO write this...
        return self.Hz
