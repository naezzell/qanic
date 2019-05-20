import random

def sidonCBP(n, sidonh = False):
    """
    Creates a complete bipartite graph Kn,n with random sidon J couplings.
    Can also put random sidon h couplings if desired, but default value for h is 0.
    """
    sidon = [-1, -19/28, -13/28, -8/28, 8/28, 13/28, 19/28, 1]
    H = {}
    for n1 in range(n):
        if sidonh is True:
            H[(n1, n1)] = random.choice(sidon)
        else:
            H[(n1, n1)] = 0

        for n2 in range(n, 2*n):
            H[(n1, n2)] = random.choice(sidon)
            if sidonh is True:
                H[(n2, n2)] = random.choice(sidon)
            else:
                H[(n2, n2)] = 0

    return H

def uniformKn(n, low=-1, high=1, randomh = False):
    """
    Creates a Kn graph with unform random J couplings ranging from [low, high].
    Setting randomh does the same thing to h, but default value is 0.
    """
    H = {}
    for n1 in range(n):
        if randomh is True:
            H[(n1, n1)] = random.uniform(-1, 1)
        else:
            H[(n1, n1)] = 0

        for n2 in range(n1+1, n):
            H[(n1, n2)] = random.uniform(-1, 1)

    return H

def sidonKn(n, sidonh = False):
    """
    Creates a Kn graph with randomly assigned Sidon J couplings.
    Sidon = {-1, -19/28, -13/28, -8/28, 8/28, 13/28, 19/28, 1}
    If sidonh is true, also gives random Sidon h values; otherwise,
    h bias fields are all set to 0.
    """
    sidon = [-1, -19/28, -13/28, -8/28, 8/28, 13/28, 19/28, 1]
    H = {}
    for n1 in range(n):
        if sidonh is True:
            H[(n1, n1)] = random.choice(sidon)
        else:
            H[(n1, n1)] = 0

        for n2 in range(n1+1, n):
            H[(n1, n2)] = random.choice(sidon)

    return H

def degenKn(n):
    """
    Creates a frustrated Kn (complete) Hamiltonian, DKn. To ensure some frustration,
    all couplings set to anti-ferromagnetic with 1 mis-aligned qubit bias.
    """
    H = {}
    for n1 in range(n):
        if n1!=0:
            H[(n1, n1)] = 1
        else:
            H[(n1, n1)] = -1
        for n2 in range(n1+1, n):
            H[(n1, n2)] = 1

    return H
