import numpy as np
import random

class BarycentricSimplex:
    def __init__(self, coords):
        
        self.coords = coords

    def split(self, strategy='random'):
        if strategy == 'random':
            # select a random edge to split on and use the remaining vertices
            edge_idxs = random.choice(range(len(coords
            pass
        elif strategy == 'longest':
            pass
        else:
            raise Exception(f'{strategy} is not a valid splitting strategy')
            
        
