import math
import numpy as np

class Connection:
    """Holds information about the inter-neuron connections, such as Weight and the Delta Weight for momentum."""
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0
