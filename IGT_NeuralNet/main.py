
from Network import Network
from Neuron import Neuron
import math

def main():
    topology = []
    topology.append(2)
    topology.append(3)
    topology.append(2)
    net = Network(topology)
    Neuron.eta = 0.09
    Neuron.alpha = 0.015
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0, 0], [1, 0], [1, 0], [0, 1]]
    while True:
        err = 0
        for i in range(len(inputs)):
            net.setInput(inputs[i])
            net.feedForward()
            net.backPropagate(outputs[i])
            err = err + net.getError(outputs[i])
        print("error: ", err)
        if err < 0.01:
            break

    while True:
        a = input("type 1st input :")
        b = input("type 2nd input :")
        net.setInput([a, b])
        net.feedForword()
        print(net.getThResults())


if __name__ == '__main__':
    main()