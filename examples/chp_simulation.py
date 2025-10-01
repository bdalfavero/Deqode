from deqode.chp_sim import StabilizerTableau
from deqode.gate import HadamardGate, CNOTGate, MeasureGate
from deqode.circuit import Circuit

def main():
    circuit = Circuit((2))
    circuit.append(HadamardGate(0))
    circuit.append(CNOTGate(0, 1))
    circuit.append(MeasureGate(0))
    circuit.append(MeasureGate(1))
    print(circuit)
    result = circuit.sample()
    print(result)


if __name__ == "__main__":
    main()