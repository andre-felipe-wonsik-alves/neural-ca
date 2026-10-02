from controllers.neuron_network import NeuronalNetwork

if __name__ == "__main__":
    network = NeuronalNetwork(40,40)
    
    raw_signal = network.run(3000)
    print(raw_signal)
    
    # ?? Simular o EEG com Butterworth
    # ?? Comparar com os dados reais
    