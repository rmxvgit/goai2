package neuralnet

import (
	"goai/mat"
	"goai/utils"
)

// represents the network structure
type Net struct {
	Layer_sizes []uint
	Layers      []Layer
	Connections []Connection
}

// valores associados a uma camada de neuronios
type Layer struct {
	Biases []float64
}

// value associated with the connections beteen two neuron layers
type Connection struct {
	Weights [][]float64
}

// NetworkRunObject is a struct that runs and holds information about a network run
type NetworkRunObject struct {
	Reference_network *Net // reference network used to generate this run info
	Layers            []LayerRunInfo
	Connections       []ConnectionRunInfo
}

type ConnectionRunInfo struct {
	Weights_derivatives [][]float64 // this tells how much the output of the neuron changes with respect to each weight
}

type LayerRunInfo struct {
	Nodes_sigmoided  []float64
	Nodes_raw        []float64
	Nodes_derivative []float64 // this tells how much the output of the neuron changes with respect to each node
}

func (net *Net) GetNumLayers() int {
	return len(net.Layer_sizes)
}

func NewNet(layer_sizes []uint) *Net {
	net := &Net{
		Layer_sizes: make([]uint, len(layer_sizes)),
		Layers:      make([]Layer, len(layer_sizes)),
		Connections: make([]Connection, len(layer_sizes)-1),
	}

	copy(net.Layer_sizes, layer_sizes)

	for i := range net.Layers {
		net.Layers[i].Biases = make([]float64, net.Layer_sizes[i])
	}

	// generation of a 2 dimentional matrix with dimentions [output_layer_size X input_layer_size ]. [lin x col]
	for inp_layer_index := range net.Connections {
		output_layer_index := inp_layer_index + 1
		net.Connections[inp_layer_index].Weights = make([][]float64, net.Layer_sizes[output_layer_index])
		for ii := range net.Connections[inp_layer_index].Weights {
			net.Connections[inp_layer_index].Weights[ii] = make([]float64, net.Layer_sizes[inp_layer_index])
		}
	}

	return net
}

func NewNetRun(net *Net) *NetworkRunObject {
	run := &NetworkRunObject{
		Reference_network: net,
		Connections:       make([]ConnectionRunInfo, len(net.Connections)),
		Layers:            make([]LayerRunInfo, len(net.Layers)),
	}

	for i := range run.Layers {
		run.Layers[i].Nodes_derivative = make([]float64, net.Layer_sizes[i])
		run.Layers[i].Nodes_raw = make([]float64, net.Layer_sizes[i])
		run.Layers[i].Nodes_sigmoided = make([]float64, net.Layer_sizes[i])
	}

	// generation of a 2 dimentional matrix with dimentions [output_layer_size X input_layer_size ]. [lin x col]
	for inp_layer_index := range run.Connections {
		output_layer_index := inp_layer_index + 1
		run.Connections[inp_layer_index].Weights_derivatives = make([][]float64, net.Layer_sizes[output_layer_index])
		for ii := range run.Connections[inp_layer_index].Weights_derivatives {
			run.Connections[inp_layer_index].Weights_derivatives[ii] = make([]float64, net.Layer_sizes[inp_layer_index])
		}
	}

	return run
}

// get the value of the weight and bias between two neurons
func (net *Net) GetConn(input_layer, input_neuron_index, output_neuron_index uint) (weight, bias float64) {
	weight = net.Connections[input_layer].Weights[output_neuron_index][input_neuron_index] // weight between the input and output neuron
	bias = net.Layers[input_layer+1].Biases[output_neuron_index]                           // bias associated with the output neuron
	return weight, bias
}

/*
get an especific bias

WARNING: the index used in this function is different from the one used in GetConn
*/
func (net *Net) GetBias(layer, neuron_index uint) float64 {
	return net.Layers[layer].Biases[neuron_index]
}

func (net *Net) SetWeight(input_layer, input_neuron_index, output_neuron_index uint, value float64) {
	net.Connections[input_layer].Weights[output_neuron_index][input_neuron_index] = value
}

func (net *Net) SetBias(layer, neuron_index uint, value float64) {
	net.Layers[layer].Biases[neuron_index] = value
}

/*
Given the neurons activation of one layer, calculate the activation of the next layer

# This is the root function from which all the forward-family functions are derived

# I do not recommend changing the values of the output slices of this function
*/
func (run *NetworkRunObject) ForwardLayer(input_layer_index uint, input_layer []float64) (output_sigmoided, output_raw []float64) {
	output_layer_index := input_layer_index + 1
	output_raw = mat.MatApply(run.Reference_network.Connections[input_layer_index].Weights, input_layer)
	output_raw = utils.AddSlice(output_raw, run.Reference_network.Layers[output_layer_index].Biases)
	output_sigmoided = utils.SliceApply(output_raw, mat.Sigmoid)
	run.Layers[output_layer_index].Nodes_raw = output_raw
	run.Layers[output_layer_index].Nodes_sigmoided = output_sigmoided
	return output_sigmoided, output_raw
}

/*
This function calculates the activation of the network from the given layer.

# Overwrites any previous activations of the network
*/
func (run *NetworkRunObject) ForwardAll(input []float64) {
	run.ForwardFrom(0, input)
}

/*
ForwardFrom calculates the activation of the network from the given layer.

# Overwrites any previous activations of the network
*/
func (run *NetworkRunObject) ForwardFrom(input_layer_index uint, input []float64) {
	copy(run.Layers[input_layer_index].Nodes_sigmoided, input)
	last_fowardable_layer_index := uint(len(run.Reference_network.Connections))
	for i := input_layer_index; i < last_fowardable_layer_index; i++ {
		run.ForwardLayer(i, run.Layers[i].Nodes_sigmoided)
	}
}

func (run *NetworkRunObject) GetFinalSigmoidedOutput() []float64 {
	return run.Layers[len(run.Layers)-1].Nodes_sigmoided
}

func (run *NetworkRunObject) GetFinalRawOutput() []float64 {
	return run.Layers[len(run.Layers)-1].Nodes_raw
}

func (run *NetworkRunObject) GetLayerState(layer_index uint) (layer_sigmoided, layer_raw []float64) {
	return run.Layers[layer_index].Nodes_sigmoided, run.Layers[layer_index].Nodes_raw
}
