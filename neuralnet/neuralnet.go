package neuralnet

import (
	"goai/mat"
	"goai/utils"
	"math/rand/v2"
)

// represents the network structure
type Net struct {
	Layer_sizes []uint
	Layers      []Layer
	Connections []Connection
}

// values associated with a neuron layer
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
	Expected_outputs  []float64
}

type ConnectionRunInfo struct {
	Weights_derivatives [][]float64 // this tells how much the output of the neuron changes with respect to each weight
}

type LayerRunInfo struct {
	Nodes_sigmoided    []float64
	Nodes_raw          []float64
	Nodes_derivative   []float64 // this tells how much the output of the neuron changes with respect to each node
	Biases_derivatives []float64 // this tells how much the output of the neuron changes with respect to each bias
}

type AverageRunObject struct {
	Reference_network *Net // reference network used to generate this run info
	Number_of_runs    uint
	Layers            []LayerRunInfo
	Connections       []ConnectionRunInfo
}

type Updater interface {
	updateNetwork(net *Net, learning_rate float64)
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

func NewRandNet(layer_sizes []uint) *Net {
	net := NewNet(layer_sizes)
	for i := range net.Connections {
		for ii := range net.Connections[i].Weights {
			for iii := range net.Connections[i].Weights[ii] {
				net.Connections[i].Weights[ii][iii] = rand.Float64()*2 - 1
			}
		}
	}

	return net
}

func NewNetRun(net *Net) *NetworkRunObject {
	run := &NetworkRunObject{
		Reference_network: net,
		Connections:       make([]ConnectionRunInfo, len(net.Connections)),
		Layers:            make([]LayerRunInfo, len(net.Layers)),
		Expected_outputs:  make([]float64, net.Layer_sizes[net.FinalLayerIndex()]),
	}

	for i := range run.Layers {
		run.Layers[i].Nodes_derivative = make([]float64, net.Layer_sizes[i])
		run.Layers[i].Nodes_raw = make([]float64, net.Layer_sizes[i])
		run.Layers[i].Nodes_sigmoided = make([]float64, net.Layer_sizes[i])
		run.Layers[i].Biases_derivatives = make([]float64, net.Layer_sizes[i])
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

func NewAverageRun(net *Net) *AverageRunObject {
	average := &AverageRunObject{
		Reference_network: net,
		Number_of_runs:    0,
		Layers:            make([]LayerRunInfo, len(net.Layer_sizes)),
		Connections:       make([]ConnectionRunInfo, len(net.Connections)),
	}

	for i := range average.Layers {
		average.Layers[i].Nodes_derivative = make([]float64, net.Layer_sizes[i])
		average.Layers[i].Nodes_raw = make([]float64, net.Layer_sizes[i])
		average.Layers[i].Nodes_sigmoided = make([]float64, net.Layer_sizes[i])
		average.Layers[i].Biases_derivatives = make([]float64, net.Layer_sizes[i])
	}

	// generation of a 2 dimentional matrix with dimentions [output_layer_size X input_layer_size ]. [lin x col]
	for inp_layer_index := range average.Connections {
		output_layer_index := inp_layer_index + 1
		average.Connections[inp_layer_index].Weights_derivatives = make([][]float64, net.Layer_sizes[output_layer_index])
		for ii := range average.Connections[inp_layer_index].Weights_derivatives {
			average.Connections[inp_layer_index].Weights_derivatives[ii] = make([]float64, net.Layer_sizes[inp_layer_index])
		}
	}
	return average
}

func (net *Net) Run(input []float64) *NetworkRunObject {
	run := NewNetRun(net)
	run.ForwardAll(input)
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

func (net *Net) FinalLayerIndex() uint {
	return uint(len(net.Layers) - 1)
}

func (net *Net) Update(updatable_object Updater, learning_rate float64) {
	updatable_object.updateNetwork(net, learning_rate)
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

// computes the derivative of any neuron and updates the run object with it
func (run *NetworkRunObject) DerivativeOfNeuron(layer_index uint, neuron_index uint) float64 {
	if layer_index == run.Reference_network.FinalLayerIndex() {
		derivative := run.DerivativeFinalNeuron(neuron_index)
		return derivative
	}

	next_layer_index := layer_index + 1
	run.Layers[layer_index].Nodes_derivative[neuron_index] = 0 // reset derivative just in case

	for output_neuron_index := range run.Reference_network.Layer_sizes[next_layer_index] {
		// TODO: Write a documentation for this part
		respective_weight, _ := run.Reference_network.GetConn(layer_index, neuron_index, output_neuron_index)
		derivative_neuron_with_respect_to_raw := mat.SigmoidDerivative(run.Layers[next_layer_index].Nodes_raw[output_neuron_index])
		derivative := respective_weight * derivative_neuron_with_respect_to_raw * run.Layers[next_layer_index].Nodes_derivative[output_neuron_index]
		run.Layers[layer_index].Nodes_derivative[neuron_index] += derivative
	}

	return run.Layers[layer_index].Nodes_derivative[neuron_index]
}

// computes the derivative of the final neuron and updates the run object with it
func (run *NetworkRunObject) DerivativeFinalNeuron(neuron_index uint) float64 {
	distance_to_expected_result := run.Expected_outputs[neuron_index] - run.Layers[run.Reference_network.FinalLayerIndex()].Nodes_sigmoided[neuron_index]
	derivative_of_neuron := 2 * distance_to_expected_result
	run.Layers[run.Reference_network.FinalLayerIndex()].Nodes_derivative[neuron_index] = derivative_of_neuron
	return derivative_of_neuron
}

func (run *NetworkRunObject) DerivativeOfWeight(input_layer_index uint, input_neuron_index uint, output_neuron_index uint) float64 {
	output_layer_index := input_layer_index + 1
	input_neuron_value := run.Layers[input_layer_index].Nodes_sigmoided[input_neuron_index]

	derivative_of_neuron_with_respect_raw := mat.SigmoidDerivative(run.Layers[output_layer_index].Nodes_raw[output_neuron_index])
	derivative := input_neuron_value * derivative_of_neuron_with_respect_raw * run.Layers[output_layer_index].Nodes_derivative[output_neuron_index]
	run.Connections[input_layer_index].Weights_derivatives[output_neuron_index][input_neuron_index] = derivative
	return derivative
}

func (run *NetworkRunObject) DerivativeOfBias(layer_index uint, neuron_index uint) float64 {
	derivative_of_neuron_with_respect_raw := mat.SigmoidDerivative(run.Layers[layer_index].Nodes_raw[neuron_index])
	derivative := derivative_of_neuron_with_respect_raw * run.Layers[layer_index].Nodes_derivative[neuron_index]
	run.Layers[layer_index].Biases_derivatives[neuron_index] = derivative
	return derivative
}

func (run *NetworkRunObject) ComputeDerivatives() {
	last_layer_index := run.Reference_network.FinalLayerIndex()
	for layer_index := last_layer_index; layer_index > 0; layer_index-- {
		previous_layer_index := layer_index - 1
		for neuron_index := range run.Layers[layer_index].Nodes_derivative {
			run.DerivativeOfNeuron(layer_index, uint(neuron_index))
			run.DerivativeOfBias(layer_index, uint(neuron_index))
			for previous_neuron_index := range run.Layers[previous_layer_index].Nodes_derivative {
				run.DerivativeOfWeight(previous_layer_index, uint(previous_neuron_index), uint(neuron_index))
			}
		}
	}
}

func (average *AverageRunObject) Add(run *NetworkRunObject) {
	var correction_coefitient float64 = float64(average.Number_of_runs) / float64(average.Number_of_runs+1)
	for layer_index := range average.Layers {
		for node_index := range average.Layers[layer_index].Nodes_raw {
			bias_average := &average.Layers[layer_index].Biases_derivatives[node_index]
			bias_run := &average.Layers[layer_index].Biases_derivatives[node_index]
			*bias_average = (*bias_average)*correction_coefitient + (*bias_run)/float64(average.Number_of_runs+1)
		}
	}

	for connection_index := range average.Connections {
		for line_index := range average.Connections[connection_index].Weights_derivatives {
			for collumn_index := range average.Connections[connection_index].Weights_derivatives[line_index] {
				weight_average := &average.Connections[connection_index].Weights_derivatives[line_index][collumn_index]
				weight_run := &run.Connections[connection_index].Weights_derivatives[line_index][collumn_index]
				*weight_average = (*weight_average)*correction_coefitient + (*weight_run)/float64(average.Number_of_runs+1)
			}
		}
	}
	average.Number_of_runs += 1
}

func (average *AverageRunObject) updateNetwork(net *Net, learning_rate float64) {
	for layer_index := range average.Layers {
		for node_index := range average.Layers[layer_index].Nodes_raw {
			bias_average_derivative := &average.Layers[layer_index].Biases_derivatives[node_index]
			bias_value := &net.Layers[layer_index].Biases[node_index]
			*bias_value = (*bias_value) + learning_rate*(*bias_average_derivative)
		}
	}

	for connection_index := range average.Connections {
		for line_index := range average.Connections[connection_index].Weights_derivatives {
			for collumn_index := range average.Connections[connection_index].Weights_derivatives[line_index] {
				weight_average_derivative := &average.Connections[connection_index].Weights_derivatives[line_index][collumn_index]
				weight_value := &net.Connections[connection_index].Weights[line_index][collumn_index]
				*weight_value = (*weight_value) + learning_rate*(*weight_average_derivative)
			}
		}
	}
	average.Number_of_runs += 1
}
