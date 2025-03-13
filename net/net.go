package net

type Net struct {
	Layer_sizes []uint
	Layers      []Layer
	Connections []Connection
}

// valores associados a uma camada de neuronios
type Layer struct {
	Biases     []float64
	Nodes_raw  []float64 // sem ter passado pelo sigmoide
	Nodes_sigm []float64 // passado pelo sigmoide
}

// valores associados a uma conexao entre duas camadas de neuronios
type Connection struct {
	weights [][]float64
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
		net.Layers[i].Nodes_raw = make([]float64, net.Layer_sizes[i])
		net.Layers[i].Nodes_sigm = make([]float64, net.Layer_sizes[i])
	}

	return net
}
