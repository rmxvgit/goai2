package main

import (
	"goai/export"
	"goai/neuralnet"
	"goai/samples"
	"goai/training"
)

func main() {
	net := neuralnet.NewRandNet([]uint{1, 10, 10, 10, 1})

	for range 3000 {
		smpls := samples.MakeSinSamples(10)
		training.TrainBatch(net, smpls)
	}

	export.NetworkResults(net, samples.MakeSinSamples(200))
}
