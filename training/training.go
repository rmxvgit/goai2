package training

import (
	"goai/neuralnet"
	"goai/samples"
)

func TrainBatch(net *neuralnet.Net, samples []samples.Sample) {
	average := neuralnet.NewAverageRun(net)
	for sample_index := range samples {
		run := net.Run(samples[sample_index].Input)
		run.Expected_outputs = samples[sample_index].Expected_output
		run.ComputeDerivatives()
		average.Add(run)
	}
	net.Update(average, 10)
}
