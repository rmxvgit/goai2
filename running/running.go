package running

import (
	"goai/neuralnet"
	"goai/samples"
)

func Train(net *neuralnet.Net) {
	for range 200 {
		samples := samples.MakeSamples(10)
		average := neuralnet.NewAverageRun(net)
		for _, sample := range samples {
			run := neuralnet.NewNetRun(net)
			run.Expected_outputs = []float64{sample.Expected_output}
			run.ForwardAll([]float64{sample.Input})
			average.AddRun(run)
		}
		net.Update(average, 5)
	}
}
