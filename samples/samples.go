package samples

import (
	"math"
	"math/rand"
)

type Sample struct {
	Input           []float64
	Expected_output []float64
	NetOutput       []float64
}

func MakeSinSamples(n_samples uint) []Sample {
	smpls := make([]Sample, n_samples)
	for i := range smpls {
		input := rand.Float64() * 3
		smpls[i].Input = []float64{input}
		smpls[i].Expected_output = []float64{math.Sin(input) / 3}
	}
	return smpls
}
