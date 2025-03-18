package samples

import (
	"math"
	"math/rand/v2"
)

type SinSample struct {
	Input           float64
	Expected_output float64
}

func MakeSamples(n_samples uint) []SinSample {
	samples := make([]SinSample, n_samples)

	for i := range samples {
		input := rand.Float64() * 20
		expect := math.Sin(input) / 2
		samples[i] = SinSample{
			Input:           input,
			Expected_output: expect,
		}
	}
	return samples
}
