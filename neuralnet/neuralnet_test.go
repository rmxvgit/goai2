package neuralnet_test

import (
	"goai/mat"
	"goai/neuralnet"
	"goai/utils"
	"testing"
)

func TestGetConn(t *testing.T) {
	t.Parallel()
	network := neuralnet.NewNet([]uint{2, 2})
	network.Connections[0].Weights[1][0] = 3
	network.Layers[1].Biases[1] = 2
	weight, bias := network.GetConn(0, 0, 1)

	if weight != 3 {
		t.Errorf("Expected weight to be 3, got %f", weight)
	}
	if bias != 2 {
		t.Errorf("Expected bias to be 2, got %f", bias)
	}
}

func TestForwardLayer(t *testing.T) {
	t.Parallel()
	network := neuralnet.NewNet([]uint{2, 2})
	input := []float64{0.5, 1}
	expected_unsigmoided_output := []float64{10.5, 5.5}
	expected_sigmoided_output := utils.SliceApply(expected_unsigmoided_output, mat.Sigmoid)
	network.SetWeight(0, 0, 0, 3)
	network.SetWeight(0, 0, 1, 1)
	network.SetWeight(0, 1, 0, 9)
	network.SetWeight(0, 1, 1, 5)
	sigmoided_result, unsigmoided_result := network.ForwardLayer(0, input)

	if !utils.SliceCompare(unsigmoided_result, expected_unsigmoided_output) {
		t.Errorf("Expected unsigmoided output to be %v, got %v", expected_unsigmoided_output, unsigmoided_result)
	}

	if !utils.SliceCompare(sigmoided_result, expected_sigmoided_output) {
		t.Errorf("Expected unsigmoided output to be %v, got %v", sigmoided_result, expected_sigmoided_output)
	}
}
