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

func TestForwardLayer1(t *testing.T) {
	t.Parallel()

	input := []float64{0.5, 1}
	expected_unsigmoided_output := []float64{10.5, 5.5}
	expected_sigmoided_output := utils.SliceApply(expected_unsigmoided_output, mat.Sigmoid)

	network := neuralnet.NewNet([]uint{2, 2})
	network.SetWeight(0, 0, 0, 3)
	network.SetWeight(0, 0, 1, 1)
	network.SetWeight(0, 1, 0, 9)
	network.SetWeight(0, 1, 1, 5)

	run := neuralnet.NewNetRun(network)
	run.ForwardAll(input)

	sigmoided_result := run.GetFinalSigmoidedOutput()
	unsigmoided_result := run.GetFinalRawOutput()

	if !utils.SliceCompare(unsigmoided_result, expected_unsigmoided_output) {
		t.Errorf("Expected unsigmoided output to be %v, got %v", expected_unsigmoided_output, unsigmoided_result)
	}

	if !utils.SliceCompare(sigmoided_result, expected_sigmoided_output) {
		t.Errorf("Expected sigmoided output to be %v, got %v", expected_sigmoided_output, sigmoided_result)
	}
}

func TestForwardLayer2(t *testing.T) {
	t.Parallel()

	input := []float64{1, 1}

	expected_middle_layer_raw := []float64{5, 7, 9}
	expected_middle_layer := utils.SliceApply(expected_middle_layer_raw, mat.Sigmoid)

	network := neuralnet.NewNet([]uint{2, 3, 2})
	network.SetWeight(0, 0, 0, 1)
	network.SetWeight(0, 0, 1, 2)
	network.SetWeight(0, 0, 2, 3)
	network.SetWeight(0, 1, 0, 4)
	network.SetWeight(0, 1, 1, 5)
	network.SetWeight(0, 1, 2, 6)

	network.SetWeight(1, 0, 0, -1)
	network.SetWeight(1, 0, 1, -2)
	network.SetWeight(1, 1, 0, -3)
	network.SetWeight(1, 1, 1, -4)
	network.SetWeight(1, 2, 0, -5)
	network.SetWeight(1, 2, 1, -6)

	run := neuralnet.NewNetRun(network)
	run.ForwardAll(input)

	middle_layer, middle_layer_raw := run.GetLayerState(1)

	if !utils.SliceCompare(middle_layer_raw, expected_middle_layer_raw) {
		t.Errorf("Expected middle layer raw output to be %v, got %v", expected_middle_layer_raw, middle_layer_raw)
	}

	if !utils.SliceCompare(middle_layer, expected_middle_layer) {
		t.Errorf("Expected middle layer sigmoided output to be %v, got %v", expected_middle_layer, middle_layer)
	}
}
