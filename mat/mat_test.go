package mat_test

import (
	"goai/mat"
	"testing"
)

func TestMatMul(t *testing.T) {
	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{5, 6}, {7, 8}}
	expected := [][]float64{{19, 22}, {43, 50}}
	result := mat.MatMul(a, b)
	if !mat.EqualMatrices(result, expected) {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}
