package mat

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	e_to_the_neg_x := math.Exp(-x)
	return e_to_the_neg_x / math.Pow((1+e_to_the_neg_x), 2)
}

// Performs matrix multiplication
func MatMul(a [][]float64, b [][]float64) [][]float64 {
	rowsA := len(a)
	colsA := len(a[0])
	rowsB := len(b)
	colsB := len(b[0])

	if colsA != rowsB {
		panic("Matrix dimensions do not match for multiplication")
	}

	result := make([][]float64, rowsA)
	for i := range rowsA {
		result[i] = make([]float64, colsB)
		for j := range colsB {
			for k := range colsA {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

// multilication of a matrix by a collumn vector
func MatApply(matrix [][]float64, column_vector []float64) []float64 {
	if len(matrix[0]) != len(column_vector) {
		panic("Matrix and vector dimensions do not match")
	}

	output := make([]float64, len(matrix))

	for i := range len(matrix) {
		for j := range len(column_vector) {
			output[i] += matrix[i][j] * column_vector[j]
		}
	}

	return output
}

// comapres two matrixes
func EqualMatrices(a [][]float64, b [][]float64) bool {
	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		return false
	}

	for i := range a {
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}
