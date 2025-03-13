package utils

import "golang.org/x/exp/constraints"

func SliceApply[T any](slice []T, f func(T) T) []T {
	result := make([]T, len(slice))
	for i, v := range slice {
		result[i] = f(v)
	}
	return result
}

func SliceCompare[T comparable](slice1, slice2 []T) bool {
	if len(slice1) != len(slice2) {
		return false
	}
	for i, v := range slice1 {
		if v != slice2[i] {
			return false
		}
	}
	return true
}

func AddSlice[T constraints.Integer | constraints.Float](slice1, slice2 []T) []T {
	var result = make([]T, len(slice1))
	for i := range result {
		result[i] = slice1[i] + slice2[i]
	}
	return result
}
