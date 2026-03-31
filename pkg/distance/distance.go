// Package distance provides vector distance/similarity metrics.
// All functions operate on float32 slices and return float32 distances.
package distance

import (
	"fmt"
	"math"
)

// Metric identifies a distance function.
type Metric int

const (
	Cosine     Metric = iota // 1 - cos(a,b)
	Euclidean                // L2 distance
	DotProduct               // negative dot product (lower = more similar)
	Manhattan                // L1 distance
)

// String returns the human-readable name of the metric.
func (m Metric) String() string {
	switch m {
	case Cosine:
		return "cosine"
	case Euclidean:
		return "euclidean"
	case DotProduct:
		return "dotproduct"
	case Manhattan:
		return "manhattan"
	default:
		return fmt.Sprintf("unknown(%d)", int(m))
	}
}

// ParseMetric converts a string to a Metric.
func ParseMetric(s string) (Metric, error) {
	switch s {
	case "cosine":
		return Cosine, nil
	case "euclidean", "l2":
		return Euclidean, nil
	case "dotproduct", "dot":
		return DotProduct, nil
	case "manhattan", "l1":
		return Manhattan, nil
	default:
		return 0, fmt.Errorf("unknown distance metric: %q", s)
	}
}

// Func is the signature for a distance function.
type Func func(a, b []float32) float32

// Get returns the distance function for the given metric.
func Get(m Metric) Func {
	switch m {
	case Cosine:
		return CosineDistance
	case Euclidean:
		return EuclideanDistance
	case DotProduct:
		return DotProductDistance
	case Manhattan:
		return ManhattanDistance
	default:
		return CosineDistance
	}
}

// CosineDistance returns 1 - cosine_similarity(a, b).
// Returns 1.0 if either vector has zero magnitude.
func CosineDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 1.0
	}
	var dot, normA, normB float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	// Clamp to [-1, 1] to handle floating point errors.
	if sim > 1.0 {
		sim = 1.0
	}
	if sim < -1.0 {
		sim = -1.0
	}
	return float32(1.0 - sim)
}

// EuclideanDistance returns the L2 distance between a and b.
func EuclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return float32(math.Sqrt(sum))
}

// EuclideanDistanceSquared returns the squared L2 distance (avoids sqrt).
func EuclideanDistanceSquared(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return float32(sum)
}

// DotProductDistance returns the negative dot product of a and b.
// Lower values indicate higher similarity.
func DotProductDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return float32(-sum)
}

// ManhattanDistance returns the L1 distance between a and b.
func ManhattanDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var sum float64
	for i := range a {
		sum += math.Abs(float64(a[i]) - float64(b[i]))
	}
	return float32(sum)
}

// Normalize returns a unit-length copy of the vector.
func Normalize(v []float32) []float32 {
	if len(v) == 0 {
		return nil
	}
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		out := make([]float32, len(v))
		return out
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / norm)
	}
	return out
}
