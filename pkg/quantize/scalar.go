// Package quantize provides vector quantization for memory-efficient storage.
package quantize

import (
	"fmt"
	"math"
)

// ScalarQuantizer compresses float32 vectors to uint8 (SQ8).
// Uses per-dimension min/max scaling to [0, 255].
type ScalarQuantizer struct {
	dim    int
	mins   []float32
	maxs   []float32
	scales []float32 // (max - min) / 255
	trained bool
}

// NewScalarQuantizer creates a new SQ8 quantizer for the given dimension.
func NewScalarQuantizer(dim int) *ScalarQuantizer {
	return &ScalarQuantizer{
		dim:  dim,
		mins: make([]float32, dim),
		maxs: make([]float32, dim),
		scales: make([]float32, dim),
	}
}

// Train computes per-dimension min/max from training data.
func (sq *ScalarQuantizer) Train(data [][]float32) error {
	if len(data) == 0 {
		return fmt.Errorf("no training data")
	}
	for _, v := range data {
		if len(v) != sq.dim {
			return fmt.Errorf("dimension mismatch: got %d, want %d", len(v), sq.dim)
		}
	}

	// Initialize with first vector
	for i := 0; i < sq.dim; i++ {
		sq.mins[i] = data[0][i]
		sq.maxs[i] = data[0][i]
	}

	// Scan all data
	for _, v := range data[1:] {
		for i := 0; i < sq.dim; i++ {
			if v[i] < sq.mins[i] {
				sq.mins[i] = v[i]
			}
			if v[i] > sq.maxs[i] {
				sq.maxs[i] = v[i]
			}
		}
	}

	// Compute scales
	for i := 0; i < sq.dim; i++ {
		r := sq.maxs[i] - sq.mins[i]
		if r == 0 {
			sq.scales[i] = 1 // avoid division by zero
		} else {
			sq.scales[i] = r / 255.0
		}
	}

	sq.trained = true
	return nil
}

// Encode quantizes a float32 vector to uint8.
func (sq *ScalarQuantizer) Encode(v []float32) ([]uint8, error) {
	if !sq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}
	if len(v) != sq.dim {
		return nil, fmt.Errorf("dimension mismatch: got %d, want %d", len(v), sq.dim)
	}

	encoded := make([]uint8, sq.dim)
	for i := 0; i < sq.dim; i++ {
		val := (v[i] - sq.mins[i]) / sq.scales[i]
		if val < 0 {
			val = 0
		}
		if val > 255 {
			val = 255
		}
		encoded[i] = uint8(math.Round(float64(val)))
	}
	return encoded, nil
}

// Decode reconstructs a float32 vector from uint8 codes.
func (sq *ScalarQuantizer) Decode(encoded []uint8) ([]float32, error) {
	if !sq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}
	if len(encoded) != sq.dim {
		return nil, fmt.Errorf("dimension mismatch: got %d, want %d", len(encoded), sq.dim)
	}

	decoded := make([]float32, sq.dim)
	for i := 0; i < sq.dim; i++ {
		decoded[i] = sq.mins[i] + float32(encoded[i])*sq.scales[i]
	}
	return decoded, nil
}

// IsTrained returns whether the quantizer has been trained.
func (sq *ScalarQuantizer) IsTrained() bool {
	return sq.trained
}

// Dimension returns the vector dimension.
func (sq *ScalarQuantizer) Dimension() int {
	return sq.dim
}

// CompressionRatio returns the compression ratio (4x for SQ8).
func (sq *ScalarQuantizer) CompressionRatio() float64 {
	return 4.0 // float32 (4 bytes) → uint8 (1 byte)
}
