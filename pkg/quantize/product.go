package quantize

import (
	"fmt"
	"math"
	"math/rand"
)

// ProductQuantizer implements product quantization (PQ).
// Splits vectors into M subspaces and trains K centroids per subspace.
type ProductQuantizer struct {
	dim          int
	subspaces    int     // M: number of subspaces
	subDim       int     // dimension per subspace (dim / M)
	codebookSize int     // K: centroids per subspace (typically 256)
	codebooks    [][][]float32 // [M][K][subDim] centroid vectors
	iterations   int
	trained      bool
}

// NewProductQuantizer creates a PQ quantizer.
// subspaces must evenly divide dim. codebookSize is typically 256 for 1-byte codes.
func NewProductQuantizer(dim, subspaces, codebookSize, iterations int) (*ProductQuantizer, error) {
	if dim%subspaces != 0 {
		return nil, fmt.Errorf("dim %d must be divisible by subspaces %d", dim, subspaces)
	}
	if codebookSize <= 0 || codebookSize > 256 {
		return nil, fmt.Errorf("codebook size must be in [1, 256], got %d", codebookSize)
	}
	if iterations <= 0 {
		iterations = 20
	}
	return &ProductQuantizer{
		dim:          dim,
		subspaces:    subspaces,
		subDim:       dim / subspaces,
		codebookSize: codebookSize,
		iterations:   iterations,
	}, nil
}

// Train runs k-means on each subspace to build codebooks.
func (pq *ProductQuantizer) Train(data [][]float32) error {
	if len(data) == 0 {
		return fmt.Errorf("no training data")
	}
	for _, v := range data {
		if len(v) != pq.dim {
			return fmt.Errorf("dimension mismatch: got %d, want %d", len(v), pq.dim)
		}
	}
	if len(data) < pq.codebookSize {
		return fmt.Errorf("need at least %d training vectors, got %d", pq.codebookSize, len(data))
	}

	rng := rand.New(rand.NewSource(42))
	pq.codebooks = make([][][]float32, pq.subspaces)

	for m := 0; m < pq.subspaces; m++ {
		// Extract subvectors for this subspace
		offset := m * pq.subDim
		subvecs := make([][]float32, len(data))
		for i, v := range data {
			subvecs[i] = v[offset : offset+pq.subDim]
		}
		pq.codebooks[m] = kmeans(subvecs, pq.codebookSize, pq.iterations, pq.subDim, rng)
	}

	pq.trained = true
	return nil
}

// Encode compresses a vector to M bytes (one centroid index per subspace).
func (pq *ProductQuantizer) Encode(v []float32) ([]uint8, error) {
	if !pq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}
	if len(v) != pq.dim {
		return nil, fmt.Errorf("dimension mismatch: got %d, want %d", len(v), pq.dim)
	}

	codes := make([]uint8, pq.subspaces)
	for m := 0; m < pq.subspaces; m++ {
		offset := m * pq.subDim
		sub := v[offset : offset+pq.subDim]
		codes[m] = pq.nearestCentroid(m, sub)
	}
	return codes, nil
}

// Decode reconstructs a vector from PQ codes.
func (pq *ProductQuantizer) Decode(codes []uint8) ([]float32, error) {
	if !pq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}
	if len(codes) != pq.subspaces {
		return nil, fmt.Errorf("code length mismatch: got %d, want %d", len(codes), pq.subspaces)
	}
	for m, c := range codes {
		if int(c) >= pq.codebookSize {
			return nil, fmt.Errorf("code[%d]=%d exceeds codebook size %d", m, c, pq.codebookSize)
		}
	}

	decoded := make([]float32, pq.dim)
	for m := 0; m < pq.subspaces; m++ {
		centroid := pq.codebooks[m][codes[m]]
		copy(decoded[m*pq.subDim:(m+1)*pq.subDim], centroid)
	}
	return decoded, nil
}

// DistanceTable precomputes distances from a query to all centroids.
// Returns [M][K] table where table[m][k] = dist(query_sub_m, centroid_m_k).
func (pq *ProductQuantizer) DistanceTable(query []float32) ([][]float32, error) {
	if !pq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}
	if len(query) != pq.dim {
		return nil, fmt.Errorf("dimension mismatch: got %d, want %d", len(query), pq.dim)
	}

	table := make([][]float32, pq.subspaces)
	for m := 0; m < pq.subspaces; m++ {
		offset := m * pq.subDim
		sub := query[offset : offset+pq.subDim]
		table[m] = make([]float32, pq.codebookSize)
		for k := 0; k < pq.codebookSize; k++ {
			table[m][k] = squaredDist(sub, pq.codebooks[m][k])
		}
	}
	return table, nil
}

// DistanceWithTable computes approximate distance using precomputed table.
// Panics are prevented by bounds-checking codes against table dimensions.
func DistanceWithTable(table [][]float32, codes []uint8) float32 {
	var sum float32
	for m, code := range codes {
		if m >= len(table) || int(code) >= len(table[m]) {
			return float32(math.MaxFloat32)
		}
		sum += table[m][code]
	}
	return float32(math.Sqrt(float64(sum)))
}

func (pq *ProductQuantizer) nearestCentroid(m int, sub []float32) uint8 {
	bestIdx := 0
	bestDist := float32(math.MaxFloat32)
	for k, centroid := range pq.codebooks[m] {
		d := squaredDist(sub, centroid)
		if d < bestDist {
			bestDist = d
			bestIdx = k
		}
	}
	return uint8(bestIdx)
}

// IsTrained returns whether the quantizer has been trained.
func (pq *ProductQuantizer) IsTrained() bool {
	return pq.trained
}

// Subspaces returns M.
func (pq *ProductQuantizer) Subspaces() int {
	return pq.subspaces
}

// CompressionRatio returns the compression ratio.
func (pq *ProductQuantizer) CompressionRatio() float64 {
	return float64(pq.dim*4) / float64(pq.subspaces) // float32 bytes / PQ bytes
}

// squaredDist computes squared L2 distance between two sub-vectors.
func squaredDist(a, b []float32) float32 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return float32(sum)
}

// kmeans runs k-means clustering and returns K centroids.
func kmeans(data [][]float32, k, iterations, dim int, rng *rand.Rand) [][]float32 {
	n := len(data)

	// Random initialization: pick k random data points
	centroids := make([][]float32, k)
	perm := rng.Perm(n)
	for i := 0; i < k; i++ {
		centroids[i] = make([]float32, dim)
		copy(centroids[i], data[perm[i]])
	}

	assignments := make([]int, n)

	for iter := 0; iter < iterations; iter++ {
		// Assignment step
		for i, v := range data {
			bestIdx := 0
			bestDist := squaredDist(v, centroids[0])
			for j := 1; j < k; j++ {
				d := squaredDist(v, centroids[j])
				if d < bestDist {
					bestDist = d
					bestIdx = j
				}
			}
			assignments[i] = bestIdx
		}

		// Update step
		counts := make([]int, k)
		newCentroids := make([][]float32, k)
		for i := 0; i < k; i++ {
			newCentroids[i] = make([]float32, dim)
		}
		for i, v := range data {
			c := assignments[i]
			counts[c]++
			for d := 0; d < dim; d++ {
				newCentroids[c][d] += v[d]
			}
		}
		for i := 0; i < k; i++ {
			if counts[i] > 0 {
				for d := 0; d < dim; d++ {
					newCentroids[i][d] /= float32(counts[i])
				}
				centroids[i] = newCentroids[i]
			}
		}
	}

	return centroids
}
