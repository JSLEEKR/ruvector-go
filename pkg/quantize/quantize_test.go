package quantize

import (
	"math"
	"math/rand"
	"testing"
)

// --- Scalar Quantizer ---

func TestSQNewQuantizer(t *testing.T) {
	sq := NewScalarQuantizer(128)
	if sq.Dimension() != 128 {
		t.Errorf("dim = %d, want 128", sq.Dimension())
	}
	if sq.IsTrained() {
		t.Error("should not be trained initially")
	}
}

func TestSQTrainEmpty(t *testing.T) {
	sq := NewScalarQuantizer(4)
	err := sq.Train(nil)
	if err == nil {
		t.Error("expected error for empty training data")
	}
}

func TestSQTrainDimMismatch(t *testing.T) {
	sq := NewScalarQuantizer(4)
	err := sq.Train([][]float32{{1, 2, 3}})
	if err == nil {
		t.Error("expected dimension mismatch error")
	}
}

func TestSQEncodeBeforeTrain(t *testing.T) {
	sq := NewScalarQuantizer(4)
	_, err := sq.Encode([]float32{1, 2, 3, 4})
	if err == nil {
		t.Error("expected error before training")
	}
}

func TestSQRoundtrip(t *testing.T) {
	sq := NewScalarQuantizer(4)
	data := [][]float32{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{-1, -2, 0, 10},
	}
	if err := sq.Train(data); err != nil {
		t.Fatalf("train: %v", err)
	}
	if !sq.IsTrained() {
		t.Error("should be trained")
	}

	for _, v := range data {
		enc, err := sq.Encode(v)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		dec, err := sq.Decode(enc)
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		// Check reconstruction error is small
		for i := range v {
			if math.Abs(float64(v[i]-dec[i])) > 0.1 {
				t.Errorf("dim %d: original=%f, decoded=%f", i, v[i], dec[i])
			}
		}
	}
}

func TestSQEncodeDimMismatch(t *testing.T) {
	sq := NewScalarQuantizer(4)
	_ = sq.Train([][]float32{{0, 0, 0, 0}, {1, 1, 1, 1}})
	_, err := sq.Encode([]float32{1, 2})
	if err == nil {
		t.Error("expected dim mismatch")
	}
}

func TestSQDecodeDimMismatch(t *testing.T) {
	sq := NewScalarQuantizer(4)
	_ = sq.Train([][]float32{{0, 0, 0, 0}, {1, 1, 1, 1}})
	_, err := sq.Decode([]uint8{1, 2})
	if err == nil {
		t.Error("expected dim mismatch")
	}
}

func TestSQCompressionRatio(t *testing.T) {
	sq := NewScalarQuantizer(128)
	if sq.CompressionRatio() != 4.0 {
		t.Errorf("ratio = %f, want 4.0", sq.CompressionRatio())
	}
}

func TestSQClamp(t *testing.T) {
	sq := NewScalarQuantizer(2)
	_ = sq.Train([][]float32{{0, 0}, {1, 1}})
	// Values outside training range should be clamped
	enc, _ := sq.Encode([]float32{-5, 10})
	if enc[0] != 0 {
		t.Errorf("below-min encoded to %d, want 0", enc[0])
	}
	if enc[1] != 255 {
		t.Errorf("above-max encoded to %d, want 255", enc[1])
	}
}

func TestSQConstantDimension(t *testing.T) {
	sq := NewScalarQuantizer(3)
	// All values same for dim 0
	data := [][]float32{{5, 0, 1}, {5, 1, 2}, {5, 2, 3}}
	err := sq.Train(data)
	if err != nil {
		t.Fatalf("train: %v", err)
	}
	enc, _ := sq.Encode([]float32{5, 1, 2})
	dec, _ := sq.Decode(enc)
	if math.Abs(float64(dec[0]-5)) > 0.1 {
		t.Errorf("constant dim decoded to %f, want 5", dec[0])
	}
}

func TestSQLargeDataRoundtrip(t *testing.T) {
	dim := 64
	n := 500
	rng := rand.New(rand.NewSource(42))
	sq := NewScalarQuantizer(dim)

	data := make([][]float32, n)
	for i := range data {
		data[i] = make([]float32, dim)
		for j := range data[i] {
			data[i][j] = rng.Float32()*2 - 1
		}
	}
	if err := sq.Train(data); err != nil {
		t.Fatalf("train: %v", err)
	}

	// Check average error
	var totalErr float64
	for _, v := range data {
		enc, _ := sq.Encode(v)
		dec, _ := sq.Decode(enc)
		for i := range v {
			totalErr += math.Abs(float64(v[i] - dec[i]))
		}
	}
	avgErr := totalErr / float64(n*dim)
	if avgErr > 0.01 {
		t.Errorf("average error = %f, want < 0.01", avgErr)
	}
}

// --- Product Quantizer ---

func TestPQNewInvalidSubspaces(t *testing.T) {
	_, err := NewProductQuantizer(10, 3, 256, 10)
	if err == nil {
		t.Error("expected error for non-divisible dim")
	}
}

func TestPQNewInvalidCodebook(t *testing.T) {
	_, err := NewProductQuantizer(8, 4, 257, 10)
	if err == nil {
		t.Error("expected error for codebook > 256")
	}
}

func TestPQTrainEmpty(t *testing.T) {
	pq, _ := NewProductQuantizer(8, 4, 4, 10)
	err := pq.Train(nil)
	if err == nil {
		t.Error("expected error")
	}
}

func TestPQTrainDimMismatch(t *testing.T) {
	pq, _ := NewProductQuantizer(8, 4, 4, 10)
	err := pq.Train([][]float32{{1, 2, 3}})
	if err == nil {
		t.Error("expected dim mismatch")
	}
}

func TestPQTrainTooFew(t *testing.T) {
	pq, _ := NewProductQuantizer(8, 4, 16, 10)
	data := [][]float32{{1, 2, 3, 4, 5, 6, 7, 8}}
	err := pq.Train(data)
	if err == nil {
		t.Error("expected too few vectors error")
	}
}

func TestPQRoundtrip(t *testing.T) {
	dim := 8
	subspaces := 4
	codebookSize := 8
	rng := rand.New(rand.NewSource(42))

	pq, err := NewProductQuantizer(dim, subspaces, codebookSize, 20)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	n := 100
	data := make([][]float32, n)
	for i := range data {
		data[i] = make([]float32, dim)
		for j := range data[i] {
			data[i][j] = rng.Float32()*2 - 1
		}
	}

	if err := pq.Train(data); err != nil {
		t.Fatalf("train: %v", err)
	}
	if !pq.IsTrained() {
		t.Error("should be trained")
	}

	// Encode and decode
	for i, v := range data[:10] {
		codes, err := pq.Encode(v)
		if err != nil {
			t.Fatalf("encode %d: %v", i, err)
		}
		if len(codes) != subspaces {
			t.Errorf("code length = %d, want %d", len(codes), subspaces)
		}
		dec, err := pq.Decode(codes)
		if err != nil {
			t.Fatalf("decode %d: %v", i, err)
		}
		if len(dec) != dim {
			t.Errorf("decoded length = %d, want %d", len(dec), dim)
		}
	}
}

func TestPQEncodeBeforeTrain(t *testing.T) {
	pq, _ := NewProductQuantizer(8, 4, 4, 10)
	_, err := pq.Encode([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	if err == nil {
		t.Error("expected error before training")
	}
}

func TestPQDistanceTable(t *testing.T) {
	dim := 8
	rng := rand.New(rand.NewSource(42))
	pq, _ := NewProductQuantizer(dim, 4, 8, 20)

	n := 50
	data := make([][]float32, n)
	for i := range data {
		data[i] = make([]float32, dim)
		for j := range data[i] {
			data[i][j] = rng.Float32()
		}
	}
	_ = pq.Train(data)

	query := data[0]
	table, err := pq.DistanceTable(query)
	if err != nil {
		t.Fatalf("distance table: %v", err)
	}
	if len(table) != 4 {
		t.Errorf("table length = %d, want 4", len(table))
	}
	for m, row := range table {
		if len(row) != 8 {
			t.Errorf("table[%d] length = %d, want 8", m, len(row))
		}
	}

	// Distance from query to itself via table should be small
	codes, _ := pq.Encode(query)
	d := DistanceWithTable(table, codes)
	if d > 1.0 {
		t.Errorf("self-distance = %f, want small", d)
	}
}

func TestPQSubspaces(t *testing.T) {
	pq, _ := NewProductQuantizer(16, 8, 4, 5)
	if pq.Subspaces() != 8 {
		t.Errorf("subspaces = %d, want 8", pq.Subspaces())
	}
}

func TestPQCompressionRatio(t *testing.T) {
	pq, _ := NewProductQuantizer(128, 16, 256, 5)
	ratio := pq.CompressionRatio()
	// 128 * 4 bytes / 16 bytes = 32
	if ratio != 32 {
		t.Errorf("ratio = %f, want 32", ratio)
	}
}

func TestPQDecodeCodesMismatch(t *testing.T) {
	dim := 8
	rng := rand.New(rand.NewSource(42))
	pq, _ := NewProductQuantizer(dim, 4, 8, 10)
	data := make([][]float32, 20)
	for i := range data {
		data[i] = make([]float32, dim)
		for j := range data[i] {
			data[i][j] = rng.Float32()
		}
	}
	_ = pq.Train(data)
	_, err := pq.Decode([]uint8{1, 2})
	if err == nil {
		t.Error("expected code length mismatch")
	}
}
