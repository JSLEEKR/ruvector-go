package distance

import (
	"math"
	"testing"
)

func almostEqual(a, b, eps float32) bool {
	return float32(math.Abs(float64(a-b))) < eps
}

// --- Cosine ---

func TestCosineIdentical(t *testing.T) {
	v := []float32{1, 2, 3}
	d := CosineDistance(v, v)
	if !almostEqual(d, 0, 1e-6) {
		t.Errorf("cosine(v, v) = %f, want 0", d)
	}
}

func TestCosineOrthogonal(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	d := CosineDistance(a, b)
	if !almostEqual(d, 1.0, 1e-6) {
		t.Errorf("cosine(orthogonal) = %f, want 1.0", d)
	}
}

func TestCosineOpposite(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{-1, 0}
	d := CosineDistance(a, b)
	if !almostEqual(d, 2.0, 1e-6) {
		t.Errorf("cosine(opposite) = %f, want 2.0", d)
	}
}

func TestCosineZeroVector(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	d := CosineDistance(a, b)
	if d != 1.0 {
		t.Errorf("cosine(zero, b) = %f, want 1.0", d)
	}
}

func TestCosineDifferentLengths(t *testing.T) {
	a := []float32{1, 2}
	b := []float32{1, 2, 3}
	d := CosineDistance(a, b)
	if d != 1.0 {
		t.Errorf("cosine(diff len) = %f, want 1.0", d)
	}
}

func TestCosineEmpty(t *testing.T) {
	d := CosineDistance(nil, nil)
	if d != 1.0 {
		t.Errorf("cosine(nil, nil) = %f, want 1.0", d)
	}
}

// --- Euclidean ---

func TestEuclideanIdentical(t *testing.T) {
	v := []float32{1, 2, 3}
	d := EuclideanDistance(v, v)
	if d != 0 {
		t.Errorf("euclidean(v, v) = %f, want 0", d)
	}
}

func TestEuclideanKnown(t *testing.T) {
	a := []float32{0, 0}
	b := []float32{3, 4}
	d := EuclideanDistance(a, b)
	if !almostEqual(d, 5.0, 1e-6) {
		t.Errorf("euclidean = %f, want 5.0", d)
	}
}

func TestEuclideanSquared(t *testing.T) {
	a := []float32{0, 0}
	b := []float32{3, 4}
	d := EuclideanDistanceSquared(a, b)
	if !almostEqual(d, 25.0, 1e-6) {
		t.Errorf("euclideanSq = %f, want 25.0", d)
	}
}

func TestEuclideanEmpty(t *testing.T) {
	d := EuclideanDistance(nil, nil)
	if d != 0 {
		t.Errorf("euclidean(nil) = %f, want 0", d)
	}
}

// --- DotProduct ---

func TestDotProductKnown(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	d := DotProductDistance(a, b)
	// 1*4+2*5+3*6 = 32, negated = -32
	if !almostEqual(d, -32, 1e-6) {
		t.Errorf("dot = %f, want -32", d)
	}
}

func TestDotProductOrthogonal(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	d := DotProductDistance(a, b)
	if !almostEqual(d, 0, 1e-6) {
		t.Errorf("dot(orthogonal) = %f, want 0", d)
	}
}

func TestDotProductEmpty(t *testing.T) {
	d := DotProductDistance(nil, nil)
	if d != 0 {
		t.Errorf("dot(nil) = %f, want 0", d)
	}
}

// --- Manhattan ---

func TestManhattanKnown(t *testing.T) {
	a := []float32{0, 0}
	b := []float32{3, 4}
	d := ManhattanDistance(a, b)
	if !almostEqual(d, 7.0, 1e-6) {
		t.Errorf("manhattan = %f, want 7.0", d)
	}
}

func TestManhattanIdentical(t *testing.T) {
	v := []float32{1, 2, 3}
	d := ManhattanDistance(v, v)
	if d != 0 {
		t.Errorf("manhattan(v, v) = %f, want 0", d)
	}
}

func TestManhattanEmpty(t *testing.T) {
	d := ManhattanDistance(nil, nil)
	if d != 0 {
		t.Errorf("manhattan(nil) = %f, want 0", d)
	}
}

// --- Normalize ---

func TestNormalize(t *testing.T) {
	v := []float32{3, 4}
	n := Normalize(v)
	if !almostEqual(n[0], 0.6, 1e-6) || !almostEqual(n[1], 0.8, 1e-6) {
		t.Errorf("normalize(%v) = %v, want [0.6, 0.8]", v, n)
	}
}

func TestNormalizeZero(t *testing.T) {
	v := []float32{0, 0}
	n := Normalize(v)
	if n[0] != 0 || n[1] != 0 {
		t.Errorf("normalize(zero) = %v, want [0, 0]", n)
	}
}

func TestNormalizeNil(t *testing.T) {
	n := Normalize(nil)
	if n != nil {
		t.Errorf("normalize(nil) = %v, want nil", n)
	}
}

// --- Metric helpers ---

func TestMetricString(t *testing.T) {
	tests := []struct {
		m    Metric
		want string
	}{
		{Cosine, "cosine"},
		{Euclidean, "euclidean"},
		{DotProduct, "dotproduct"},
		{Manhattan, "manhattan"},
		{Metric(99), "unknown(99)"},
	}
	for _, tt := range tests {
		if got := tt.m.String(); got != tt.want {
			t.Errorf("%d.String() = %q, want %q", int(tt.m), got, tt.want)
		}
	}
}

func TestParseMetric(t *testing.T) {
	tests := []struct {
		s       string
		want    Metric
		wantErr bool
	}{
		{"cosine", Cosine, false},
		{"euclidean", Euclidean, false},
		{"l2", Euclidean, false},
		{"dotproduct", DotProduct, false},
		{"dot", DotProduct, false},
		{"manhattan", Manhattan, false},
		{"l1", Manhattan, false},
		{"unknown", 0, true},
	}
	for _, tt := range tests {
		got, err := ParseMetric(tt.s)
		if (err != nil) != tt.wantErr {
			t.Errorf("ParseMetric(%q) error = %v, wantErr %v", tt.s, err, tt.wantErr)
		}
		if got != tt.want {
			t.Errorf("ParseMetric(%q) = %v, want %v", tt.s, got, tt.want)
		}
	}
}

func TestGetFunc(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}

	fn := Get(Cosine)
	d := fn(a, b)
	if !almostEqual(d, 1.0, 1e-6) {
		t.Errorf("Get(Cosine) = %f, want 1.0", d)
	}

	fn = Get(Metric(99))
	d = fn(a, b)
	if !almostEqual(d, 1.0, 1e-6) {
		t.Errorf("Get(unknown) should default to cosine, got %f", d)
	}
}

// --- High-dimensional ---

func TestCosineHighDim(t *testing.T) {
	dim := 384
	a := make([]float32, dim)
	b := make([]float32, dim)
	for i := range a {
		a[i] = float32(i) * 0.01
		b[i] = float32(dim-i) * 0.01
	}
	d := CosineDistance(a, b)
	if d < 0 || d > 2 {
		t.Errorf("cosine high-dim out of range: %f", d)
	}
}

func TestEuclideanHighDim(t *testing.T) {
	dim := 384
	a := make([]float32, dim)
	b := make([]float32, dim)
	for i := range a {
		a[i] = 1.0
		b[i] = 2.0
	}
	d := EuclideanDistance(a, b)
	expected := float32(math.Sqrt(float64(dim)))
	if !almostEqual(d, expected, 0.01) {
		t.Errorf("euclidean high-dim = %f, want %f", d, expected)
	}
}
