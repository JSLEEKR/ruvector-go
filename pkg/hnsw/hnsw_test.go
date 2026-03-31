package hnsw

import (
	"math"
	"math/rand"
	"sync"
	"testing"

	"github.com/JSLEEKR/ruvector-go/pkg/distance"
)

func randomVector(dim int, rng *rand.Rand) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}

func TestNewIndex(t *testing.T) {
	idx := New(128, distance.Cosine, DefaultConfig())
	if idx.Dimension() != 128 {
		t.Errorf("dim = %d, want 128", idx.Dimension())
	}
	if idx.Len() != 0 {
		t.Errorf("len = %d, want 0", idx.Len())
	}
}

func TestInsertSingle(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	err := idx.Insert(1, []float32{1, 2, 3})
	if err != nil {
		t.Fatalf("insert: %v", err)
	}
	if idx.Len() != 1 {
		t.Errorf("len = %d, want 1", idx.Len())
	}
}

func TestInsertDuplicate(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	_ = idx.Insert(1, []float32{1, 2, 3})
	err := idx.Insert(1, []float32{4, 5, 6})
	if err == nil {
		t.Error("expected error for duplicate insert")
	}
}

func TestInsertDimensionMismatch(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	err := idx.Insert(1, []float32{1, 2})
	if err == nil {
		t.Error("expected dimension mismatch error")
	}
}

func TestSearchEmpty(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	results, err := idx.Search([]float32{1, 2, 3}, 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("results = %d, want 0", len(results))
	}
}

func TestSearchSingleVector(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	_ = idx.Insert(1, []float32{1, 2, 3})
	results, err := idx.Search([]float32{1, 2, 3}, 1)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("results = %d, want 1", len(results))
	}
	if results[0].ID != 1 {
		t.Errorf("id = %d, want 1", results[0].ID)
	}
	if results[0].Distance > 1e-6 {
		t.Errorf("distance = %f, want ~0", results[0].Distance)
	}
}

func TestSearchKNN(t *testing.T) {
	idx := New(2, distance.Euclidean, DefaultConfig())
	// Insert points at known positions
	vectors := map[uint64][]float32{
		1: {0, 0},
		2: {1, 0},
		3: {0, 1},
		4: {10, 10},
		5: {-1, 0},
	}
	for id, v := range vectors {
		if err := idx.Insert(id, v); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}

	results, err := idx.Search([]float32{0, 0}, 3)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("results = %d, want 3", len(results))
	}

	// Closest should be ID 1 (the origin)
	if results[0].ID != 1 {
		t.Errorf("nearest = %d, want 1", results[0].ID)
	}
	// ID 4 (10,10) should NOT be in top 3
	for _, r := range results {
		if r.ID == 4 {
			t.Error("(10,10) should not be in top 3 nearest to origin")
		}
	}
}

func TestSearchDimensionMismatch(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	_ = idx.Insert(1, []float32{1, 2, 3})
	_, err := idx.Search([]float32{1, 2}, 1)
	if err == nil {
		t.Error("expected dimension mismatch error")
	}
}

func TestSearchInvalidK(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	_, err := idx.Search([]float32{1, 2, 3}, 0)
	if err == nil {
		t.Error("expected error for k=0")
	}
}

func TestDelete(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	_ = idx.Insert(1, []float32{1, 2, 3})
	_ = idx.Insert(2, []float32{4, 5, 6})
	if !idx.Delete(1) {
		t.Error("delete returned false for existing ID")
	}
	if idx.Len() != 1 {
		t.Errorf("len = %d, want 1", idx.Len())
	}
	if idx.Delete(1) {
		t.Error("delete returned true for already-deleted ID")
	}
}

func TestDeleteNonexistent(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	if idx.Delete(999) {
		t.Error("delete returned true for nonexistent ID")
	}
}

func TestDeleteEntryPoint(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	_ = idx.Insert(1, []float32{1, 2, 3})
	_ = idx.Insert(2, []float32{4, 5, 6})
	_ = idx.Insert(3, []float32{7, 8, 9})
	idx.Delete(idx.entryPoint)
	if idx.Len() != 2 {
		t.Errorf("len = %d, want 2", idx.Len())
	}
	// Should still be able to search
	results, err := idx.Search([]float32{4, 5, 6}, 1)
	if err != nil {
		t.Fatalf("search after delete: %v", err)
	}
	if len(results) == 0 {
		t.Error("search returned no results after entry point deletion")
	}
}

func TestDeleteAll(t *testing.T) {
	idx := New(2, distance.Euclidean, DefaultConfig())
	_ = idx.Insert(1, []float32{1, 0})
	_ = idx.Insert(2, []float32{0, 1})
	idx.Delete(1)
	idx.Delete(2)
	if idx.Len() != 0 {
		t.Errorf("len = %d, want 0", idx.Len())
	}
}

func TestGetVector(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	_ = idx.Insert(1, []float32{1, 2, 3})
	v, ok := idx.GetVector(1)
	if !ok {
		t.Fatal("GetVector returned false")
	}
	if v[0] != 1 || v[1] != 2 || v[2] != 3 {
		t.Errorf("GetVector = %v, want [1, 2, 3]", v)
	}
	// Mutation should not affect stored vector
	v[0] = 99
	v2, _ := idx.GetVector(1)
	if v2[0] != 1 {
		t.Error("stored vector was mutated")
	}
}

func TestGetVectorNotFound(t *testing.T) {
	idx := New(3, distance.Euclidean, DefaultConfig())
	_, ok := idx.GetVector(999)
	if ok {
		t.Error("expected false for missing vector")
	}
}

func TestStats(t *testing.T) {
	idx := New(128, distance.Cosine, DefaultConfig())
	_ = idx.Insert(1, randomVector(128, rand.New(rand.NewSource(1))))
	s := idx.Stats()
	if s.Size != 1 || s.Dim != 128 {
		t.Errorf("stats = %+v", s)
	}
}

func TestRecallBruteForce(t *testing.T) {
	// Insert N random vectors, query with a random vector,
	// compare HNSW top-k against brute-force top-k.
	dim := 32
	n := 500
	k := 10
	rng := rand.New(rand.NewSource(42))

	cfg := DefaultConfig()
	cfg.EfConstruction = 100
	cfg.EfSearch = 50
	idx := New(dim, distance.Euclidean, cfg)

	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := randomVector(dim, rng)
		vectors[i] = v
		if err := idx.Insert(uint64(i), v); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	query := randomVector(dim, rng)

	// Brute force
	type scored struct {
		id   uint64
		dist float32
	}
	bf := make([]scored, n)
	distFn := distance.Get(distance.Euclidean)
	for i := 0; i < n; i++ {
		bf[i] = scored{uint64(i), distFn(query, vectors[i])}
	}
	// Sort
	for i := 0; i < len(bf); i++ {
		for j := i + 1; j < len(bf); j++ {
			if bf[j].dist < bf[i].dist {
				bf[i], bf[j] = bf[j], bf[i]
			}
		}
	}
	bfTop := make(map[uint64]bool)
	for i := 0; i < k && i < len(bf); i++ {
		bfTop[bf[i].id] = true
	}

	// HNSW search
	results, err := idx.Search(query, k)
	if err != nil {
		t.Fatalf("search: %v", err)
	}

	// Count recall
	hits := 0
	for _, r := range results {
		if bfTop[r.ID] {
			hits++
		}
	}
	recall := float64(hits) / float64(k)
	if recall < 0.7 {
		t.Errorf("recall@%d = %.2f, want >= 0.7", k, recall)
	}
}

func TestRecallCosine(t *testing.T) {
	dim := 64
	n := 300
	k := 5
	rng := rand.New(rand.NewSource(99))

	cfg := DefaultConfig()
	cfg.EfSearch = 100
	idx := New(dim, distance.Cosine, cfg)

	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := distance.Normalize(randomVector(dim, rng))
		vectors[i] = v
		_ = idx.Insert(uint64(i), v)
	}

	query := distance.Normalize(randomVector(dim, rng))
	distFn := distance.Get(distance.Cosine)

	type scored struct {
		id   uint64
		dist float32
	}
	bf := make([]scored, n)
	for i := 0; i < n; i++ {
		bf[i] = scored{uint64(i), distFn(query, vectors[i])}
	}
	for i := 0; i < len(bf); i++ {
		for j := i + 1; j < len(bf); j++ {
			if bf[j].dist < bf[i].dist {
				bf[i], bf[j] = bf[j], bf[i]
			}
		}
	}
	bfTop := make(map[uint64]bool)
	for i := 0; i < k; i++ {
		bfTop[bf[i].id] = true
	}

	results, _ := idx.Search(query, k)
	hits := 0
	for _, r := range results {
		if bfTop[r.ID] {
			hits++
		}
	}
	recall := float64(hits) / float64(k)
	if recall < 0.6 {
		t.Errorf("cosine recall@%d = %.2f, want >= 0.6", k, recall)
	}
}

func TestConcurrentInsertSearch(t *testing.T) {
	dim := 16
	idx := New(dim, distance.Euclidean, DefaultConfig())
	rng := rand.New(rand.NewSource(123))

	// Pre-insert some vectors
	for i := uint64(0); i < 50; i++ {
		_ = idx.Insert(i, randomVector(dim, rng))
	}

	var wg sync.WaitGroup
	// Concurrent inserts
	for i := uint64(50); i < 100; i++ {
		wg.Add(1)
		go func(id uint64) {
			defer wg.Done()
			v := make([]float32, dim)
			for j := range v {
				v[j] = float32(id) * 0.01
			}
			idx.Insert(id, v)
		}(i)
	}
	// Concurrent searches
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := make([]float32, dim)
			for j := range q {
				q[j] = 0.5
			}
			idx.Search(q, 5)
		}()
	}
	wg.Wait()
	if idx.Len() < 50 {
		t.Errorf("expected at least 50 vectors, got %d", idx.Len())
	}
}

func TestSearchEfParameter(t *testing.T) {
	dim := 16
	idx := New(dim, distance.Euclidean, DefaultConfig())
	rng := rand.New(rand.NewSource(7))
	for i := uint64(0); i < 100; i++ {
		_ = idx.Insert(i, randomVector(dim, rng))
	}
	// Low ef
	r1, _ := idx.SearchEf(randomVector(dim, rng), 5, 5)
	// High ef
	r2, _ := idx.SearchEf(randomVector(dim, rng), 5, 200)
	if len(r1) == 0 || len(r2) == 0 {
		t.Error("expected non-empty results")
	}
}

func TestLargeInsert(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large test")
	}
	dim := 64
	n := 2000
	rng := rand.New(rand.NewSource(55))
	cfg := DefaultConfig()
	cfg.M = 8
	cfg.EfConstruction = 50
	idx := New(dim, distance.Euclidean, cfg)

	for i := 0; i < n; i++ {
		_ = idx.Insert(uint64(i), randomVector(dim, rng))
	}
	if idx.Len() != n {
		t.Errorf("len = %d, want %d", idx.Len(), n)
	}
}

func TestResultsOrdered(t *testing.T) {
	dim := 16
	idx := New(dim, distance.Euclidean, DefaultConfig())
	rng := rand.New(rand.NewSource(77))
	for i := uint64(0); i < 200; i++ {
		_ = idx.Insert(i, randomVector(dim, rng))
	}
	results, _ := idx.Search(randomVector(dim, rng), 10)
	for i := 1; i < len(results); i++ {
		if results[i].Distance < results[i-1].Distance {
			t.Errorf("results not sorted: [%d].dist=%f < [%d].dist=%f",
				i, results[i].Distance, i-1, results[i-1].Distance)
		}
	}
}

func TestMetricAccessor(t *testing.T) {
	idx := New(3, distance.Manhattan, DefaultConfig())
	if idx.Metric() != distance.Manhattan {
		t.Errorf("metric = %v, want manhattan", idx.Metric())
	}
}

func TestSearchAfterDelete(t *testing.T) {
	dim := 8
	idx := New(dim, distance.Euclidean, DefaultConfig())
	rng := rand.New(rand.NewSource(11))

	for i := uint64(0); i < 50; i++ {
		_ = idx.Insert(i, randomVector(dim, rng))
	}
	// Delete half
	for i := uint64(0); i < 25; i++ {
		idx.Delete(i)
	}
	results, err := idx.Search(randomVector(dim, rng), 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	for _, r := range results {
		if r.ID < 25 {
			t.Errorf("found deleted ID %d in results", r.ID)
		}
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.M != 16 {
		t.Errorf("M = %d, want 16", cfg.M)
	}
	if cfg.Mmax != 32 {
		t.Errorf("Mmax = %d, want 32", cfg.Mmax)
	}
	expectedML := 1.0 / math.Log(16)
	if math.Abs(cfg.ML-expectedML) > 1e-10 {
		t.Errorf("ML = %f, want %f", cfg.ML, expectedML)
	}
}

func TestInsertManyThenSearchExact(t *testing.T) {
	dim := 4
	idx := New(dim, distance.Euclidean, DefaultConfig())
	target := []float32{5, 5, 5, 5}
	_ = idx.Insert(100, target)
	// Insert distant vectors
	for i := uint64(0); i < 99; i++ {
		v := []float32{float32(i) * 10, 0, 0, 0}
		_ = idx.Insert(i, v)
	}
	results, _ := idx.Search(target, 1)
	if len(results) != 1 || results[0].ID != 100 {
		t.Errorf("expected ID 100, got %v", results)
	}
}
