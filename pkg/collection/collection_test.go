package collection

import (
	"testing"

	"github.com/JSLEEKR/ruvector-go/pkg/distance"
	"github.com/JSLEEKR/ruvector-go/pkg/hnsw"
	"github.com/JSLEEKR/ruvector-go/pkg/store"
)

func testConfig(name string) Config {
	return Config{
		Name:      name,
		Dimension: 4,
		Metric:    distance.Euclidean,
		HNSW:      hnsw.DefaultConfig(),
	}
}

func TestNewCollection(t *testing.T) {
	c, err := New(testConfig("test"), t.TempDir())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if c.Len() != 0 {
		t.Errorf("len = %d, want 0", c.Len())
	}
}

func TestNewCollectionNoName(t *testing.T) {
	cfg := testConfig("")
	_, err := New(cfg, t.TempDir())
	if err == nil {
		t.Error("expected error for empty name")
	}
}

func TestNewCollectionBadDim(t *testing.T) {
	cfg := testConfig("test")
	cfg.Dimension = 0
	_, err := New(cfg, t.TempDir())
	if err == nil {
		t.Error("expected error for zero dimension")
	}
}

func TestInsertSearch(t *testing.T) {
	c, _ := New(testConfig("test"), t.TempDir())
	_ = c.Insert(1, []float32{0, 0, 0, 0})
	_ = c.Insert(2, []float32{1, 1, 1, 1})
	_ = c.Insert(3, []float32{10, 10, 10, 10})

	results, err := c.Search([]float32{0, 0, 0, 0}, 2)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("results = %d, want 2", len(results))
	}
	if results[0].ID != 1 {
		t.Errorf("nearest = %d, want 1", results[0].ID)
	}
}

func TestInsertBatch(t *testing.T) {
	c, _ := New(testConfig("test"), t.TempDir())
	entries := []store.VectorEntry{
		{ID: 1, Vector: []float32{1, 0, 0, 0}},
		{ID: 2, Vector: []float32{0, 1, 0, 0}},
		{ID: 3, Vector: []float32{0, 0, 1, 0}},
	}
	if err := c.InsertBatch(entries); err != nil {
		t.Fatalf("batch insert: %v", err)
	}
	if c.Len() != 3 {
		t.Errorf("len = %d, want 3", c.Len())
	}
}

func TestGet(t *testing.T) {
	c, _ := New(testConfig("test"), t.TempDir())
	_ = c.Insert(1, []float32{1, 2, 3, 4})
	v, ok := c.Get(1)
	if !ok {
		t.Fatal("not found")
	}
	if v[0] != 1 || v[3] != 4 {
		t.Errorf("got %v", v)
	}
}

func TestGetNotFound(t *testing.T) {
	c, _ := New(testConfig("test"), t.TempDir())
	_, ok := c.Get(999)
	if ok {
		t.Error("expected not found")
	}
}

func TestDeleteFromCollection(t *testing.T) {
	c, _ := New(testConfig("test"), t.TempDir())
	_ = c.Insert(1, []float32{1, 2, 3, 4})
	if !c.Delete(1) {
		t.Error("delete returned false")
	}
	if c.Len() != 0 {
		t.Errorf("len = %d, want 0", c.Len())
	}
}

func TestSaveAndReload(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig("persist")
	c, _ := New(cfg, dir)
	_ = c.Insert(1, []float32{1, 2, 3, 4})
	_ = c.Insert(2, []float32{5, 6, 7, 8})
	if err := c.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Reopen
	c2, err := New(cfg, dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if c2.Len() != 2 {
		t.Errorf("len after reload = %d, want 2", c2.Len())
	}
	// Should be searchable
	results, _ := c2.Search([]float32{1, 2, 3, 4}, 1)
	if len(results) != 1 || results[0].ID != 1 {
		t.Errorf("search after reload: %v", results)
	}
}

func TestSearchEf(t *testing.T) {
	c, _ := New(testConfig("test"), t.TempDir())
	_ = c.Insert(1, []float32{0, 0, 0, 0})
	_ = c.Insert(2, []float32{1, 1, 1, 1})
	results, err := c.SearchEf([]float32{0, 0, 0, 0}, 1, 100)
	if err != nil {
		t.Fatalf("searchEf: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("results = %d, want 1", len(results))
	}
}

func TestCollectionConfig(t *testing.T) {
	cfg := testConfig("mytest")
	c, _ := New(cfg, t.TempDir())
	if c.Config().Name != "mytest" {
		t.Errorf("name = %q", c.Config().Name)
	}
}

func TestCollectionStats(t *testing.T) {
	c, _ := New(testConfig("test"), t.TempDir())
	_ = c.Insert(1, []float32{1, 0, 0, 0})
	s := c.Stats()
	if s.Size != 1 || s.Dim != 4 {
		t.Errorf("stats = %+v", s)
	}
}

// --- Manager ---

func TestManagerCreate(t *testing.T) {
	m, err := NewManager(t.TempDir())
	if err != nil {
		t.Fatalf("new manager: %v", err)
	}

	_, err = m.Create(testConfig("col1"))
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if len(m.List()) != 1 {
		t.Errorf("list = %d, want 1", len(m.List()))
	}
}

func TestManagerCreateDuplicate(t *testing.T) {
	m, _ := NewManager(t.TempDir())
	_, _ = m.Create(testConfig("col1"))
	_, err := m.Create(testConfig("col1"))
	if err == nil {
		t.Error("expected error for duplicate")
	}
}

func TestManagerGet(t *testing.T) {
	m, _ := NewManager(t.TempDir())
	_, _ = m.Create(testConfig("col1"))
	c, err := m.Get("col1")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if c.Config().Name != "col1" {
		t.Errorf("name = %q", c.Config().Name)
	}
}

func TestManagerGetNotFound(t *testing.T) {
	m, _ := NewManager(t.TempDir())
	_, err := m.Get("missing")
	if err == nil {
		t.Error("expected not found error")
	}
}

func TestManagerDelete(t *testing.T) {
	m, _ := NewManager(t.TempDir())
	_, _ = m.Create(testConfig("col1"))
	if err := m.Delete("col1"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if len(m.List()) != 0 {
		t.Errorf("list = %d after delete", len(m.List()))
	}
}

func TestManagerDeleteNotFound(t *testing.T) {
	m, _ := NewManager(t.TempDir())
	err := m.Delete("missing")
	if err == nil {
		t.Error("expected not found error")
	}
}

func TestManagerSaveAll(t *testing.T) {
	m, _ := NewManager(t.TempDir())
	c, _ := m.Create(testConfig("col1"))
	_ = c.Insert(1, []float32{1, 2, 3, 4})
	if err := m.SaveAll(); err != nil {
		t.Fatalf("save all: %v", err)
	}
}

func TestManagerPersistence(t *testing.T) {
	dir := t.TempDir()
	m, _ := NewManager(dir)
	c, _ := m.Create(testConfig("persist"))
	_ = c.Insert(1, []float32{1, 2, 3, 4})
	_ = m.SaveAll()

	// Reopen manager
	m2, err := NewManager(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if len(m2.List()) != 1 {
		t.Fatalf("list = %d, want 1", len(m2.List()))
	}
	c2, _ := m2.Get("persist")
	if c2.Len() != 1 {
		t.Errorf("len = %d, want 1", c2.Len())
	}
}
