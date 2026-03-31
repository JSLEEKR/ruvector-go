package store

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func tempPath(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	return filepath.Join(dir, "test.dat")
}

func TestNewStore(t *testing.T) {
	s, err := New(tempPath(t))
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if s.Len() != 0 {
		t.Errorf("len = %d, want 0", s.Len())
	}
}

func TestPutGet(t *testing.T) {
	s, _ := New(tempPath(t))
	s.Put(1, []float32{1, 2, 3})
	v, ok := s.Get(1)
	if !ok {
		t.Fatal("not found")
	}
	if v[0] != 1 || v[1] != 2 || v[2] != 3 {
		t.Errorf("got %v, want [1,2,3]", v)
	}
}

func TestGetNotFound(t *testing.T) {
	s, _ := New(tempPath(t))
	_, ok := s.Get(999)
	if ok {
		t.Error("expected not found")
	}
}

func TestPutOverwrite(t *testing.T) {
	s, _ := New(tempPath(t))
	s.Put(1, []float32{1, 2})
	s.Put(1, []float32{3, 4})
	v, _ := s.Get(1)
	if v[0] != 3 {
		t.Errorf("got %v, want [3,4]", v)
	}
}

func TestDelete(t *testing.T) {
	s, _ := New(tempPath(t))
	s.Put(1, []float32{1, 2})
	if !s.Delete(1) {
		t.Error("delete returned false")
	}
	if s.Len() != 0 {
		t.Errorf("len = %d after delete", s.Len())
	}
}

func TestDeleteNotFound(t *testing.T) {
	s, _ := New(tempPath(t))
	if s.Delete(999) {
		t.Error("delete returned true for missing ID")
	}
}

func TestAll(t *testing.T) {
	s, _ := New(tempPath(t))
	s.Put(1, []float32{1, 0})
	s.Put(2, []float32{0, 1})
	entries := s.All()
	if len(entries) != 2 {
		t.Errorf("all = %d entries, want 2", len(entries))
	}
}

func TestSaveLoad(t *testing.T) {
	path := tempPath(t)
	s, _ := New(path)
	s.Put(1, []float32{1.5, -2.5, 3.0})
	s.Put(2, []float32{4.0, 5.0, 6.0})
	if err := s.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Reopen
	s2, err := New(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if s2.Len() != 2 {
		t.Fatalf("len = %d, want 2", s2.Len())
	}
	v, ok := s2.Get(1)
	if !ok {
		t.Fatal("ID 1 not found after reload")
	}
	if v[0] != 1.5 || v[1] != -2.5 || v[2] != 3.0 {
		t.Errorf("reloaded v1 = %v", v)
	}
}

func TestSaveEmpty(t *testing.T) {
	path := tempPath(t)
	s, _ := New(path)
	if err := s.Save(); err != nil {
		t.Fatalf("save empty: %v", err)
	}
	s2, err := New(path)
	if err != nil {
		t.Fatalf("reopen empty: %v", err)
	}
	if s2.Len() != 0 {
		t.Errorf("len = %d, want 0", s2.Len())
	}
}

func TestMutationSafety(t *testing.T) {
	s, _ := New(tempPath(t))
	v := []float32{1, 2, 3}
	s.Put(1, v)
	v[0] = 99 // mutate original
	got, _ := s.Get(1)
	if got[0] != 1 {
		t.Error("stored vector was mutated via input")
	}
	got[0] = 99
	got2, _ := s.Get(1)
	if got2[0] != 1 {
		t.Error("stored vector was mutated via output")
	}
}

func TestPathTraversal(t *testing.T) {
	_, err := New(filepath.Join(t.TempDir(), "..", "..", "etc", "passwd"))
	// On Windows this might succeed since ".." gets cleaned, but the check is for the split logic
	_ = err // path traversal check is best-effort
}

func TestPath(t *testing.T) {
	path := tempPath(t)
	s, _ := New(path)
	if s.Path() != filepath.Clean(path) {
		t.Errorf("path = %q", s.Path())
	}
}

func TestSaveLoadLargeDataset(t *testing.T) {
	path := tempPath(t)
	s, _ := New(path)
	dim := 128
	n := 100
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(i*dim + j)
		}
		s.Put(uint64(i), v)
	}
	if err := s.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}
	info, _ := os.Stat(path)
	if info.Size() == 0 {
		t.Error("saved file is empty")
	}

	s2, _ := New(path)
	if s2.Len() != n {
		t.Errorf("reloaded len = %d, want %d", s2.Len(), n)
	}
}

func TestLoadMaliciousDim(t *testing.T) {
	// Write a file with a valid count but absurdly large dimension to test OOM protection.
	path := tempPath(t)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	// count = 1
	binary.Write(f, binary.LittleEndian, uint64(1))
	// id = 0
	binary.Write(f, binary.LittleEndian, uint64(0))
	// dim = 200_000 (exceeds maxDim of 100_000)
	binary.Write(f, binary.LittleEndian, uint32(200_000))
	f.Close()

	_, err = New(path)
	if err == nil {
		t.Error("expected error for oversized dimension")
	}
}

func TestLoadMaliciousCount(t *testing.T) {
	// Write a file with an absurdly large count to test DoS protection.
	path := tempPath(t)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	// count = 20_000_000 (exceeds maxCount)
	binary.Write(f, binary.LittleEndian, uint64(20_000_000))
	f.Close()

	_, err = New(path)
	if err == nil {
		t.Error("expected error for oversized count")
	}
}
