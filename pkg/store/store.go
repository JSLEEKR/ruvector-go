// Package store provides persistence for vector data using a custom binary format.
// No external dependencies -- uses encoding/binary and os.
package store

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
)

// VectorEntry represents a stored vector with its ID.
type VectorEntry struct {
	ID     uint64
	Vector []float32
}

// Store handles saving and loading vectors to/from disk.
type Store struct {
	mu   sync.RWMutex
	path string
	data map[uint64][]float32
}

// New creates or opens a store at the given path.
func New(path string) (*Store, error) {
	// Validate path (no ".." for security)
	cleaned := filepath.Clean(path)
	for _, part := range splitPath(cleaned) {
		if part == ".." {
			return nil, fmt.Errorf("path traversal not allowed: %q", path)
		}
	}

	dir := filepath.Dir(cleaned)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("create directory: %w", err)
	}

	s := &Store{
		path: cleaned,
		data: make(map[uint64][]float32),
	}

	// Load existing data if file exists
	if _, err := os.Stat(cleaned); err == nil {
		if err := s.load(); err != nil {
			return nil, fmt.Errorf("load existing data: %w", err)
		}
	}

	return s, nil
}

// Put stores a vector.
func (s *Store) Put(id uint64, vector []float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v := make([]float32, len(vector))
	copy(v, vector)
	s.data[id] = v
}

// Get retrieves a vector by ID.
func (s *Store) Get(id uint64) ([]float32, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.data[id]
	if !ok {
		return nil, false
	}
	out := make([]float32, len(v))
	copy(out, v)
	return out, true
}

// Delete removes a vector by ID.
func (s *Store) Delete(id uint64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.data[id]
	if ok {
		delete(s.data, id)
	}
	return ok
}

// Len returns the number of stored vectors.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.data)
}

// All returns all stored vectors.
func (s *Store) All() []VectorEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	entries := make([]VectorEntry, 0, len(s.data))
	for id, v := range s.data {
		vc := make([]float32, len(v))
		copy(vc, v)
		entries = append(entries, VectorEntry{ID: id, Vector: vc})
	}
	return entries
}

// Save writes all vectors to disk in binary format using atomic write (write to temp, then rename).
// Format: [count:uint64] [id:uint64 dim:uint32 vector:float32*dim]...
func (s *Store) Save() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	tmpPath := s.path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}

	if err := binary.Write(f, binary.LittleEndian, uint64(len(s.data))); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("write count: %w", err)
	}

	for id, vec := range s.data {
		if err := binary.Write(f, binary.LittleEndian, id); err != nil {
			f.Close()
			os.Remove(tmpPath)
			return fmt.Errorf("write id: %w", err)
		}
		if err := binary.Write(f, binary.LittleEndian, uint32(len(vec))); err != nil {
			f.Close()
			os.Remove(tmpPath)
			return fmt.Errorf("write dim: %w", err)
		}
		for _, v := range vec {
			if err := binary.Write(f, binary.LittleEndian, math.Float32bits(v)); err != nil {
				f.Close()
				os.Remove(tmpPath)
				return fmt.Errorf("write vector: %w", err)
			}
		}
	}

	if err := f.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("close temp file: %w", err)
	}

	if err := os.Rename(tmpPath, s.path); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("rename temp file: %w", err)
	}

	return nil
}

// load reads vectors from disk.
func (s *Store) load() error {
	f, err := os.Open(s.path)
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	var count uint64
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		if err == io.EOF {
			return nil // empty file
		}
		return fmt.Errorf("read count: %w", err)
	}

	// Sanity limit: reject files claiming more than 10 million entries.
	const maxCount = 10_000_000
	if count > maxCount {
		return fmt.Errorf("entry count %d exceeds maximum %d", count, maxCount)
	}

	for i := uint64(0); i < count; i++ {
		var id uint64
		if err := binary.Read(f, binary.LittleEndian, &id); err != nil {
			return fmt.Errorf("read id: %w", err)
		}
		var dim uint32
		if err := binary.Read(f, binary.LittleEndian, &dim); err != nil {
			return fmt.Errorf("read dim: %w", err)
		}
		// Sanity limit: reject dimensions larger than 100K to prevent OOM.
		const maxDim = 100_000
		if dim > maxDim {
			return fmt.Errorf("dimension %d exceeds maximum %d", dim, maxDim)
		}
		vec := make([]float32, dim)
		for j := uint32(0); j < dim; j++ {
			var bits uint32
			if err := binary.Read(f, binary.LittleEndian, &bits); err != nil {
				return fmt.Errorf("read vector: %w", err)
			}
			vec[j] = math.Float32frombits(bits)
		}
		s.data[id] = vec
	}

	return nil
}

// Path returns the store file path.
func (s *Store) Path() string {
	return s.path
}

func splitPath(p string) []string {
	var parts []string
	for p != "" {
		dir, file := filepath.Split(p)
		if file != "" {
			parts = append([]string{file}, parts...)
		}
		if dir == p {
			break
		}
		p = filepath.Clean(dir)
	}
	return parts
}
