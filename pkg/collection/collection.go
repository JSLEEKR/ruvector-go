// Package collection provides named vector collections with HNSW indexing.
package collection

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/JSLEEKR/ruvector-go/pkg/distance"
	"github.com/JSLEEKR/ruvector-go/pkg/hnsw"
	"github.com/JSLEEKR/ruvector-go/pkg/store"
)

// Config defines collection parameters.
type Config struct {
	Name      string          `json:"name"`
	Dimension int             `json:"dimension"`
	Metric    distance.Metric `json:"metric"`
	HNSW      hnsw.Config     `json:"hnsw"`
}

// Collection is a named set of vectors with an HNSW index.
type Collection struct {
	mu     sync.RWMutex
	config Config
	index  *hnsw.Index
	store  *store.Store
}

// New creates a new collection.
func New(cfg Config, dataDir string) (*Collection, error) {
	if cfg.Name == "" {
		return nil, fmt.Errorf("collection name is required")
	}
	if cfg.Dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", cfg.Dimension)
	}

	storePath := filepath.Join(dataDir, cfg.Name+".dat")
	st, err := store.New(storePath)
	if err != nil {
		return nil, fmt.Errorf("create store: %w", err)
	}

	idx := hnsw.New(cfg.Dimension, cfg.Metric, cfg.HNSW)

	c := &Collection{
		config: cfg,
		index:  idx,
		store:  st,
	}

	// Rebuild index from existing store data
	entries := st.All()
	for _, e := range entries {
		if err := idx.Insert(e.ID, e.Vector); err != nil {
			return nil, fmt.Errorf("rebuild index for ID %d: %w", e.ID, err)
		}
	}

	return c, nil
}

// Insert adds a vector to the collection.
func (c *Collection) Insert(id uint64, vector []float32) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.index.Insert(id, vector); err != nil {
		return fmt.Errorf("index insert: %w", err)
	}
	c.store.Put(id, vector)
	return nil
}

// InsertBatch adds multiple vectors atomically.
// If any vector fails validation, no vectors are inserted.
func (c *Collection) InsertBatch(entries []store.VectorEntry) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Pre-validate: check dimensions, duplicates in batch, and existing IDs before modifying state.
	seen := make(map[uint64]bool, len(entries))
	for _, e := range entries {
		if len(e.Vector) != c.config.Dimension {
			return fmt.Errorf("index insert ID %d: dimension mismatch: got %d, want %d",
				e.ID, len(e.Vector), c.config.Dimension)
		}
		if seen[e.ID] {
			return fmt.Errorf("index insert ID %d: duplicate ID in batch", e.ID)
		}
		seen[e.ID] = true
		if _, exists := c.store.Get(e.ID); exists {
			return fmt.Errorf("index insert ID %d: vector ID %d already exists", e.ID, e.ID)
		}
	}

	for _, e := range entries {
		if err := c.index.Insert(e.ID, e.Vector); err != nil {
			return fmt.Errorf("index insert ID %d: %w", e.ID, err)
		}
		c.store.Put(e.ID, e.Vector)
	}
	return nil
}

// Search finds the k nearest neighbors.
func (c *Collection) Search(query []float32, k int) ([]hnsw.SearchResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.index.Search(query, k)
}

// SearchEf searches with a specific ef parameter.
func (c *Collection) SearchEf(query []float32, k, ef int) ([]hnsw.SearchResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.index.SearchEf(query, k, ef)
}

// Get retrieves a vector by ID.
func (c *Collection) Get(id uint64) ([]float32, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.store.Get(id)
}

// Delete removes a vector.
func (c *Collection) Delete(id uint64) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.index.Delete(id)
	return c.store.Delete(id)
}

// Len returns the number of vectors.
func (c *Collection) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.index.Len()
}

// Save persists the collection to disk.
func (c *Collection) Save() error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.store.Save()
}

// Config returns the collection configuration.
func (c *Collection) Config() Config {
	return c.config
}

// Stats returns collection statistics.
func (c *Collection) Stats() hnsw.Stats {
	return c.index.Stats()
}

// Manager manages multiple named collections.
type Manager struct {
	mu          sync.RWMutex
	collections map[string]*Collection
	dataDir     string
}

// NewManager creates a collection manager.
func NewManager(dataDir string) (*Manager, error) {
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, fmt.Errorf("create data dir: %w", err)
	}
	m := &Manager{
		collections: make(map[string]*Collection),
		dataDir:     dataDir,
	}

	// Load existing collections from metadata files
	metaPattern := filepath.Join(dataDir, "*.meta.json")
	matches, _ := filepath.Glob(metaPattern)
	for _, metaPath := range matches {
		data, err := os.ReadFile(metaPath)
		if err != nil {
			continue
		}
		var cfg Config
		if err := json.Unmarshal(data, &cfg); err != nil {
			continue
		}
		coll, err := New(cfg, dataDir)
		if err != nil {
			continue
		}
		m.collections[cfg.Name] = coll
	}

	return m, nil
}

// Create creates a new collection.
func (m *Manager) Create(cfg Config) (*Collection, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.collections[cfg.Name]; exists {
		return nil, fmt.Errorf("collection %q already exists", cfg.Name)
	}

	coll, err := New(cfg, m.dataDir)
	if err != nil {
		return nil, err
	}
	m.collections[cfg.Name] = coll

	// Save metadata
	metaPath := filepath.Join(m.dataDir, cfg.Name+".meta.json")
	data, _ := json.MarshalIndent(cfg, "", "  ")
	if err := os.WriteFile(metaPath, data, 0644); err != nil {
		return nil, fmt.Errorf("save metadata: %w", err)
	}

	return coll, nil
}

// Get returns a collection by name.
func (m *Manager) Get(name string) (*Collection, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	coll, ok := m.collections[name]
	if !ok {
		return nil, fmt.Errorf("collection %q not found", name)
	}
	return coll, nil
}

// Delete removes a collection and its data.
func (m *Manager) Delete(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.collections[name]; !ok {
		return fmt.Errorf("collection %q not found", name)
	}

	delete(m.collections, name)

	// Remove files
	os.Remove(filepath.Join(m.dataDir, name+".dat"))
	os.Remove(filepath.Join(m.dataDir, name+".meta.json"))

	return nil
}

// List returns all collection names.
func (m *Manager) List() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	names := make([]string, 0, len(m.collections))
	for name := range m.collections {
		names = append(names, name)
	}
	return names
}

// SaveAll persists all collections.
func (m *Manager) SaveAll() error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for name, coll := range m.collections {
		if err := coll.Save(); err != nil {
			return fmt.Errorf("save collection %q: %w", name, err)
		}
	}
	return nil
}
