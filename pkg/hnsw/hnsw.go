// Package hnsw implements a Hierarchical Navigable Small World graph index
// for approximate nearest neighbor search. Built from scratch.
package hnsw

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/JSLEEKR/ruvector-go/pkg/distance"
)

// Config holds HNSW construction and search parameters.
type Config struct {
	M              int     // max connections per layer (default 16)
	Mmax           int     // max connections at layer 0 (default 2*M)
	EfConstruction int     // build-time candidate list size (default 200)
	EfSearch       int     // query-time candidate list size (default 50)
	ML             float64 // level multiplier (default 1/ln(M))
}

// DefaultConfig returns reasonable defaults for HNSW.
func DefaultConfig() Config {
	m := 16
	return Config{
		M:              m,
		Mmax:           2 * m,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0 / math.Log(float64(m)),
	}
}

// node represents a single vector in the HNSW graph.
type node struct {
	id      uint64
	vector  []float32
	level   int
	friends [][]uint64 // friends[layer] = list of neighbor IDs
}

// Index is a thread-safe HNSW index.
type Index struct {
	mu         sync.RWMutex
	nodes      map[uint64]*node
	entryPoint uint64
	maxLevel   int
	config     Config
	distFn     distance.Func
	metric     distance.Metric
	dim        int
	rng        *rand.Rand
	size       int
}

// SearchResult holds a single search result.
type SearchResult struct {
	ID       uint64
	Distance float32
}

// New creates a new HNSW index.
func New(dim int, metric distance.Metric, cfg Config) *Index {
	if cfg.M <= 0 {
		cfg.M = 16
	}
	if cfg.Mmax <= 0 {
		cfg.Mmax = 2 * cfg.M
	}
	if cfg.EfConstruction <= 0 {
		cfg.EfConstruction = 200
	}
	if cfg.EfSearch <= 0 {
		cfg.EfSearch = 50
	}
	if cfg.ML <= 0 {
		cfg.ML = 1.0 / math.Log(float64(cfg.M))
	}
	return &Index{
		nodes:    make(map[uint64]*node),
		config:   cfg,
		distFn:   distance.Get(metric),
		metric:   metric,
		dim:      dim,
		rng:      rand.New(rand.NewSource(42)),
		maxLevel: -1,
	}
}

// randomLevel assigns a level using exponential distribution.
func (idx *Index) randomLevel() int {
	r := idx.rng.Float64()
	if r == 0 {
		r = 1e-10
	}
	return int(math.Floor(-math.Log(r) * idx.config.ML))
}

// Insert adds a vector to the index. Returns error if ID already exists or dimension mismatches.
func (idx *Index) Insert(id uint64, vector []float32) error {
	if len(vector) != idx.dim {
		return fmt.Errorf("dimension mismatch: got %d, want %d", len(vector), idx.dim)
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if _, exists := idx.nodes[id]; exists {
		return fmt.Errorf("vector ID %d already exists", id)
	}

	level := idx.randomLevel()
	n := &node{
		id:      id,
		vector:  make([]float32, len(vector)),
		level:   level,
		friends: make([][]uint64, level+1),
	}
	copy(n.vector, vector)
	for i := range n.friends {
		n.friends[i] = nil
	}

	idx.nodes[id] = n
	idx.size++

	// First node: set as entry point.
	if idx.size == 1 {
		idx.entryPoint = id
		idx.maxLevel = level
		return nil
	}

	ep := idx.entryPoint
	epNode := idx.nodes[ep]

	// Phase 1: greedy search from top to level+1
	currBest := ep
	currDist := idx.distFn(vector, epNode.vector)
	for lc := idx.maxLevel; lc > level; lc-- {
		changed := true
		for changed {
			changed = false
			if lc < len(idx.nodes[currBest].friends) {
				for _, friendID := range idx.nodes[currBest].friends[lc] {
					if friendNode, ok := idx.nodes[friendID]; ok {
						d := idx.distFn(vector, friendNode.vector)
						if d < currDist {
							currBest = friendID
							currDist = d
							changed = true
						}
					}
				}
			}
		}
	}

	// Phase 2: insert at layers min(level, maxLevel) down to 0
	for lc := min(level, idx.maxLevel); lc >= 0; lc-- {
		candidates := idx.searchLayer(vector, currBest, idx.config.EfConstruction, lc)

		// Select M best neighbors
		neighbors := idx.selectNeighborsSimple(candidates, idx.config.M)

		// Connect n to selected neighbors
		n.friends[lc] = make([]uint64, 0, len(neighbors))
		for _, nb := range neighbors {
			n.friends[lc] = append(n.friends[lc], nb.id)
		}

		// Add bidirectional edges and prune if necessary
		mmax := idx.config.Mmax
		if lc == 0 {
			mmax = idx.config.Mmax
		} else {
			mmax = idx.config.M
		}
		for _, nb := range neighbors {
			nbNode := idx.nodes[nb.id]
			if lc >= len(nbNode.friends) {
				// Expand friends slices if needed
				for len(nbNode.friends) <= lc {
					nbNode.friends = append(nbNode.friends, nil)
				}
			}
			nbNode.friends[lc] = append(nbNode.friends[lc], id)
			if len(nbNode.friends[lc]) > mmax {
				// Prune: keep only closest mmax neighbors
				nbNode.friends[lc] = idx.pruneConnections(nbNode, lc, mmax)
			}
		}

		if len(candidates) > 0 {
			currBest = candidates[0].id
		}
	}

	// Update entry point if new node has higher level
	if level > idx.maxLevel {
		idx.entryPoint = id
		idx.maxLevel = level
	}

	return nil
}

// searchLayer performs beam search at a given layer, returning ef closest nodes.
func (idx *Index) searchLayer(query []float32, entryID uint64, ef int, layer int) []candidateItem {
	entryNode := idx.nodes[entryID]
	entryDist := idx.distFn(query, entryNode.vector)

	visited := make(map[uint64]bool)
	visited[entryID] = true

	// candidates: min-heap (closest first for expansion)
	candidates := &candidateHeap{}
	heap.Init(candidates)
	heap.Push(candidates, candidateItem{id: entryID, dist: entryDist})

	// results: max-heap (farthest first for eviction)
	results := &candidateHeap{maxHeap: true}
	heap.Init(results)
	heap.Push(results, candidateItem{id: entryID, dist: entryDist})

	for candidates.Len() > 0 {
		c := heap.Pop(candidates).(candidateItem)
		farthestResult := results.items[0]
		if c.dist > farthestResult.dist {
			break
		}

		cNode := idx.nodes[c.id]
		if layer < len(cNode.friends) {
			for _, friendID := range cNode.friends[layer] {
				if visited[friendID] {
					continue
				}
				visited[friendID] = true

				friendNode, ok := idx.nodes[friendID]
				if !ok {
					continue
				}
				d := idx.distFn(query, friendNode.vector)

				if results.Len() < ef || d < results.items[0].dist {
					heap.Push(candidates, candidateItem{id: friendID, dist: d})
					heap.Push(results, candidateItem{id: friendID, dist: d})
					if results.Len() > ef {
						heap.Pop(results)
					}
				}
			}
		}
	}

	// Extract results sorted by distance (ascending)
	out := make([]candidateItem, results.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(results).(candidateItem)
	}
	return out
}

// selectNeighborsSimple picks the M closest candidates.
func (idx *Index) selectNeighborsSimple(candidates []candidateItem, m int) []candidateItem {
	if len(candidates) <= m {
		return candidates
	}
	return candidates[:m]
}

// pruneConnections keeps only the mmax closest connections for a node at a given layer.
func (idx *Index) pruneConnections(n *node, layer, mmax int) []uint64 {
	friends := n.friends[layer]
	if len(friends) <= mmax {
		return friends
	}

	type fd struct {
		id   uint64
		dist float32
	}
	scored := make([]fd, 0, len(friends))
	for _, fid := range friends {
		if fNode, ok := idx.nodes[fid]; ok {
			d := idx.distFn(n.vector, fNode.vector)
			scored = append(scored, fd{id: fid, dist: d})
		}
	}

	// Sort by distance
	for i := 0; i < len(scored); i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].dist < scored[i].dist {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}

	if len(scored) > mmax {
		scored = scored[:mmax]
	}
	result := make([]uint64, len(scored))
	for i, s := range scored {
		result[i] = s.id
	}
	return result
}

// Search finds the k nearest neighbors to the query vector.
func (idx *Index) Search(query []float32, k int) ([]SearchResult, error) {
	return idx.SearchEf(query, k, 0)
}

// SearchEf finds the k nearest neighbors using the specified ef value.
// If ef <= 0, uses the configured EfSearch.
func (idx *Index) SearchEf(query []float32, k int, ef int) ([]SearchResult, error) {
	if len(query) != idx.dim {
		return nil, fmt.Errorf("dimension mismatch: got %d, want %d", len(query), idx.dim)
	}
	if k <= 0 {
		return nil, fmt.Errorf("k must be positive, got %d", k)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.size == 0 {
		return nil, nil
	}

	if ef <= 0 {
		ef = idx.config.EfSearch
	}
	if ef < k {
		ef = k
	}

	ep := idx.entryPoint

	// Phase 1: greedy descent from top layer to layer 1
	currBest := ep
	currDist := idx.distFn(query, idx.nodes[ep].vector)
	for lc := idx.maxLevel; lc > 0; lc-- {
		changed := true
		for changed {
			changed = false
			if lc < len(idx.nodes[currBest].friends) {
				for _, friendID := range idx.nodes[currBest].friends[lc] {
					if friendNode, ok := idx.nodes[friendID]; ok {
						d := idx.distFn(query, friendNode.vector)
						if d < currDist {
							currBest = friendID
							currDist = d
							changed = true
						}
					}
				}
			}
		}
	}

	// Phase 2: beam search at layer 0
	candidates := idx.searchLayer(query, currBest, ef, 0)

	if len(candidates) > k {
		candidates = candidates[:k]
	}

	results := make([]SearchResult, len(candidates))
	for i, c := range candidates {
		results[i] = SearchResult{ID: c.id, Distance: c.dist}
	}
	return results, nil
}

// Delete removes a vector from the index. Returns false if not found.
// Uses lazy deletion: removes node from graph but does not repair edges.
func (idx *Index) Delete(id uint64) bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	n, ok := idx.nodes[id]
	if !ok {
		return false
	}

	// Remove edges pointing to this node
	for layer := 0; layer <= n.level; layer++ {
		if layer < len(n.friends) {
			for _, friendID := range n.friends[layer] {
				if friendNode, ok := idx.nodes[friendID]; ok {
					if layer < len(friendNode.friends) {
						friendNode.friends[layer] = removeID(friendNode.friends[layer], id)
					}
				}
			}
		}
	}

	delete(idx.nodes, id)
	idx.size--

	// Update entry point if necessary
	if idx.size == 0 {
		idx.maxLevel = -1
		return true
	}
	if id == idx.entryPoint {
		// Find new entry point with highest level
		bestLevel := -1
		for nid, nn := range idx.nodes {
			if nn.level > bestLevel {
				bestLevel = nn.level
				idx.entryPoint = nid
			}
		}
		idx.maxLevel = bestLevel
	}

	return true
}

func removeID(slice []uint64, id uint64) []uint64 {
	for i, v := range slice {
		if v == id {
			return append(slice[:i], slice[i+1:]...)
		}
	}
	return slice
}

// Len returns the number of vectors in the index.
func (idx *Index) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.size
}

// GetVector returns the stored vector for a given ID.
func (idx *Index) GetVector(id uint64) ([]float32, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	n, ok := idx.nodes[id]
	if !ok {
		return nil, false
	}
	out := make([]float32, len(n.vector))
	copy(out, n.vector)
	return out, true
}

// Dimension returns the vector dimension of the index.
func (idx *Index) Dimension() int {
	return idx.dim
}

// Metric returns the distance metric used by the index.
func (idx *Index) Metric() distance.Metric {
	return idx.metric
}

// Stats returns index statistics.
type Stats struct {
	Size     int
	MaxLevel int
	Dim      int
	M        int
}

// Stats returns current index statistics.
func (idx *Index) Stats() Stats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return Stats{
		Size:     idx.size,
		MaxLevel: idx.maxLevel,
		Dim:      idx.dim,
		M:        idx.config.M,
	}
}

