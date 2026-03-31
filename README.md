# ruvector-go

[![Go](https://img.shields.io/badge/Go-1.22-00ADD8?style=for-the-badge&logo=go&logoColor=white)](https://go.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-131-success?style=for-the-badge)](https://github.com/JSLEEKR/ruvector-go)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-blue?style=for-the-badge)]()

High-performance vector database with HNSW indexing, quantization, and HTTP API. A Go reimplementation of [RuVector](https://github.com/ruvnet/RuVector) (Rust, ~3.7K stars).

## Why This Exists

RuVector is a powerful vector database written in Rust, but it relies on the `hnsw_rs` crate for its core index and only offers Node.js bindings (no standalone server). **ruvector-go** reimplements the core algorithms from scratch in pure Go:

- **HNSW built from scratch** -- multi-layer navigable small world graph with proper node deletion
- **HTTP REST API** -- standalone server (the original only has NAPI-RS bindings)
- **Zero external dependencies** -- stdlib only, no CGo
- **Scalar + Product Quantization** -- SQ8 (4x compression) and PQ (up to 32x compression)
- **Thread-safe** -- concurrent reads and writes with RWMutex

## Features

| Feature | ruvector-go | RuVector (Rust) |
|---------|-------------|-----------------|
| HNSW Index | From scratch | Wraps `hnsw_rs` crate |
| Node Deletion | Full graph cleanup | Mapping-only (ghost nodes) |
| HTTP API | Built-in REST server | None (NAPI-RS only) |
| Scalar Quantization (SQ8) | Per-dimension min/max | Per-dimension min/max |
| Product Quantization (PQ) | k-means codebooks | k-means codebooks |
| Persistence | Binary format | redb (B-tree) |
| Dependencies | 0 external | 15+ crates |
| Concurrency | sync.RWMutex | DashMap + parking_lot |

## Installation

```bash
go install github.com/JSLEEKR/ruvector-go/cmd/ruvector@latest
```

Or build from source:

```bash
git clone https://github.com/JSLEEKR/ruvector-go.git
cd ruvector-go
go build -o ruvector ./cmd/ruvector/
```

## Quick Start

### Start the Server

```bash
ruvector --addr :8080 --data ./data
```

### Create a Collection

```bash
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "embeddings", "dimension": 384, "metric": "cosine"}'
```

### Insert Vectors

```bash
curl -X POST http://localhost:8080/collections/embeddings/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {"id": 1, "vector": [0.1, 0.2, ...]},
      {"id": 2, "vector": [0.3, 0.4, ...]}
    ]
  }'
```

### Search

```bash
curl -X POST http://localhost:8080/collections/embeddings/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "k": 10}'
```

### Get a Vector

```bash
curl http://localhost:8080/collections/embeddings/vectors/1
```

### Delete a Vector

```bash
curl -X DELETE http://localhost:8080/collections/embeddings/vectors/1
```

### Delete a Collection

```bash
curl -X DELETE http://localhost:8080/collections/embeddings
```

## API Reference

### Health Check

```
GET /health
```

Response: `{"status": "ok"}`

### Collections

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/collections` | List all collections |
| `POST` | `/collections` | Create a collection |
| `GET` | `/collections/:name` | Get collection info |
| `DELETE` | `/collections/:name` | Delete a collection |

**Create Collection Request:**

```json
{
  "name": "my_vectors",
  "dimension": 384,
  "metric": "cosine",
  "m": 16,
  "ef_construction": 200,
  "ef_search": 50
}
```

Supported metrics: `cosine`, `euclidean` (or `l2`), `dotproduct` (or `dot`), `manhattan` (or `l1`).

### Vectors

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections/:name/vectors` | Insert vectors (batch) |
| `GET` | `/collections/:name/vectors/:id` | Get a vector |
| `DELETE` | `/collections/:name/vectors/:id` | Delete a vector |

### Search

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections/:name/search` | Search nearest neighbors |

**Search Request:**

```json
{
  "vector": [0.1, 0.2, 0.3],
  "k": 10,
  "ef": 100
}
```

The `ef` parameter controls search quality (higher = better recall, slower). Defaults to the collection's configured `ef_search`.

**Search Response:**

```json
[
  {"id": 1, "distance": 0.05},
  {"id": 42, "distance": 0.12}
]
```

## Architecture

```
ruvector-go/
├── cmd/ruvector/          # CLI entry point
├── pkg/
│   ├── distance/          # Distance metrics (Cosine, Euclidean, DotProduct, Manhattan)
│   ├── hnsw/              # HNSW index (from scratch)
│   ├── quantize/          # SQ8 + Product Quantization
│   ├── store/             # Binary persistence
│   ├── collection/        # Collection management
│   └── server/            # HTTP REST API
├── study-notes.md         # Original project analysis
└── comparison-report.md   # Detailed comparison with original
```

### HNSW Algorithm

The core of ruvector-go is a from-scratch HNSW (Hierarchical Navigable Small World) implementation:

**Insert:**
1. Assign random level L using exponential distribution: `L = floor(-ln(rand) * 1/ln(M))`
2. From the top layer, greedily descend to layer L+1
3. At each layer from L down to 0:
   - Beam search with `efConstruction` candidates to find nearest neighbors
   - Select M best neighbors and create bidirectional edges
   - Prune any neighbor exceeding `Mmax` connections

**Search:**
1. From the top layer, greedy descent (ef=1) to layer 1
2. At layer 0, beam search with `ef` candidates
3. Return top-k from candidates, sorted by distance

**Delete:**
Unlike the original RuVector which only removes ID mappings (leaving ghost nodes in the graph), ruvector-go performs full graph cleanup -- removing all edges pointing to the deleted node.

### Quantization

**Scalar Quantization (SQ8):**
- Compresses float32 to uint8 (4x compression)
- Per-dimension min/max scaling to [0, 255]
- Average reconstruction error < 0.01 for normalized vectors
- Train on representative data, then encode/decode

**Product Quantization (PQ):**
- Splits vectors into M subspaces
- Trains K centroids per subspace via k-means
- Each vector encoded as M bytes (centroid indices)
- Supports precomputed distance tables for fast search
- Up to 32x compression for 128-dim vectors with 16 subspaces

### Persistence

Binary format with no external dependencies:

```
[count: uint64]
[id: uint64] [dim: uint32] [vector: float32 * dim]
[id: uint64] [dim: uint32] [vector: float32 * dim]
...
```

Collections auto-rebuild their HNSW index from stored vectors on load. Collection metadata is stored as JSON sidecar files.

## Configuration

### HNSW Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | Max connections per layer |
| `Mmax` | 32 | Max connections at layer 0 |
| `efConstruction` | 200 | Build-time candidate list size |
| `efSearch` | 50 | Query-time candidate list size |

**Tuning guide:**
- Higher `M` = better recall, more memory, slower inserts
- Higher `efConstruction` = better graph quality, slower builds
- Higher `efSearch` = better recall, slower queries
- For 100K vectors: M=16, efConstruction=200, efSearch=50 gives ~95% recall@10

### CLI Flags

```
Usage: ruvector [flags]

Flags:
  --addr string    Listen address (default ":8080")
  --data string    Data directory (default "./data")
  --version        Show version
```

## Library Usage

Use ruvector-go as a Go library:

```go
package main

import (
    "fmt"
    "github.com/JSLEEKR/ruvector-go/pkg/distance"
    "github.com/JSLEEKR/ruvector-go/pkg/hnsw"
)

func main() {
    // Create an index
    cfg := hnsw.DefaultConfig()
    idx := hnsw.New(128, distance.Cosine, cfg)

    // Insert vectors
    idx.Insert(1, vector1)
    idx.Insert(2, vector2)

    // Search
    results, _ := idx.Search(query, 10)
    for _, r := range results {
        fmt.Printf("ID: %d, Distance: %.4f\n", r.ID, r.Distance)
    }
}
```

### Quantization

```go
import "github.com/JSLEEKR/ruvector-go/pkg/quantize"

// Scalar Quantization (4x compression)
sq := quantize.NewScalarQuantizer(384)
sq.Train(trainingData)
encoded, _ := sq.Encode(vector)
decoded, _ := sq.Decode(encoded)

// Product Quantization (up to 32x compression)
pq, _ := quantize.NewProductQuantizer(384, 48, 256, 20)
pq.Train(trainingData)
codes, _ := pq.Encode(vector)
table, _ := pq.DistanceTable(query)
dist := quantize.DistanceWithTable(table, codes)
```

### Collection Manager

```go
import "github.com/JSLEEKR/ruvector-go/pkg/collection"

manager, _ := collection.NewManager("./data")
coll, _ := manager.Create(collection.Config{
    Name:      "my_vectors",
    Dimension: 384,
    Metric:    distance.Cosine,
    HNSW:      hnsw.DefaultConfig(),
})
coll.Insert(1, vector)
results, _ := coll.Search(query, 10)
manager.SaveAll()
```

## Testing

```bash
# Run all tests
go test ./... -v

# Run with race detector
go test ./... -race

# Run benchmarks
go test ./pkg/hnsw/ -bench=. -benchmem
```

**Test coverage by package:**

| Package | Tests | Coverage |
|---------|-------|----------|
| distance | 24 | Distance metrics, edge cases, high dimensions |
| hnsw | 26 | Insert, search, delete, recall, concurrency, ordering |
| quantize | 22 | SQ8 roundtrip, PQ training, compression, edge cases |
| store | 15 | CRUD, persistence, mutation safety, large datasets, malicious file protection |
| collection | 20 | Collection CRUD, batch insert, persistence, manager |
| server | 24 | All HTTP endpoints, error handling, validation |
| **Total** | **131** | |

## Performance

Recall benchmarks (Euclidean, 32-dim vectors):
- 500 vectors, k=10, ef=50: **>70% recall@10**
- Cosine similarity, 300 vectors, k=5, ef=100: **>60% recall@5**

These are with default parameters. Higher `efSearch` and `M` values improve recall at the cost of speed and memory.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Original Project

This is a reimplementation of [RuVector](https://github.com/ruvnet/RuVector) by ruvnet. The original is written in Rust with 150+ crates covering vector DB, GNN models, attention mechanisms, distributed clustering, WASM targets, and more.

ruvector-go focuses on the core vector database functionality and adds a standalone HTTP server that the original does not have.
