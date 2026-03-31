# Comparison Report: ruvector-go vs RuVector

## Overview

| Aspect | RuVector (Original) | ruvector-go (Ours) |
|--------|---------------------|---------------------|
| Language | Rust | Go |
| Stars | ~3.7K | — |
| Dependencies | 87+ crates (RocksDB, HNSW crate, etc.) | 0 (stdlib only) |
| HNSW | Wrapped (`hnsw_rs` crate) | Implemented from scratch |
| Storage | RocksDB | In-memory + binary persistence |
| API | Node.js bindings (NAPI-RS) | HTTP REST API |
| Tests | ~40 | 131 |
| Binary | Requires Rust toolchain + RocksDB | Single binary, zero C deps |

## What We Reimplemented

### Core Modules (6 packages)

| Module | Original | Our Implementation | Improvement |
|--------|----------|-------------------|-------------|
| **HNSW Index** | Wrapped `hnsw_rs` | `pkg/hnsw/` (from scratch) | Full control, proper deletion with graph cleanup |
| **Distance** | 4 metrics | `pkg/distance/` | Same 4 metrics, float64 precision |
| **Quantization** | SQ8 + PQ | `pkg/quantize/` | SQ8 (4x) + PQ (up to 32x compression) |
| **Storage** | RocksDB | `pkg/store/` | Atomic binary persistence, OOM protection |
| **Collections** | In-memory | `pkg/collection/` | Named collections with metadata persistence |
| **API** | NAPI-RS (Node.js) | `pkg/server/` | HTTP REST with body limits and timeouts |

## Key Improvements

### 1. HNSW From Scratch (vs Wrapped)
Original delegates to `hnsw_rs` crate. Our implementation builds the multi-layer graph, beam search, and neighbor pruning from scratch — giving full control over the algorithm and enabling proper node deletion with bidirectional edge cleanup.

### 2. Zero Dependencies
Original requires 87+ Rust crates including RocksDB (C++ dependency). Our implementation uses only Go stdlib. Single binary, no C compiler needed.

### 3. HTTP REST API (vs NAPI-RS)
Original only offers Node.js bindings. Our implementation provides a standard HTTP REST API accessible from any language.

### 4. Security Hardening
- Request body size limits (MaxBytesReader)
- OOM protection on file loading (max dim/count limits)
- Atomic persistence (temp file + rename)
- Path traversal prevention
- Server timeouts (read/write/idle)

### 5. Proper Node Deletion
Original marks vectors as deleted but doesn't clean graph edges. Our implementation removes all bidirectional edges, maintaining graph quality.

### 6. 3x More Tests
131 tests vs ~40, including recall benchmarks against brute force, concurrent access with race detector, and malicious input protection.

## Limitations

- **No DiskANN/Vamana**: Original has a disk-based ANN implementation we skipped
- **No GNN layer**: Original claims GNN attention (though implementation is minimal)
- **In-memory only**: Dataset limited to available RAM
- **No SIMD**: Original uses NEON/AVX2 for quantized distance; we use scalar math

## Conclusion

ruvector-go reimplements the core vector database with genuine improvements: HNSW from scratch (vs wrapped crate), zero dependencies (vs 87+ crates), HTTP API (vs Node.js-only bindings), proper deletion, security hardening, and 3x test coverage. The single-binary zero-dep deployment is a major practical advantage.
