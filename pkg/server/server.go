// Package server provides an HTTP REST API for vector operations.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/JSLEEKR/ruvector-go/pkg/collection"
	"github.com/JSLEEKR/ruvector-go/pkg/distance"
	"github.com/JSLEEKR/ruvector-go/pkg/hnsw"
	"github.com/JSLEEKR/ruvector-go/pkg/store"
)

// Server is the HTTP API server.
type Server struct {
	manager *collection.Manager
	mux     *http.ServeMux
	srv     *http.Server
}

// New creates a new server backed by the given collection manager.
func New(manager *collection.Manager) *Server {
	s := &Server{manager: manager}
	s.mux = http.NewServeMux()
	s.registerRoutes()
	return s
}

// Start begins serving on the given address.
func (s *Server) Start(addr string) error {
	s.srv = &http.Server{
		Addr:           addr,
		Handler:        s.mux,
		ReadTimeout:    30 * time.Second,
		WriteTimeout:   30 * time.Second,
		IdleTimeout:    120 * time.Second,
		MaxHeaderBytes: 1 << 20, // 1MB max header size
	}
	log.Printf("ruvector-go server starting on %s", addr)
	return s.srv.ListenAndServe()
}

// Shutdown gracefully shuts down the server.
// Safe to call even if Start has not been called yet.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.srv == nil {
		return nil
	}
	return s.srv.Shutdown(ctx)
}

// Handler returns the http.Handler for testing.
func (s *Server) Handler() http.Handler {
	return s.mux
}

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/collections", s.handleCollections)
	s.mux.HandleFunc("/collections/", s.handleCollectionRoutes)
}

// --- Request/Response types ---

type createCollectionReq struct {
	Name      string `json:"name"`
	Dimension int    `json:"dimension"`
	Metric    string `json:"metric"`
	M         int    `json:"m,omitempty"`
	EfConst   int    `json:"ef_construction,omitempty"`
	EfSearch  int    `json:"ef_search,omitempty"`
}

type insertVectorReq struct {
	ID     uint64    `json:"id"`
	Vector []float32 `json:"vector"`
}

type insertBatchReq struct {
	Vectors []insertVectorReq `json:"vectors"`
}

type searchReq struct {
	Vector []float32 `json:"vector"`
	K      int       `json:"k"`
	Ef     int       `json:"ef,omitempty"`
}

type searchResultResp struct {
	ID       uint64  `json:"id"`
	Distance float32 `json:"distance"`
}

type errorResp struct {
	Error string `json:"error"`
}

type statusResp struct {
	Status string `json:"status"`
}

type collectionInfoResp struct {
	Name      string `json:"name"`
	Dimension int    `json:"dimension"`
	Size      int    `json:"size"`
	MaxLevel  int    `json:"max_level"`
}

// --- Handlers ---

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, statusResp{Status: "ok"})
}

func (s *Server) handleCollections(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		names := s.manager.List()
		infos := make([]collectionInfoResp, 0, len(names))
		for _, name := range names {
			c, err := s.manager.Get(name)
			if err != nil {
				continue
			}
			stats := c.Stats()
			infos = append(infos, collectionInfoResp{
				Name:      name,
				Dimension: stats.Dim,
				Size:      stats.Size,
				MaxLevel:  stats.MaxLevel,
			})
		}
		writeJSON(w, http.StatusOK, infos)

	case http.MethodPost:
		r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit
		var req createCollectionReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		if req.Name == "" {
			writeError(w, http.StatusBadRequest, "name is required")
			return
		}
		if req.Dimension <= 0 {
			writeError(w, http.StatusBadRequest, "dimension must be positive")
			return
		}

		metric := distance.Cosine
		if req.Metric != "" {
			var err error
			metric, err = distance.ParseMetric(req.Metric)
			if err != nil {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
		}

		cfg := collection.Config{
			Name:      req.Name,
			Dimension: req.Dimension,
			Metric:    metric,
			HNSW:      hnsw.DefaultConfig(),
		}
		if req.M > 0 {
			cfg.HNSW.M = req.M
			cfg.HNSW.Mmax = 2 * req.M
		}
		if req.EfConst > 0 {
			cfg.HNSW.EfConstruction = req.EfConst
		}
		if req.EfSearch > 0 {
			cfg.HNSW.EfSearch = req.EfSearch
		}

		_, err := s.manager.Create(cfg)
		if err != nil {
			writeError(w, http.StatusConflict, err.Error())
			return
		}
		writeJSON(w, http.StatusCreated, statusResp{Status: "created"})

	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (s *Server) handleCollectionRoutes(w http.ResponseWriter, r *http.Request) {
	// Parse: /collections/{name}/...
	path := strings.TrimPrefix(r.URL.Path, "/collections/")
	parts := strings.SplitN(path, "/", 2)
	if len(parts) == 0 || parts[0] == "" {
		writeError(w, http.StatusBadRequest, "collection name required")
		return
	}
	name := parts[0]
	sub := ""
	if len(parts) > 1 {
		sub = parts[1]
	}

	switch sub {
	case "":
		s.handleCollectionInfo(w, r, name)
	case "vectors":
		s.handleVectors(w, r, name)
	case "search":
		s.handleSearch(w, r, name)
	default:
		// Check for vectors/{id}
		if strings.HasPrefix(sub, "vectors/") {
			idStr := strings.TrimPrefix(sub, "vectors/")
			s.handleVectorByID(w, r, name, idStr)
		} else {
			writeError(w, http.StatusNotFound, "not found")
		}
	}
}

func (s *Server) handleCollectionInfo(w http.ResponseWriter, r *http.Request, name string) {
	switch r.Method {
	case http.MethodGet:
		c, err := s.manager.Get(name)
		if err != nil {
			writeError(w, http.StatusNotFound, err.Error())
			return
		}
		stats := c.Stats()
		writeJSON(w, http.StatusOK, collectionInfoResp{
			Name:      name,
			Dimension: stats.Dim,
			Size:      stats.Size,
			MaxLevel:  stats.MaxLevel,
		})

	case http.MethodDelete:
		if err := s.manager.Delete(name); err != nil {
			writeError(w, http.StatusNotFound, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, statusResp{Status: "deleted"})

	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (s *Server) handleVectors(w http.ResponseWriter, r *http.Request, name string) {
	c, err := s.manager.Get(name)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	switch r.Method {
	case http.MethodPost:
		r.Body = http.MaxBytesReader(w, r.Body, 100<<20) // 100MB limit for batch vectors
		var batch insertBatchReq
		if err := json.NewDecoder(r.Body).Decode(&batch); err != nil {
			writeError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		if len(batch.Vectors) == 0 {
			writeError(w, http.StatusBadRequest, "no vectors provided")
			return
		}

		entries := make([]store.VectorEntry, len(batch.Vectors))
		for i, v := range batch.Vectors {
			entries[i] = store.VectorEntry{ID: v.ID, Vector: v.Vector}
		}
		if err := c.InsertBatch(entries); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		writeJSON(w, http.StatusCreated, statusResp{
			Status: fmt.Sprintf("inserted %d vectors", len(batch.Vectors)),
		})

	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (s *Server) handleVectorByID(w http.ResponseWriter, r *http.Request, name string, idStr string) {
	c, err := s.manager.Get(name)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	id, err := strconv.ParseUint(idStr, 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid vector ID")
		return
	}

	switch r.Method {
	case http.MethodGet:
		v, ok := c.Get(id)
		if !ok {
			writeError(w, http.StatusNotFound, "vector not found")
			return
		}
		writeJSON(w, http.StatusOK, insertVectorReq{ID: id, Vector: v})

	case http.MethodDelete:
		if !c.Delete(id) {
			writeError(w, http.StatusNotFound, "vector not found")
			return
		}
		writeJSON(w, http.StatusOK, statusResp{Status: "deleted"})

	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request, name string) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	c, err := s.manager.Get(name)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit for search
	var req searchReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.K <= 0 {
		writeError(w, http.StatusBadRequest, "k must be positive")
		return
	}

	results, err := c.SearchEf(req.Vector, req.K, req.Ef)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	resp := make([]searchResultResp, len(results))
	for i, r := range results {
		resp[i] = searchResultResp{ID: r.ID, Distance: r.Distance}
	}
	writeJSON(w, http.StatusOK, resp)
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, errorResp{Error: msg})
}
