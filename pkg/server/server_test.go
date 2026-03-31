package server

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/JSLEEKR/ruvector-go/pkg/collection"
)

func setupServer(t *testing.T) *Server {
	t.Helper()
	m, err := collection.NewManager(t.TempDir())
	if err != nil {
		t.Fatalf("new manager: %v", err)
	}
	return New(m)
}

func doReq(t *testing.T, s *Server, method, path string, body interface{}) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		json.NewEncoder(&buf).Encode(body)
	}
	req := httptest.NewRequest(method, path, &buf)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	return rec
}

func TestHealth(t *testing.T) {
	s := setupServer(t)
	rec := doReq(t, s, "GET", "/health", nil)
	if rec.Code != 200 {
		t.Errorf("status = %d, want 200", rec.Code)
	}
}

func TestHealthMethodNotAllowed(t *testing.T) {
	s := setupServer(t)
	rec := doReq(t, s, "POST", "/health", nil)
	if rec.Code != 405 {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

func TestCreateCollection(t *testing.T) {
	s := setupServer(t)
	body := createCollectionReq{Name: "test", Dimension: 4, Metric: "cosine"}
	rec := doReq(t, s, "POST", "/collections", body)
	if rec.Code != 201 {
		t.Errorf("status = %d, want 201, body: %s", rec.Code, rec.Body.String())
	}
}

func TestCreateCollectionNoName(t *testing.T) {
	s := setupServer(t)
	body := createCollectionReq{Dimension: 4}
	rec := doReq(t, s, "POST", "/collections", body)
	if rec.Code != 400 {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestCreateCollectionBadDim(t *testing.T) {
	s := setupServer(t)
	body := createCollectionReq{Name: "test", Dimension: 0}
	rec := doReq(t, s, "POST", "/collections", body)
	if rec.Code != 400 {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestCreateCollectionBadMetric(t *testing.T) {
	s := setupServer(t)
	body := createCollectionReq{Name: "test", Dimension: 4, Metric: "invalid"}
	rec := doReq(t, s, "POST", "/collections", body)
	if rec.Code != 400 {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestCreateCollectionDuplicate(t *testing.T) {
	s := setupServer(t)
	body := createCollectionReq{Name: "test", Dimension: 4}
	doReq(t, s, "POST", "/collections", body)
	rec := doReq(t, s, "POST", "/collections", body)
	if rec.Code != 409 {
		t.Errorf("status = %d, want 409", rec.Code)
	}
}

func TestListCollections(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "a", Dimension: 4})
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "b", Dimension: 8})
	rec := doReq(t, s, "GET", "/collections", nil)
	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	var list []collectionInfoResp
	json.NewDecoder(rec.Body).Decode(&list)
	if len(list) != 2 {
		t.Errorf("list = %d, want 2", len(list))
	}
}

func TestGetCollectionInfo(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "test", Dimension: 4})
	rec := doReq(t, s, "GET", "/collections/test", nil)
	if rec.Code != 200 {
		t.Errorf("status = %d", rec.Code)
	}
	var info collectionInfoResp
	json.NewDecoder(rec.Body).Decode(&info)
	if info.Name != "test" || info.Dimension != 4 {
		t.Errorf("info = %+v", info)
	}
}

func TestGetCollectionNotFound(t *testing.T) {
	s := setupServer(t)
	rec := doReq(t, s, "GET", "/collections/missing", nil)
	if rec.Code != 404 {
		t.Errorf("status = %d, want 404", rec.Code)
	}
}

func TestDeleteCollection(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "test", Dimension: 4})
	rec := doReq(t, s, "DELETE", "/collections/test", nil)
	if rec.Code != 200 {
		t.Errorf("status = %d", rec.Code)
	}
}

func TestDeleteCollectionNotFound(t *testing.T) {
	s := setupServer(t)
	rec := doReq(t, s, "DELETE", "/collections/missing", nil)
	if rec.Code != 404 {
		t.Errorf("status = %d, want 404", rec.Code)
	}
}

func TestInsertVectors(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "test", Dimension: 3})
	body := insertBatchReq{Vectors: []insertVectorReq{
		{ID: 1, Vector: []float32{1, 0, 0}},
		{ID: 2, Vector: []float32{0, 1, 0}},
	}}
	rec := doReq(t, s, "POST", "/collections/test/vectors", body)
	if rec.Code != 201 {
		t.Errorf("status = %d, body: %s", rec.Code, rec.Body.String())
	}
}

func TestInsertVectorsCollectionNotFound(t *testing.T) {
	s := setupServer(t)
	body := insertBatchReq{Vectors: []insertVectorReq{
		{ID: 1, Vector: []float32{1, 0, 0}},
	}}
	rec := doReq(t, s, "POST", "/collections/missing/vectors", body)
	if rec.Code != 404 {
		t.Errorf("status = %d, want 404", rec.Code)
	}
}

func TestInsertVectorsEmpty(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "test", Dimension: 3})
	body := insertBatchReq{Vectors: nil}
	rec := doReq(t, s, "POST", "/collections/test/vectors", body)
	if rec.Code != 400 {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestGetVector(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "test", Dimension: 3})
	doReq(t, s, "POST", "/collections/test/vectors", insertBatchReq{
		Vectors: []insertVectorReq{{ID: 42, Vector: []float32{1, 2, 3}}},
	})
	rec := doReq(t, s, "GET", "/collections/test/vectors/42", nil)
	if rec.Code != 200 {
		t.Errorf("status = %d", rec.Code)
	}
	var v insertVectorReq
	json.NewDecoder(rec.Body).Decode(&v)
	if v.ID != 42 {
		t.Errorf("id = %d, want 42", v.ID)
	}
}

func TestGetVectorNotFound(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "test", Dimension: 3})
	rec := doReq(t, s, "GET", "/collections/test/vectors/999", nil)
	if rec.Code != 404 {
		t.Errorf("status = %d, want 404", rec.Code)
	}
}

func TestDeleteVector(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "test", Dimension: 3})
	doReq(t, s, "POST", "/collections/test/vectors", insertBatchReq{
		Vectors: []insertVectorReq{{ID: 1, Vector: []float32{1, 2, 3}}},
	})
	rec := doReq(t, s, "DELETE", "/collections/test/vectors/1", nil)
	if rec.Code != 200 {
		t.Errorf("status = %d", rec.Code)
	}
}

func TestSearch(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "test", Dimension: 3, Metric: "euclidean"})
	doReq(t, s, "POST", "/collections/test/vectors", insertBatchReq{
		Vectors: []insertVectorReq{
			{ID: 1, Vector: []float32{0, 0, 0}},
			{ID: 2, Vector: []float32{1, 1, 1}},
			{ID: 3, Vector: []float32{10, 10, 10}},
		},
	})
	rec := doReq(t, s, "POST", "/collections/test/search", searchReq{
		Vector: []float32{0, 0, 0}, K: 2,
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d, body: %s", rec.Code, rec.Body.String())
	}
	var results []searchResultResp
	json.NewDecoder(rec.Body).Decode(&results)
	if len(results) != 2 {
		t.Errorf("results = %d, want 2", len(results))
	}
	if results[0].ID != 1 {
		t.Errorf("nearest = %d, want 1", results[0].ID)
	}
}

func TestSearchCollectionNotFound(t *testing.T) {
	s := setupServer(t)
	rec := doReq(t, s, "POST", "/collections/missing/search", searchReq{
		Vector: []float32{0, 0}, K: 1,
	})
	if rec.Code != 404 {
		t.Errorf("status = %d, want 404", rec.Code)
	}
}

func TestSearchBadK(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "test", Dimension: 3})
	rec := doReq(t, s, "POST", "/collections/test/search", searchReq{
		Vector: []float32{0, 0, 0}, K: 0,
	})
	if rec.Code != 400 {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestSearchMethodNotAllowed(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "test", Dimension: 3})
	rec := doReq(t, s, "GET", "/collections/test/search", nil)
	if rec.Code != 405 {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

func TestCreateWithHNSWParams(t *testing.T) {
	s := setupServer(t)
	body := createCollectionReq{
		Name: "test", Dimension: 64, Metric: "euclidean",
		M: 32, EfConst: 400, EfSearch: 100,
	}
	rec := doReq(t, s, "POST", "/collections", body)
	if rec.Code != 201 {
		t.Errorf("status = %d", rec.Code)
	}
}

func TestInvalidVectorID(t *testing.T) {
	s := setupServer(t)
	doReq(t, s, "POST", "/collections", createCollectionReq{Name: "test", Dimension: 3})
	rec := doReq(t, s, "GET", "/collections/test/vectors/abc", nil)
	if rec.Code != 400 {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}
