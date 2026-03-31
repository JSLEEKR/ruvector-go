package hnsw

// candidateItem is an element in the priority queue.
type candidateItem struct {
	id   uint64
	dist float32
}

// candidateHeap implements heap.Interface.
// By default it's a min-heap (closest first).
// Set maxHeap=true for a max-heap (farthest first).
type candidateHeap struct {
	maxHeap bool
	items   []candidateItem
}

// Implement sort.Interface indirectly through the slice.
func (h candidateHeap) Len() int { return len(h.items) }

func (h candidateHeap) Less(i, j int) bool {
	if h.maxHeap {
		return h.items[i].dist > h.items[j].dist
	}
	return h.items[i].dist < h.items[j].dist
}

func (h candidateHeap) Swap(i, j int) { h.items[i], h.items[j] = h.items[j], h.items[i] }

func (h *candidateHeap) Push(x interface{}) {
	h.items = append(h.items, x.(candidateItem))
}

func (h *candidateHeap) Pop() interface{} {
	old := h.items
	n := len(old)
	item := old[n-1]
	h.items = old[:n-1]
	return item
}
