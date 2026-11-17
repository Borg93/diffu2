# Lance: vectors, blobs, indexes, distributed write, observability

Source: Lance docs at 15.0.0-beta, `docs/src/guide/`, each page read in full. Installed: pylance 13.0.0.
Only what the pages state. Items that may be newer than 13.0.0 are collected in the last section; check them
against the installed package before relying on them.

Contents:
1. `guide/vector_search.md`
2. `guide/blob.md`
3. `guide/indexing.md`
4. `guide/scalar_indexing.md`
5. `guide/distributed_indexing.md`
6. `guide/distributed_write.md`
7. `guide/observability.md`
8. What the pages do not say (for the vectors-in-scan-table decision)
9. Unclear or possibly newer than 13.0.0

---

## 1. `docs/src/guide/vector_search.md`

### Column type
- Vector columns "should use an Arrow fixed-size list or fixed-shape tensor type with a consistent dimension".
- The page names no element type (float16/float32/float64 are not mentioned). The only typed example is Rust `Float32Array`.

### Exact search (no index)
```python
result = dataset.to_table(
    columns=["id", "category", "_distance"],
    nearest={"column": "vector", "q": query_vector, "k": 10},
)
```
- With an explicit `columns` list, include `"_distance"` to get the distance.
- Without a usable vector index, Lance computes distances against the vectors in the search scope: exact, a correctness baseline.
- Force exact search when an index exists: `nearest["use_index"] = False`.

### Metrics
- `l2` (also `euclidean`): squared Euclidean distance; smaller is closer.
- `cosine`: cosine distance; smaller is closer.
- `dot`: dot-product distance; smaller is closer.
- Use the same metric for index creation and querying. Dot may need normalised vectors when a cosine-like ranking is meant.

### Index types (all `IVF_*` partition with an IVF coarse quantizer first)
| Type | Stores | Strength | Tradeoff |
|---|---|---|---|
| `IVF_FLAT` | original vectors | exact distances within searched partitions; no quantizer training | larger index, more candidate I/O |
| `IVF_PQ` | product-quantized | small index, low candidate I/O | lower recall without enough probes/refinement; requires `num_sub_vectors` |
| `IVF_SQ` | scalar-quantized | simple compression, more information per dimension than typical PQ | more storage than PQ |
| `IVF_RQ` | RaBitQ-quantized | compact, multi-bit distance estimates | tune `num_bits`; more bits = better estimates, more build work and storage |
| `IVF_HNSW_FLAT` | original vectors + HNSW graph | graph navigation within partitions; exact candidate distances | larger graph and storage |
| `IVF_HNSW_SQ` | scalar-quantized + HNSW graph | graph search with compressed storage | more build parameters, graph overhead |
| `IVF_HNSW_PQ` | product-quantized + HNSW graph | graph search with small storage | quantization and graph tuning both affect recall |

- The page's recommendation: "Start with `IVF_RQ` for most vector-search workloads." `IVF_PQ` when the smaller footprint or GPU-accelerated training matters; `IVF_HNSW_*` when graph search fits the latency and recall needs.
- Validate by measured recall and latency on representative data, not by index size.
- Index-size formulas and training memory estimates are in `performance.md#vector-index` (not read here).

### Create
```python
dataset.create_index("vector", index_type="IVF_RQ", metric="l2", num_partitions=256)

dataset.create_index("vector", index_type="IVF_PQ", metric="l2",
                     num_partitions=256, num_sub_vectors=16)

dataset.create_index("vector", index_type="IVF_HNSW_SQ", metric="l2",
                     num_partitions=256, m=20, ef_construction=100)
```
- `num_partitions`: number of IVF partitions. More partitions = less work per probe, more probes for the same recall.
- `num_sub_vectors`: PQ sub-vectors; required for PQ variants.
- `num_bits`: RQ defaults to 5; PQ supports 4 or 8. `num_bits=1` for RQ minimises index size and build I/O, but searches cannot use the multi-bit estimate: measure recall.
- `ivf_centroids`, `pq_codebook`: optional pre-trained artifacts.
- `m`, `max_level`, `ef_construction`: HNSW graph parameters.
- `accelerator="cuda"` or `"mps"`: optional GPU acceleration for IVF_PQ training (only IVF_PQ is named).
- Default index name: `<column>_idx`.
- Vector index creation is `replace=False` by default; pass `replace=True` to replace an index of the same name.

### Search parameters
```python
dataset.to_table(nearest={"column": "vector", "q": q, "k": 10, "search_effort": 0.25})
dataset.to_table(nearest={"column": "vector", "q": q, "k": 10, "nprobes": 20, "refine_factor": 5})
```
- `search_effort` in `[0, 1]`, continuous, interpolated between anchors:
  - `0.1`: 80 % mean routing recall target
  - `0.25`: 90 %
  - `0.5` (default): 95 %
  - `1.0`: search all available partitions (caller maximum still applies)
- Calibrated for L2, cosine, dot with `k <= 100000`. Hamming and larger `k` keep the old interpolation with no calibrated target.
- Targets are average coverage of the true neighbours' partitions on unfiltered calibration data: not a guarantee per query, dataset or final Recall@k. Quantization, refinement, filtering and HNSW change final recall.
- Explicit `search_effort` overrides `nprobes`; explicit `minimum_nprobes` and `maximum_nprobes` still apply.
- Filters can trigger more probing to obtain `k` rows, so the initial budget is not a limit on total work.
- `nprobes`: IVF partitions searched. `refine_factor`: fetch up to `k * refine_factor` candidates, re-rank with more accurate distances, return top `k`. `ef`: HNSW search breadth.

### Recall check
```python
exact  = dataset.to_table(nearest={"column": "vector", "q": q, "k": 10, "use_index": False})
approx = dataset.to_table(nearest={"column": "vector", "q": q, "k": 10, "nprobes": 20})
recall = len(set(exact["id"].to_pylist()) & set(approx["id"].to_pylist())) / 10
```
- Average over a representative set of queries; compare latency under the same data, filter, cache and storage.

### Filtered nearest-neighbour search
```python
result = dataset.scanner(
    nearest={"column": "vector", "q": query_vector, "k": 10},
    filter="category = 'science'",
    prefilter=True,
).to_table()
```
- `prefilter=False` is the DEFAULT: top-k first, then the SQL filter. Rejected rows are not replaced, so fewer than `k` rows may come back.
- `prefilter=True`: restrict candidates to matching rows, then top-k. "Usually the desired behavior for metadata-constrained retrieval"; can be more expensive; does not make an approximate index exact.
- A scalar index on the filter column can accelerate prefiltering.
- Deletions and explicit fragment scopes restrict candidates regardless of `prefilter`.
- The filter is a SQL filter on columns of the same dataset as the vector column (the page shows no filter across tables).

### Full-text + vector
```python
from lance.query import MatchQuery
dataset.scanner(
    nearest={"column": "vector", "q": q, "k": 10},
    filter={"search_filter": MatchQuery("machine learning", "text"),
            "expr_filter": "category = 'science'"},
).to_table()
```
- Needs an inverted index on the text column and a vector index on the embedding column.

### Maintenance
- Appending rows does NOT add them to existing vector indexes. By default Lance combines index results with exact search over unindexed data: new rows stay visible, unindexed search cost grows.
- Add new data: `dataset.optimize.optimize_indices(index_names=["vector_idx"])`. Assigns new vectors to existing IVF partitions and quantizer models, no retraining.
- Retrain when the distribution changed: `dataset.optimize.optimize_indices(index_names=["vector_idx"], retrain=True)`. More expensive.
- `fast_search=True`: only when skipping recently appended unindexed data is acceptable. Check first:
```python
stats = dataset.stats.index_stats("vector_idx")
print(stats["num_unindexed_rows"], stats["num_indexed_rows"])
```

---

## 2. `docs/src/guide/blob.md`

### Blob column or inline binary (the page's own rule)
- Plain Arrow `binary` / `large_binary` stores bytes inline, interleaved with the other columns: "simplest and fastest for really small blobs (e.g., thumbnail images)".
- Use a blob column when either holds:
  - partial or streaming reads are needed (inline binary is always read in full; no byte-range fetch);
  - values are large, "roughly 1 MB or more on average": compaction and some updates rewrite whole rows and copy inline payloads forward even when unchanged (write amplification). Blob columns keep payloads in separate `.blob` files that are referenced, not re-copied.
- Rule of thumb: average payload below "a few tens of KB" and only whole-value reads: plain inline binary is fine. Above ~1 MB or file-like access: blob column.
- Blob v2 default placement: up to 64 KiB inline in the data file; over 64 KiB and up to 4 MiB packed into shared `.blob` sidecars; over 4 MiB a dedicated `.blob` file. Configurable per column.

### Versions
- `data_storage_version` = the Lance file format version of a dataset. A dataset has a default; append and compaction can select another exact `2.x` version without rewriting files or changing the default. `0.1` cannot be mixed with `2.x`.
- Blob v2 (`lance.blob.v2`) needs file format `>= 2.2`. `0.1`, `2.0`, `2.1`: legacy blob metadata (`lance-encoding:blob`) only.
- At `>= 2.2` dataset writers accept legacy blob-marked binary input and write it as Blob v2. Older files keep their encoding.
- Appending with `data_storage_version="2.2"` to a `2.0`/`2.1` dataset starts Blob v2 with no upgrade step. Once the first v2 file is committed, descriptor scans return the v2 struct `kind, position, size, blob_id, blob_uri` for all fragments; code reading descriptors by position or comparing exact schemas must adapt.

### Write
```python
import lance, pyarrow as pa
from lance import blob_array, blob_field

schema = pa.schema([pa.field("id", pa.int64()), blob_field("blob")])
table = pa.table({"id": [1], "blob": blob_array([b"hello blob v2"])}, schema=schema)
ds = lance.write_dataset(table, "./blobs_v22.lance", data_storage_version="2.2")
```
- `blob_field("blob", nullable=True)`; `blob_array([...])` accepts in one column: inline `bytes`, a URI string, `Blob.from_uri(uri, position=..., size=...)`, `None`.
- Field tag: `ARROW:extension:name = "lance.blob.v2"`. Logical shapes: Minimal (`data: LargeBinary?`, `uri: Utf8?`) or Complete (plus `position: UInt64?`, `size: UInt64?`). Python `blob_field` and `BlobType` use the Complete shape.
- Every non-null row sets exactly one of `data` and `uri`. `position` and `size` both set or both null; a range needs `uri`; explicit range needs `size > 0`. Empty blob = inline `b""`.
- Thresholds per column: `blob_field(..., inline_size_threshold=..., dedicated_size_threshold=...)`. Dedicated threshold is checked first. Stored in the dataset schema; an append that gives different thresholds for the same column is rejected.
- `blob_pack_file_size_threshold`: write option for rolling packed `.blob` sidecars; does not control inline-vs-packed.
- External URIs must map to a registered non-dataset-root base path by default; otherwise `allow_external_blob_outside_bases=True` on write (embedded mode).
- Blob v2 fields can nest inside structs and variable-length lists.

### Read
| API | Returns | When |
|---|---|---|
| `ds.read_blobs(col, indices=/ids=/addresses=)` | `List[Tuple[int, Optional[bytes]]]` (row address, payload) | whole payloads in memory: loaders, batch work |
| `ds.read_blob_ranges(col, requests=[(row, offset, length)], selector="indices")` | `List[Tuple[int, int, Optional[bytes]]]` (request_index, row address, data) | byte ranges from many rows |
| `ds.take_blobs(col, indices=/ids=/addresses=)` | `List[Optional[BlobFile]]` | file-like: streaming, seeking, partial reads |
| `lance.read_blob_files(handles)` / `lance.read_blob_file_ranges([(handle, offset, length)])` | `(request_index, data)` pairs | repeated reads of handles already held (skips the descriptor lookup) |
| `ds.scanner(columns=["blob"], blob_handling="all_binary").to_table()` | Arrow binary column | blob column in a scan result |

- Exactly one selector for `read_blobs` / `take_blobs`: `ids`, `indices` or `addresses`. Error otherwise: "Exactly one of ids, indices, or addresses must be specified".
- `indices`: positional, stable within one snapshot. `ids`: logical row ids, stable when row ids are available. `addresses`: physical, unstable.
- Row ids / addresses: `ds.to_table(columns=[], with_row_id=True).column("_rowid")`, `ds.to_table(columns=[], with_row_address=True).column("_rowaddr")`.
- Do NOT wrap `take_blobs` in a thread pool to call `read()` on each; use `read_blobs`, which plans batched reads through Lance's scheduler.
- One result per selected row; null blob = `None`; empty blob = non-null empty payload.
- `BlobFile` sequential reads are buffered, 512 KiB by default; `buffer_size=0` for unbuffered. `read_range` / `read_ranges` do not use the buffer.
- `read_blob_files`: `io_parallelism` and `io_buffer_size` keyword arguments (positive). Handles keep the locations resolved when opened; later commits do not re-resolve them.

### Legacy (file format <= 2.1)
```python
pa.field("video", pa.large_binary(), metadata={"lance-encoding:blob": "true"})
```
- Rewrite to v2: scan with `blob_handling="all_binary"`, rebuild with `blob_field` / `blob_array`, write with `data_storage_version="2.2"`. The page's example loads all payloads in memory; batch it for large datasets. `0.1` tables need a rewrite to enter `2.x`.

### Managed blobs (advanced)
- Schema metadata `lance.blob.managed` = `"true"` stores out-of-line payloads as `_blobs/<uuid>.blob`. Works with file formats 2.2 and 2.3.
- Committing the opt-in sets a reader/writer capability flag at once and for good: clients that do not understand it must refuse to open the snapshot. All maintenance clients must support Managed Blobs before activation; older clients (the page names v11.0.0) running cleanup through old handles can delete sidecars still referenced.
- Explicit sharing: `LanceFileSession(uri).open_managed_blob_writer()`, `writer.write_blob(bytes)`, `writer.finish()` returns descriptors; `lance.BlobDescriptorArrayBuilder("blob")` with `.append(descriptor)`, `.finish()`, `.field`; `append_managed(uri, offset, size, base_id=None)`; `writer.finish_column(field_name)`. No content comparison, no automatic deduplication, same table only.
- Do not run cleanup with `delete_unverified=True` while preparing uploads.

### Troubleshooting
- "Blob v2 requires file version >= 2.2": write to a dataset created with `data_storage_version="2.2"` or newer.
- "Legacy blob columns ... are not supported for file version >= 2.2": a lower-level file writer got legacy metadata; use the dataset writer.

---

## 3. `docs/src/guide/indexing.md`

- Indexes do not change logical contents; stored with the dataset; part of its transactional versioning.
- Create: scalar and full-text with `LanceDataset.create_scalar_index`, vector with `LanceDataset.create_index`:
```python
dataset.create_scalar_index("id", "BTREE")
dataset.create_scalar_index("text", "INVERTED", with_position=True)
dataset.create_index("vector", index_type="IVF_RQ", num_partitions=256)
```
- Default name `<column>_idx`; `name=...` for another, e.g. several indexes on one column.
- Inspect:
```python
for index in dataset.describe_indices():
    print(index.name, index.index_type, index.num_rows_indexed)
stats = dataset.stats.index_stats("text_idx")
print(stats["num_indexed_rows"], stats["num_unindexed_rows"])
print(dataset.scanner(filter="id >= 100").explain_plan())   # look for ScalarIndexQuery or MaterializeIndex
```
- Appends do not update indexes. Searches combine indexed results with unindexed scans by default; latency grows with unindexed data.
```python
dataset.insert(new_rows)
dataset.optimize.optimize_indices(index_names=["text_idx", "vector_idx"])   # omit index_names for all
dataset.optimize.optimize_indices(index_names=["vector_idx"], retrain=True) # rebuilds; expensive
```
- `fast_search=True`: search only indexed coverage; may skip recently appended rows.
- Updates stored in overlay files: stale index entries are excluded in every mode. Changed rows are evaluated with current values for scalar filtering in both modes; for vector search, full-text search and MinHash they are evaluated by default and SKIPPED with `fast_search=True`.
- Physical rewrites can invalidate row addresses used by indexes. Run compaction BEFORE rebuilding or optimizing indexes, and let Lance remap. Stable row IDs keep logical identity across updates and compaction but do not remove index maintenance.
- Drop: `dataset.drop_index("id_idx")`.
- Change index type, metric, tokenizer or training parameters: create again with `replace=True` and the same name. Scalar: `replace=True` is the default. Vector: pass it explicitly.

---

## 4. `docs/src/guide/scalar_indexing.md`

- Lance uses the index to produce candidate rows, then evaluates the full filter: index choice never changes results.

| Type | For | Accelerates | Limitation |
|---|---|---|---|
| `BTREE` | high-cardinality scalars (IDs, timestamps) | equality, ranges, `IN`, null predicates | larger than skipping structures |
| `BITMAP` | low-cardinality, many rows per value (categories) | equality, small ranges, set membership | many unique values grow index size and build memory |
| `LABEL_LIST` | list columns with low-cardinality elements (tags) | `array_has`, `array_contains`, `array_has_all`, `array_has_any` | not for high-cardinality arrays |
| `ZONEMAP` | approximately sorted columns | range predicates by skipping zones | little benefit on randomly distributed values |
| `NGRAM` | substring search | `contains(column, 'substring')` | candidates may need rechecking |
| `BLOOMFILTER` | membership-style equality | equality and inequality filters | inexact, limited predicates |
| `FM` | exact substring search, including one-character patterns | `contains(column, 'substring')` | specialised |
| `LISTELEMENT` (experimental) | fields of elements of a `list<struct>` column | `any_element(column, 'predicate')` | predicate must fix the first key |

- "B-tree is the usual default for scalar columns." Bitmap for genuinely low-cardinality columns; zone maps when data is sorted or nearly sorted.
```python
dataset.create_scalar_index("id", "BTREE")
dataset.create_scalar_index("category", "BITMAP")
dataset.create_scalar_index("created_at", "ZONEMAP")
dataset.create_scalar_index("tags", "LABEL_LIST")
dataset.create_scalar_index("text", "NGRAM", name="text_ngram_idx")
dataset.create_scalar_index("text", "INVERTED", name="text_fts_idx", with_position=True)
```
- Python `create_scalar_index` replaces an index of the same name by default (`replace=True`).
- No special query API; indexes are chosen from the SQL filter:
```python
dataset.to_table(columns=["id", "category"], filter="category = 'tech' AND id >= 100", limit=10)
dataset.to_table(filter="array_has_any(tags, ['python', 'rust'])")
```
- Indexed predicates combine with `AND` / `OR`. `id = 1 OR unindexed_column = 2` may not use the index on `id`. `(id = 1 AND unindexed_column = 2) OR id = 3` still uses it (every branch has an indexed predicate).
- Disable for one query: `use_scalar_index=False` on the scanner.
- Verify: `dataset.scanner(filter="id >= 100").explain_plan()` shows `ScalarIndexQuery` or `MaterializeIndex`. Coverage: `describe_indices()` (`.name`, `.index_type`, `.field_names`, `.num_rows_indexed`); `dataset.stats.index_stats("id_idx")`.
- Appends are not indexed automatically; new rows are found by scanning unindexed fragments. Run `dataset.optimize.optimize_indices(index_names=["id_idx", "category_idx"])` periodically.
- Nulls: scalar indexes keep SQL three-valued logic. A filter that evaluates to null is not true. Test on data with null values, null list elements, empty lists and all-null columns. Exclude nulls explicitly: `filter="category IS NOT NULL AND category = 'tech'"`.

---

## 5. `docs/src/guide/distributed_indexing.md`

- Lance gives APIs for an external distributed build; no scheduler or orchestration. The caller launches workers and drives the flow.
- Terms: Segment = one worker output under `indices/<segment_uuid>/` (`index.idx`, `auxiliary.idx`); Physical segment = ready to commit; Logical index = the named index, one or more physical segments.
- Flow:
  1. Each worker: Python `create_index_uncommitted(..., fragment_ids=...)` (Rust `create_index_builder(...).fragments(...).execute_uncommitted()`); writes one segment.
  2. Caller either commits worker outputs as they are with `commit_existing_index_segments(...)`, or groups them and calls `merge_existing_index_segments(...)` per group, then commits the final list with `commit_existing_index_segments(...)`.
- Within one commit, segments must have disjoint fragment coverage.
- `merge_existing_index_segments(...)` supports vector, inverted, bitmap, BTree and zone map segments. Other scalar families can commit several compatible segments but not merge them.
- Vector model scope:
  - Shared: train IVF centroids once, pass the same artifacts to every worker; for IVF-PQ segments meant to be merged, the same PQ codebook too.
  - Independent: each worker trains on its own `fragment_ids`; segments can be committed together as one logical index without shared centroids.
  - Query fans out: each segment ranks partitions with its own centroids, results merged by `_distance`. Independently trained segments return valid results.
  - Physical merge needs shared compatible model artifacts; otherwise keep segments separate.
- Shuffle memory: `LANCE_SHUFFLE_MAX_PRELOADED_OFFSETS_BYTES`, default `268435456` (256 MiB), per shuffle (adds up across concurrent builds); `0` disables offset preloading; invalid values error. Does not bound total process memory.
- Segments never committed stay as unreferenced index directories; `cleanup_old_versions(...)` removes them by the same age rules as other unreferenced index files.

---

## 6. `docs/src/guide/distributed_write.md`

- Ray and Spark integrations exist (`lance-ray`, `lance-spark`); this page is for custom setups (slurm, Kubernetes).
- Two phases: workers write `lance.LanceFragment`s in parallel; one process commits all `lance.FragmentMetadata` in one `lance.LanceOperation`.

### New data
```python
from lance.fragment import write_fragments
fragments = write_fragments(data, data_uri, schema=schema)          # on each worker
payload = [json.dumps(f.to_json()) for f in fragments]               # ship to committer

from lance import FragmentMetadata, LanceOperation
all_fragments = [FragmentMetadata.from_json(f) for f in payloads]
op = lance.LanceOperation.Overwrite(schema, all_fragments)           # create or overwrite
lance.LanceDataset.commit(data_uri, op, read_version=0)              # 0: dataset was empty
```
### Append
```python
ds = lance.dataset(data_uri)
op = lance.LanceOperation.Append(all_fragments)
lance.LanceDataset.commit(data_uri, op, read_version=ds.version)
```
### Add columns (no copy of existing data files; new data files linked by metadata)
```python
def name_len(names: RecordBatch) -> RecordBatch:
    return RecordBatch.from_arrays([pc.utf8_length(names["name"])], ["name_len"])

new_fragment1, new_schema = dataset.get_fragments()[0].merge_columns(name_len, ["name"])   # worker 1
new_fragment2, _          = dataset.get_fragments()[1].merge_columns(name_len, ["name"])   # worker 2
op = lance.LanceOperation.Merge([new_fragment1, new_fragment2], schema=new_schema)
lance.LanceDataset.commit(uri, op, read_version=dataset.version)
```
- Every fragment must be covered in the page's example (both fragments get the new column before the `Merge` commit).

### Update columns per fragment
```python
updated_fragment0, fields_modified0 = dataset.get_fragment(0).update_columns(
    update_data1, left_on="id", right_on="id")
op = lance.LanceOperation.Update(
    updated_fragments=[updated_fragment0, updated_fragment1],
    fields_modified=list(set(fields_modified0 + fields_modified1)))
lance.LanceDataset.commit(str(dataset_uri), op, read_version=dataset.version)
```
- Left-outer join of the fragment with the new data on `left_on` / `right_on`. No match: old value kept. One match: new value. Several matches: the value of a RANDOM matching row (so the key in the update data must be unique).
- In-memory hash join or spillable sort-merge join; see `update_join.md` (not read here).

### Stable row ids when assembling a transaction by hand
- Dataset created with `enable_stable_row_ids=True`: a row keeps its `_rowid` for life. When you rewrite rows yourself you must carry the ids: read with `with_row_id=True`, then
```python
from lance.fragment import RowIdSequence, write_fragments
fragments[0].row_id_meta = RowIdSequence(rewritten["_rowid"]).to_inline_metadata()
updated_fragment = dataset.get_fragments()[0].delete("id in (2, 3)")      # tombstone old copies
op = lance.LanceOperation.Update(updated_fragments=[updated_fragment], new_fragments=fragments)
dataset = lance.LanceDataset.commit(dataset_uri, op, read_version=dataset.version)
```
- PITFALL: a fragment written without `row_id_meta` commits without error and silently gives every rewritten row a new `_rowid`.
- Mixed fragment: rewritten rows FIRST, new rows last; pass only the rewritten rows' ids. Never invent ids for new rows (the commit assigns them after conflict resolution). More ids than rows is rejected.
- Leave `created_at_version_meta` and `last_updated_at_version_meta` as `None`; Lance derives them. Lineage columns: `_row_created_at_version`, `_row_last_updated_at_version`.

---

## 7. `docs/src/guide/observability.md`

- The metric catalogue is NOT in this page: it is included from `rust/lance/src/metrics.md` (`--8<--` include), which was not among the pages read. One metric name appears as an example: `lance_object_store_request_duration_seconds` (histogram).
- Python: only OpenTelemetry (no pluggable recorder). pylance wheels are built with the `metrics` feature.
```bash
pip install "pylance[otel]"
```
```python
from lance.otel import instrument_lance_metrics
instrument_lance_metrics()            # global MeterProvider; or meter_provider=...
```
- Memory metrics are empty until memory observation is on: set `LANCE_MEMORY_OBSERVE=1` before the process starts, or
```python
from lance.otel import set_memory_observation
set_memory_observation(True)
```
- Histograms are exported as three observable counters: `<name>_bucket` (with an `le` attribute, cumulative, `le="+Inf"` = total), `<name>_count`, `<name>_sum`.
- With no recorder installed, emission is a cheap no-op (stated for Rust).

---

## 8. What the pages do not say (for the vectors-in-scan-table decision)

None of the seven pages states any of the following; measure or read other pages (`performance.md`, `read_and_write.md`, `row_ids.md`) before deciding:
- Whether float16 vectors are accepted for exact search or for any index type. No element type is named.
- How null vectors (a vector column filled for only a subset of rows) are stored, what they cost on disk, or how index building and search treat them.
- Scan cost or batch size for wide vector columns; no `batch_size` advice appears in these pages.
- A recommendation for or against a separate vector table. What bears on it indirectly:
  - `prefilter=True` takes a SQL filter on the same dataset as the vector column, and a scalar index on that column speeds it up. A filter on scan features with vectors in another table is not shown.
  - Columns can be added to an existing table without copying existing data files (`merge_columns` + `LanceOperation.Merge`).
  - Compaction and some updates rewrite whole rows and copy large inline payloads forward (stated for binary; the reason given for blob columns).
  - Indexes need `optimize_indices` after appends; compaction comes before index rebuild or optimize.

---

## Unclear or possibly newer than 13.0.0

Check each against pylance 13.0.0 (`help(...)`, the installed source) before use:
- `IVF_RQ` as the recommended default index, and its `num_bits=5` default. May be newer or behave differently in 13.
- `search_effort` in `nearest` and its calibrated targets (0.1 / 0.25 / 0.5 / 1.0, `k <= 100000`). Likely new; fall back to `nprobes`, `minimum_nprobes`, `maximum_nprobes`, `refine_factor`.
- `dataset.describe_indices()` with `.num_rows_indexed`, `.field_names`, "segment UUIDs".
- "Updates stored in overlay files" and the `fast_search` table for changed rows.
- Scalar index types `FM` and `LISTELEMENT` (experimental), and `any_element(...)`.
- Blob: `read_blobs`, `read_blob_ranges`, `lance.read_blob_files`, `lance.read_blob_file_ranges`, `buffer_size` on `BlobFile`, `inline_size_threshold` / `dedicated_size_threshold` on `blob_field`, `blob_pack_file_size_threshold`, mixing legacy and v2 files by append, Blob v2 nested in structs and lists, file format `2.3`.
- Managed Blobs: `lance.blob.managed`, `LanceFileSession.open_managed_blob_writer`, `ManagedBlobWriter`, `BlobDescriptorArrayBuilder`, `append_managed`, `finish_column`, descriptor kinds 4 and 5. The page itself warns that v11.0.0 clients mishandle cleanup.
- Distributed indexing: `create_index_uncommitted(..., fragment_ids=...)`, `merge_existing_index_segments`, `commit_existing_index_segments`, independent segment models, `LANCE_SHUFFLE_MAX_PRELOADED_OFFSETS_BYTES`. The page says `indices/<segment_uuid>/` in most places and `_indices/` once: the directory name is inconsistent in the page.
- Distributed write: `RowIdSequence(...).to_inline_metadata()`, `row_id_meta`, `new_fragments=` on `LanceOperation.Update`, `created_at_version_meta` / `last_updated_at_version_meta`, the lineage columns, and the pointer to `update_join.md` (spillable sort-merge join).
- Observability: `lance.otel`, `instrument_lance_metrics`, `set_memory_observation`, `LANCE_MEMORY_OBSERVE`, the `pylance[otel]` extra.
- Unclear in the page: `write_fragments` output shows `id=0` for both workers' fragments; the page does not explain id assignment at commit.
- Unclear in the page: whether `fast_search` is a scanner argument or a key of `nearest`; it is only written as `fast_search=True`.
- Unclear in the page: `dot` is called "dot-product distance; smaller is closer" with no formula.
