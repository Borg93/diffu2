# Lance: vector search, indexes, blobs, multi-worker writes

Covers nearest-neighbour search, vector and scalar indexes and their upkeep, blob columns, commits assembled from several workers, and metrics.
Read it before building or tuning an index, running a filtered vector search, storing values near 1 MB, or committing fragments by hand.

`[13?]` marks what may be absent or different in pylance 13.0.0: confirm it with `help(...)` or the installed source first.

## Search nearest neighbours

- Exact search: `ds.to_table(columns=[..., "_distance"], nearest={"column": "vector", "q": q, "k": 10})`. An explicit `columns` list returns the distance only when it names `"_distance"`. With an index present, `nearest["use_index"] = False` forces exact search: the correctness baseline. (guide/vector_search.md)
- The docs ask for a fixed-size list or fixed-shape tensor of one dimension and name no element type: whether float16 is accepted by exact search or by any index type is unstated, so test it on a throwaway table. Null vectors and their treatment by index build and search are also unstated. (guide/vector_search.md)
- Metrics: `l2` (= `euclidean`, squared distance), `cosine`, `dot`; smaller is closer in all three (the page gives no formula for `dot`). Query with the metric the index was built with; normalise vectors when `dot` is meant to rank like cosine.
- Filtered search: `ds.scanner(nearest={...}, filter="category = 'science'", prefilter=True).to_table()`. `prefilter=False` is the default: top-k first, then the filter, rejected rows are not replaced, so fewer than `k` rows come back. `prefilter=True` restricts candidates first; it costs more, leaves an approximate index approximate, and runs faster with a scalar index on the filter column. A filter can trigger extra probing to reach `k` rows, so the probe budget does not cap the work. Deletions and explicit fragment scopes restrict candidates in either mode.
- Text plus vector: `filter={"search_filter": lance.query.MatchQuery("words", "text"), "expr_filter": "category = 'science'"}`; needs an INVERTED index on the text column and a vector index on the embedding column.

## Build a vector index

`ds.create_index("vector", index_type=..., metric="l2", num_partitions=256, ...)`. Vector indexes default to `replace=False`: pass `replace=True` to rebuild under the same name. Default name `<column>_idx`; `name=` allows several indexes on one column. (guide/vector_search.md, guide/indexing.md)

| Type | Stores | Choose for | Cost |
|---|---|---|---|
| `IVF_FLAT` | original vectors | exact distances inside probed partitions, no quantizer training | larger index, more candidate I/O |
| `IVF_PQ` | product-quantized | smallest index, GPU training | lower recall without probes or refinement; `num_sub_vectors` required |
| `IVF_SQ` | scalar-quantized | simple compression, more information a dimension than PQ | larger than PQ |
| `IVF_RQ` `[13?]` | RaBitQ | compact multi-bit estimates; the docs' default recommendation | tune `num_bits` |
| `IVF_HNSW_FLAT` / `_SQ` / `_PQ` | as above plus an HNSW graph a partition | graph search when latency and recall call for it | graph storage, more parameters (`m`, `max_level`, `ef_construction`) |

- `num_partitions`: more partitions mean less work a probe and more probes for the same recall.
- `num_bits`: PQ takes 4 or 8; RQ defaults to 5 `[13?]`. RQ with `num_bits=1` gives the smallest index and build I/O, and searches lose the multi-bit estimate: measure recall.
- `ivf_centroids`, `pq_codebook`: pre-trained artifacts. `accelerator="cuda"` or `"mps"`: GPU training, named for IVF_PQ only.
- Size formulas and training memory are in `performance.md#vector-index` (not read for this file).

## Tune search and check recall

- Choose an index by measured recall and latency, not by size: recall is the overlap of the ids from an indexed search with those from the same search under `"use_index": False`, averaged over representative queries, under the same data, filter, cache and storage.
- Keys of `nearest`: `nprobes` (partitions searched), `minimum_nprobes`, `maximum_nprobes`, `refine_factor` (fetch `k * refine_factor` candidates, re-rank with more accurate distances), `ef` (HNSW breadth).
- `search_effort` `[13?]` in `[0, 1]`, interpolated between targets of mean routing recall: 0.1 = 80 %, 0.25 = 90 %, 0.5 (default) = 95 %, 1.0 = all partitions. It overrides `nprobes`; `minimum_nprobes` and `maximum_nprobes` still apply. Calibrated for l2, cosine and dot with `k <= 100000` on unfiltered data; the target is average coverage of the true neighbours' partitions, and quantization, refinement, filtering and HNSW move final Recall@k. Without it, tune `nprobes` and `refine_factor`. (guide/vector_search.md)

## Build a scalar index and make a filter use it

`ds.create_scalar_index("id", "BTREE")`, `ds.create_scalar_index("text", "INVERTED", name="text_fts_idx", with_position=True)`. Scalar indexes default to `replace=True`. The index yields candidate rows and Lance then evaluates the whole filter, so the choice of index changes speed only. (guide/scalar_indexing.md)

| Type | Column | Speeds up | Limit |
|---|---|---|---|
| `BTREE` | many distinct values | `=`, ranges, `IN`, null predicates | larger than skipping structures |
| `BITMAP` | few distinct values | `=`, small ranges, set membership | many distinct values grow size and build memory |
| `LABEL_LIST` | lists of few distinct elements | `array_has`, `array_contains`, `array_has_all`, `array_has_any` | high-cardinality arrays |
| `ZONEMAP` | roughly sorted | ranges, by skipping zones | little gain on random order |
| `NGRAM` | strings | `contains(col, 'sub')` | candidates may need rechecking |
| `BLOOMFILTER` | membership | `=` and `!=` | inexact, few predicates |
| `FM` `[13?]` | strings | exact `contains`, also one-character patterns | specialised |
| `LISTELEMENT` `[13?]`, experimental | `list<struct>` | `any_element(col, 'predicate')` | predicate must fix the first key |

- An `OR` uses an index only when every branch holds an indexed predicate: `id = 1 OR unindexed = 2` may scan; `(id = 1 AND unindexed = 2) OR id = 3` uses the index on `id`.
- Confirm use: `ds.scanner(filter="id >= 100").explain_plan()` shows `ScalarIndexQuery` or `MaterializeIndex`. `use_scalar_index=False` on the scanner turns indexes off for one query.
- Nulls follow SQL three-valued logic: a filter that evaluates to null is not true. Write `col IS NOT NULL AND col = 'x'` to exclude them, and test on null values, null list elements, empty lists and all-null columns.

## Keep indexes current

- Coverage: `ds.stats.index_stats("vector_idx")` gives `num_indexed_rows` and `num_unindexed_rows`. `ds.describe_indices()` (`.name`, `.index_type`, `.field_names`, `.num_rows_indexed`) `[13?]`. (guide/indexing.md)
- After appends: `ds.optimize.optimize_indices(index_names=["text_idx", "vector_idx"])`; omit `index_names` for all. For a vector index this assigns new vectors to the existing partitions and quantizer without retraining; add `retrain=True` when the distribution changed (a full rebuild, expensive).
- Order: compact first, then optimize or rebuild indexes; physical rewrites invalidate the row addresses indexes hold and Lance remaps them. Stable row ids keep identity across compaction and still need this upkeep.
- `fast_search=True` searches indexed rows only: check `num_unindexed_rows` first. The page writes only `fast_search=True` and leaves unclear whether it is a scanner argument or a key of `nearest`. For rows changed by an update `[13?]` ("overlay files"): stale index entries are always excluded; scalar filters always see current values; vector search, full-text search and MinHash see changed rows by default and skip them with `fast_search=True`.
- Change type, metric, tokenizer or training parameters: create again under the same name with `replace=True`. Remove: `ds.drop_index("id_idx")`.

## Store large binary values (blobs)

- Choose a blob column when values average roughly 1 MB or more, or when partial or streaming reads are needed. Inline `binary` is always read whole, and compaction and some updates copy inline payloads forward unchanged (write amplification); blob payloads sit in `.blob` files that are referenced. Below a few tens of KB with whole-value reads, inline `binary` is simplest and fastest. (guide/blob.md)
- Write Blob v2 (needs file format `>= 2.2`):
  ```python
  from lance import blob_array, blob_field
  schema = pa.schema([pa.field("id", pa.int64()), blob_field("blob")])   # blob_field(name, nullable=True)
  table = pa.table({"id": [1], "blob": blob_array([b"bytes"])}, schema=schema)
  lance.write_dataset(table, uri, data_storage_version="2.2")
  ```
  `blob_array` takes `bytes`, a URI string, `Blob.from_uri(uri, position=..., size=...)` or `None`, mixed in one column. An empty blob is inline `b""`; a range needs `uri` and `size > 0`.
- Placement by default: up to 64 KiB inline in the data file, up to 4 MiB packed into shared `.blob` sidecars, larger in a dedicated `.blob` file. Per column `[13?]`: `blob_field(..., inline_size_threshold=..., dedicated_size_threshold=...)`; the thresholds live in the schema and an append giving other values for that column is rejected. `blob_pack_file_size_threshold` `[13?]` is a write option that only rolls sidecar files.
- External URIs must map to a registered base path outside the dataset root; otherwise pass `allow_external_blob_outside_bases=True` on write.
- Blob v2 on an existing table: appending with `data_storage_version="2.2"` to a 2.0 or 2.1 table starts v2 with no upgrade step `[13?]`. From the first v2 file on, descriptor scans return the struct `kind, position, size, blob_id, blob_uri` for all fragments: adapt code that reads descriptors by position or compares exact schemas. `0.1` files cannot mix with `2.x` and need a rewrite.
- Legacy blobs (format `<= 2.1`): `pa.field("video", pa.large_binary(), metadata={"lance-encoding:blob": "true"})`. Convert by scanning with `blob_handling="all_binary"`, rebuilding with `blob_field` / `blob_array`, and writing at `"2.2"`, in batches (the page's example holds every payload in memory).
- Errors: "Blob v2 requires file version >= 2.2": the table was created below 2.2. "Legacy blob columns ... are not supported for file version >= 2.2": a low-level file writer received legacy metadata; write through the dataset writer.
- Blob v2 nested in structs and lists, and file format 2.3 `[13?]`.
- Managed blobs `[13?]` (schema metadata `lance.blob.managed = "true"`, payloads at `_blobs/<uuid>.blob`, formats 2.2 and 2.3): the opt-in sets a capability flag at once and for good, and older clients (the page names v11.0.0) running cleanup can delete sidecars still referenced. Turn it on only when every client that maintains the table supports it, and run cleanup without `delete_unverified=True` while uploads are prepared. Sharing is explicit, within one table, with no deduplication; the writer API is in guide/blob.md.

## Read blobs

| Call | Returns | Use for |
|---|---|---|
| `ds.read_blobs(col, indices=/ids=/addresses=)` `[13?]` | `[(row address, bytes or None)]` | whole payloads: loaders, batch work |
| `ds.read_blob_ranges(col, requests=[(row, offset, length)], selector="indices")` `[13?]` | `[(request_index, row address, bytes or None)]` | byte ranges from many rows |
| `ds.take_blobs(col, indices=/ids=/addresses=)` | `[BlobFile or None]` | file-like streaming and seeking |
| `lance.read_blob_files(handles)`, `lance.read_blob_file_ranges([(handle, offset, length)])` `[13?]` | `(request_index, data)` pairs | repeated reads of handles already held; `io_parallelism`, `io_buffer_size` |
| `ds.scanner(columns=["blob"], blob_handling="all_binary")` | binary column | blobs in a scan result |

- Pass exactly one of `indices` (positional, stable within one version), `ids` (logical row ids, stable when the table has them) and `addresses` (physical, unstable). Get them with `ds.to_table(columns=[], with_row_id=True)` (`_rowid`) or `with_row_address=True` (`_rowaddr`).
- For many whole payloads call `read_blobs`, which batches reads through Lance's scheduler; a thread pool calling `read()` on `take_blobs` handles bypasses it.
- A null blob comes back as `None`, an empty one as an empty payload. `BlobFile` buffers sequential reads at 512 KiB (`buffer_size=0` turns it off `[13?]`); `read_range` / `read_ranges` skip the buffer. Handles keep the locations resolved when opened, whatever is committed later. (guide/blob.md)

## Write from several workers, commit once

Workers write fragments; one process commits their metadata in a single operation. (guide/distributed_write.md)

```python
from lance.fragment import write_fragments
fragments = write_fragments(data, uri, schema=schema)             # each worker
payload = [json.dumps(f.to_json()) for f in fragments]            # send to the committer
frags = [lance.FragmentMetadata.from_json(p) for p in payloads]
lance.LanceDataset.commit(uri, lance.LanceOperation.Overwrite(schema, frags), read_version=0)   # new table
lance.LanceDataset.commit(uri, lance.LanceOperation.Append(frags), read_version=ds.version)     # append
```

- The page shows `id=0` for fragments from both workers and does not explain how ids are assigned at commit.
- Add a column: `new_frag, new_schema = fragment.merge_columns(fn, ["name"])` on each fragment (`fn` maps a `RecordBatch` to a `RecordBatch` of the new columns), then `LanceOperation.Merge([...], schema=new_schema)`. Existing data files are linked, not copied. The page's example covers every fragment before the commit.
- Update a column: `frag, fields = ds.get_fragment(i).update_columns(data, left_on="id", right_on="id")`, then `LanceOperation.Update(updated_fragments=[...], fields_modified=list(set(...)))`. It is a left-outer join: no match keeps the old value, several matches take the value of a random matching row, so give the update data one row a key. The join is an in-memory hash join or a spillable sort-merge join (`update_join.md`, not read) `[13?]`.

## Rewrite rows and keep their row ids

Applies to tables created with `enable_stable_row_ids=True` when you assemble the transaction yourself. All of it `[13?]`. (guide/distributed_write.md)

```python
from lance.fragment import RowIdSequence, write_fragments
fragments[0].row_id_meta = RowIdSequence(rewritten["_rowid"]).to_inline_metadata()   # rows read with with_row_id=True
old = ds.get_fragments()[0].delete("id in (2, 3)")                                   # tombstone the old copies
op = lance.LanceOperation.Update(updated_fragments=[old], new_fragments=fragments)
ds = lance.LanceDataset.commit(uri, op, read_version=ds.version)
```

- Pitfall: a fragment written without `row_id_meta` commits without error and silently gives every rewritten row a new `_rowid`. Set it on every rewritten fragment.
- In a fragment mixing rewritten and new rows, put the rewritten rows first and pass ids for those rows only; the commit assigns ids to new rows after conflict resolution, and more ids than rows is rejected.
- Leave `created_at_version_meta` and `last_updated_at_version_meta` as `None`: Lance derives them (columns `_row_created_at_version`, `_row_last_updated_at_version`).

## Build an index across workers

All of it `[13?]`. Lance supplies the calls; the caller launches workers and drives the flow. (guide/distributed_indexing.md)

1. Each worker: `create_index_uncommitted(..., fragment_ids=...)` writes one segment under `indices/<segment_uuid>/` (the page writes `_indices/` once).
2. The caller commits the segments with `commit_existing_index_segments(...)`, optionally after `merge_existing_index_segments(...)` on each group. Segments in one commit cover disjoint fragments.

- Merging works for vector, inverted, bitmap, BTree and zone map segments; other scalar types commit as several segments.
- Vector segments merge only when trained on shared artifacts: the same IVF centroids for every worker, and for IVF_PQ the same PQ codebook. Segments trained independently on their own fragments commit together as one index and return valid results (each ranks partitions with its own centroids; results merge by `_distance`): keep them separate.
- `LANCE_SHUFFLE_MAX_PRELOADED_OFFSETS_BYTES`: default 268435456 (256 MiB) a shuffle, summed over concurrent builds; `0` turns preloading off; it does not bound process memory.
- Segments never committed are removed by `cleanup_old_versions(...)` under its usual age rules.

## Export metrics

All of it `[13?]`. Python exports through OpenTelemetry only. (guide/observability.md)

- Install `pylance[otel]`, then `lance.otel.instrument_lance_metrics()` (global MeterProvider, or `meter_provider=...`).
- Memory metrics stay empty until `LANCE_MEMORY_OBSERVE=1` is set before the process starts, or `lance.otel.set_memory_observation(True)` is called.
- A histogram arrives as three counters: `<name>_bucket` (cumulative, attribute `le`, `le="+Inf"` = total), `<name>_count`, `<name>_sum`.
- The metric catalogue is in `rust/lance/src/metrics.md` (not read); the page shows one name, `lance_object_store_request_duration_seconds`.
