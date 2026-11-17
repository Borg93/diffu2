# Lance file and index format: what matters for the curation tables

Source: Lance format specification pages at docs version 15.0.0-beta, under
`docs/src/format/`, each read in full on 2026-11-17. Installed: pylance 13.0.0,
data storage version 2.2. Lines marked **[measured 13.0.0]** come from a small
test run on the installed pylance, not from the spec. Lines marked **[newer?]**
are repeated in the last section.

Our table: 3.2 M rows; unique string key `image`; low-cardinality strings
(`pool`, `kind`, `mark`); booleans; float scores; one JSON column; captions;
fixed-size-list float16 vectors of 2,048 to 4,096 numbers, some mostly null.

---

## file/index.md (the container)

- A Lance file = data pages + column metadata + offset tables + footer ending in `"LANC"`. Footer holds `u16` major and minor version.
- No row groups. Each column has its own pages; a column can be read or split at any row boundary.
- Page size: 8 MB recommended default. Larger pages cost writer memory, do not help reads.
- Buffers are aligned to 64 bytes.
- The container has no types and no encodings built in. Each page names its encoding in a protobuf; an unknown encoding gives an "unsupported encoding" error, not a wrong read. Different pages of one column may use different encodings.
- No statistics and no indexes inside the file. Those live in separate index files (see `index/`).
- "Global buffers" hold the schema and other auxiliary data. Index files use them too (IVF centroids, PQ codebook, null bitmaps).
- Opening a file costs 1 to 2 reads for the footer. Each page stores the row offset of its first row, so finding the page of a row is cheap.
- Large values may be stored out of line in external buffers, with positions kept in the page (blob layout, see encoding.md).

## file/versioning.md

One version number covers container and encodings. Major = container change, minor = encoding change.

| Version | Lance min | Adds |
|---|---|---|
| `0.1` (`legacy`) | any, write up to 0.34 | first format, no longer writable |
| `2.0` | 0.16.0 | no row groups; nulls for lists, fixed size lists, primitives |
| `2.1` | 0.38.1 | better integer and string compression; nulls in struct fields; faster random access on nested fields |
| `2.2` | none stated | "newer nested type/encoding capabilities (including map support) and 2.2-era storage features" |
| `2.3` (unstable) | none stated | sparse structural pages and other experimental encodings |
| `stable` | | alias: default for new datasets in the running release |
| `next` | | alias: latest unstable version in the running release |

- Unstable versions (`2.3`, `next`): no compatibility promise; files may become unreadable by a later build. Do not use for our tables.
- Pin the version explicitly (`data_storage_version="2.2"`); `stable` and `next` move with the release.
- Compatibility caveat (2.1 and later): a `FixedSizeList` column whose **inner values** are all null. Writers before Lance 11.1.0 wrote unreadable pages (`bits_per_value=0`). Writers from 11.1.0 store per-row validity bytes; such files need a reader of 11.1.0 or later (older readers panic on `Compression::Constant`). We write null **rows**, not null inner values, and run 13.0.0, so this does not bite unless an older reader opens the files.

## file/encoding.md

### Types
- Lance normalizes Arrow types on write. `string` and `large_string` take the same variable-width path; the Arrow type asked for at read time decides the offset width returned. Two paths in all: fixed-width and variable-width.

### Page layouts (2.1 and 2.2: `lance.encodings21.PageLayout`)

| Layout | Used for | Random access | Notes |
|---|---|---|---|
| Mini block | values under 256 bytes: ints, floats, booleans, short strings | read one whole mini-block (under 32 KiB compressed, at most 4096 values by default) | 2 bytes of metadata per mini-block held in the search cache; dictionary (if any) fully loaded at open |
| Full zip | values of 256 bytes or more, e.g. vectors, long captions | fixed-width: 1 read; variable-width: 2 reads (repetition index, then data) | compression must be "transparent": per value only |
| Constant | every visible value in the page identical, or page all null | no data | all-null page = constant page with no scalar; rep/def levels still stored if there are struct or list levels |
| Blob | large binary values, recommended from 1 MiB per value | 1 read per value | data out of line; page holds `position` and `size` |

- Cutoff between mini block and full zip: 256 bytes per value.
- Search cache: an LRU that holds per-column lookup data loaded at open (mini-block metadata, dictionaries, optionally repetition indexes). Full scans can skip it.

### Nulls
- Nulls are kept as definition levels, stored with the values (inside each mini-block, or as a control word before each full-zip value). No separate validity file.
- Mini block: null cost is the definition-level buffer in each block, compressed.
- Full zip: "If the value is null or an empty list, then the control word is all that is serialized." Control word is 0 to 4 bytes. A fixed-width column with no nulls has no control words: it is a flat buffer.
- Constant: a page where every row is null stores no values.

### Fixed size lists (our vectors)
- A fixed size list of primitives is treated as one primitive value (a tensor); the list is flattened during structural encoding.
- 2,048 x float16 = 4,096 bytes per row; 4,096 x float16 = 8,192 bytes per row. Both are over 256 bytes, so the layout is full zip, uncompressed (`Flat`).
- Null **items** inside a list are not rep/def levels: their validity is a separate buffer zipped with the values.
- **Cost of a mostly-null vector column [measured 13.0.0]**, 4,000 rows x 4,096 float16, versions 2.1 and 2.2 give the same sizes:

| Share of rows valid | Bytes on disk | Dense size |
|---|---|---|
| 100 % | 32,768,866 | 32,768,000 |
| 10 % | 32,772,900 | 32,768,000 |
| 1 % | 32,772,900 | 32,768,000 |
| 0 % (all null) | 828 | 32,768,000 |

  - A null row costs the full row width plus 1 byte, whenever its page holds at least one valid row. The spec sentence above ("the control word is all that is serialized") does not hold for fixed-width values in 13.0.0.
  - A page where every row is null costs nothing.
  - Scaled to 3.2 M rows: 4,096-d = 26.2 GB per column, 2,048-d = 13.1 GB per column, however many rows are null, unless the null rows fill whole pages (or whole fragments).
  - Ways to avoid the cost: keep a sparse vector in its own table keyed by `image`; or write rows so that nulls are contiguous and fill whole pages; or (not for production) format 2.3, whose writer uses sparse structure for "mostly-null wide values".
- `lance-encoding:compression=zstd` on the field changed nothing for these vectors **[measured 13.0.0]** (file 143 bytes larger). In full zip, general compression is applied automatically only to values of 32 KiB or more.

### Strings
- Short strings (under 256 bytes: `image` paths, `pool`, `kind`, `mark`): mini block. Compression: `Fsst` (the main algorithm for variable-width data; one symbol table per page, stored in the page protobuf) over `Variable`.
- Low-cardinality strings: dictionary encoding, decided per page. Tried when unique values fit `num_values / dict-divisor` (default divisor 2), the estimated size is under `dict-size-ratio` (default 0.8) of raw, the page has at least 100 values, and cardinality is at most 100,000. Dictionary values are compressed with `lz4` by default. The dictionary is loaded whole at open.
- Long strings (captions of 256 bytes or more): full zip, 4- or 8-byte length per value, FSST per value, plus a repetition index of one offset per row (packed to 1, 2, 4 or 8 bytes). Random access = 2 reads.

### Other fixed-width columns
- `Flat`: uncompressed fixed width. In mini block, the block is the largest power of two of values under 8,186 bytes.
- `Bitpacking`: integers; 1024 values per block.
- `Rle`: applied when runs / values is under the threshold (default 0.5). Types: u8, i8, u16, i16, u32, i32, f32, u64, i64, f64. At most 2048 values per mini-block. Good for sorted or low-cardinality columns.
- `ByteStreamSplit` (BSS): f32, f64, timestamps only. Applied only when `lance-encoding:compression` is also set; does not shrink data by itself.
- `General` (LZ4, zstd): opt in. Mini block: whole block compressed, after other encodings. Full zip: per value.

### Field metadata keys a user can set (schema field metadata, or writer options)

| Key | Values | Default |
|---|---|---|
| `lance-encoding:compression` | `lz4`, `zstd`, `none`, `fsst`, `pcodec` (2.3 only) | `none` |
| `lance-encoding:compression-level` | integer; zstd `0-22` (crate default 3); pcodec 0 to 12 (default 8); lz4 not configurable | by scheme |
| `lance-encoding:rle-threshold` | `0.0-1.0` (`0.0` disables RLE, `1.0` uses it whenever runs exist) | `0.5` |
| `lance-encoding:bss` | `off`, `on`, `auto` | `auto` |
| `lance-encoding:dict-divisor` | integer over 1 | `2` |
| `lance-encoding:dict-size-ratio` | `0.0-1.0` | `0.8` |
| `lance-encoding:dict-values-compression` | `lz4`, `zstd`, `none` | `lz4` |
| `lance-encoding:dict-values-compression-level` | integer | by scheme |
| `lance-encoding:general` | `off`, `on` | `off` |
| `lance-encoding:alp` | `off`, `on`, `auto` (2.3 only) | `auto` |
| `lance-encoding:packed` | any string (packed struct, opt in) | not set |
| `lance-encoding:structural-encoding` | `miniblock`, `fullzip`, `sparse` (`sparse` needs 2.3) | not set |

Environment variables: `LANCE_ENCODING_DICT_TOO_SMALL` (100), `LANCE_ENCODING_DICT_DIVISOR` (2), `LANCE_ENCODING_DICT_MAX_CARDINALITY` (100000), `LANCE_ENCODING_DICT_SIZE_RATIO` (0.8), `LANCE_ENCODING_DICT_VALUES_COMPRESSION`, `LANCE_ENCODING_DICT_VALUES_COMPRESSION_LEVEL`, `LANCE_MINIBLOCK_MAX_VALUES` (4096, max 32768). Priority for dictionary values: field metadata, then environment, then `lz4`.

- Packed struct: stores a struct row-major; fewer reads per row, no single-field reads. 2.1: fixed-width children only (`PackedStruct`); 2.2 adds variable-width children (`VariablePackedStruct`).

### 2.3 only (unstable, not for us)
- Planar page layout (`lance.encodings23.PlanarLayout`): structure plane (Dense or Sparse) and value plane (Chunked, Zipped, Indexed, Constant, None) chosen independently. Sparse structure stores positions of nulls or of valid rows; zipped values hold only materialized (non-null) values. "A null-only selection requires no value I/O". This is the fix for mostly-null wide columns.
- `Range`, `Delta`, block `Dictionary`, generic unsigned blocks, generic variable-width offsets.
- `ALP` (f32, f64): lossless, for values with few decimal digits; never enlarges data.
- `Pcodec` via `lance-encoding:compression=pcodec`: float16, float32, float64 and fixed-size lists of them, mini-block only, transposes items so each list position compresses as one sequence. Rejected together with `lance-encoding:structural-encoding=fullzip`. Not used for pages with null items. The only float16 vector compression in the spec.
- `AxisDelta`: integer tensors only.

---

## index/index.md (how all indexes work)

- Three categories: scalar, vector, system. An index is a separate, redundant structure over row identifiers. Loaded on demand and in parts; a dataset opens without loading any index.
- An index has a name, is defined on one or more columns, and is made of **segments** (one UUID each), stored under `_indices/{UUID}/`. Index files are immutable.
- Each segment covers a disjoint set of fragments, listed in `fragment_bitmap`. Segments need not cover every fragment: the engine runs the query on the index for covered fragments and scans the rest, then merges. Results stay correct; uncovered fragments are just slower.
- `IndexMetadata` fields: `uuid`, `name`, `fields`, `covering_fields`, `fragment_bitmap`, `index_details`, `version`, `metadata` (`map<string, string>`).

### What happens on each kind of write
| Write | Effect on an index |
|---|---|
| Append rows (new fragments) | New fragments are in no segment. Queries scan them. Brought in by a later index build over the uncovered fragments (a new segment) or a segment merge. **[measured 13.0.0]** after appending 500 rows to 6,000 indexed: `num_indexed_rows` 6000, `num_unindexed_rows` 500, for all six vector index types. |
| Update a column in place (row not deleted) | The engine must remove the affected fragment IDs from `fragment_bitmap` of every segment whose `fields` or `covering_fields` include that column. The whole fragment drops out of the index for that column and is scanned until re-indexed. Other columns' indexes are untouched. |
| Delete rows | Segment keeps the entries; deleted row addresses are filtered at query time using the deletion file. |
| Delete a whole fragment | Its ID is in the bitmap but not in the dataset; its rows are filtered out. |
| Compaction | Row addresses change. Three options: drop coverage (worst for queries); rewrite segments with remapped addresses (default; write amplification); record a Fragment Reuse Index and remap at read time. |

- Consequence for us: an in-place update of `mark` over many fragments un-indexes `mark` on all of those fragments at once; re-index after bulk updates. Updating `mark` does not touch the index on `image` or on a vector column.
- An engine that does not know an index type or version skips that segment and scans its fragments.
- Stable row IDs for indexes: no remap after compaction; updates only invalidate an index if a column in its `fields` or `covering_fields` changes; costs one extra lookup per query. "This feature is currently experimental."
- `covering_fields`: columns whose values a segment carries so a query can skip the base-table take. The segment's storage schema is authoritative, not the declaration. Flags `FLAG_COVERED_INDEX_METADATA`, `FLAG_INDEPENDENT_COVERING_FIELDS`. **[newer?]**
- Overlay files (updated values held in a data overlay file): rows covered by an overlay newer than the segment are excluded from index results and re-evaluated on the flat path. **[newer?]**
- Nonempty segment `metadata` needs `FLAG_OBJECT_METADATA` (`1 << 17`). **[newer?]**

---

## Scalar indexes

Result types: **Exact** = no recheck; **AtMost** = candidates, rechecked against the data.

### index/scalar/bitmap.md
- For: low-cardinality columns. Ours: `pool`, `kind`, `mark`, booleans.
- File: `bitmap_page_lookup.lance` with `keys` (the unique value) and `bitmaps` (serialized RowAddrTreeMap of row addresses).
- Queries: `=`, `BETWEEN`, `IN`, `IS NULL` (a pre-computed null bitmap).
- Cost: one bitmap per distinct value; grows with cardinality.
- Upkeep: general segment rules.

### index/scalar/btree.md
- For: exact lookups and ranges, sorted access, any cardinality. Ours: `image` (unique key), float scores.
- Files: `page_lookup.lance` (`min`, `max`, `null_count`, `page_idx`; cached in memory) and `page_data.lance` (sorted `values` and `ids`). Page size: schema metadata `batch_size`, default "4096" rows.
- Cost: stores every value plus a `UInt64` address per row. For `image` that is a second copy of all 3.2 M paths. In memory: a few MiB of page metadata even at 1 billion values.
- Queries, all exact: `=`, `BETWEEN`, `IN`, `IS NULL` (pages where `null_count > 0`).
- A BTree may index the logical `_rowid` when stable row IDs are on (`IndexMetadata.fields` = `-3`). **[newer?]**

### index/scalar/zonemap.md
- For: scan pruning on columns whose values are clustered by position. Ours: scores only if rows are written sorted or clustered; little use on shuffled data.
- File: `zonemap.lance`, one row per zone: `min`, `max`, `null_count`, `nan_count`, `negative_nan_count` (version 1 only), `fragment_id`, `zone_start`, `zone_length`. Schema metadata `rows_per_zone` (default "8192"), optional `null_bitmap`.
- Cost: tiny (one row per 8,192 rows: about 390 rows for 3.2 M).
- Queries: `=`, `BETWEEN`, `IN` are AtMost; `IS NULL` exact with the bitmap, otherwise AtMost.
- Types without ordering (whole `List`, `FixedSizeList`, `Struct`) get null bounds; only null counts are useful. So a zone map on a vector column can answer `IS NULL` but nothing else.
- Version 0 lacks `negative_nan_count`; its `max` may hold a `+NaN` marker. Version 1 **[newer?]**.

### index/scalar/zone_write_seeds.md
- Statistics per zone (normally 8,192 rows) saved inside the data file at write time, so a zone map can be built without rereading the column. Located by data-file schema metadata `lance.seed.<column_name> = <buffer_index>:<rows_per_zone>`; the buffer is an Arrow IPC file.
- Version 1 is a proposal: "It does not change the ZoneMap index format or enable version 1 writes". Needs `FLAG_ZONE_WRITE_SEEDS` (`1 << 16`), "reserved and unsupported until runtime support is implemented". Nothing for us to set. **[newer?]**

### index/scalar/bloom_filter.md
- For: membership tests on high-cardinality columns at small size. Ours: an alternative to BTree on `image` when only "is this path present" matters and a recheck is acceptable.
- File: `bloomfilter.lance`, one Split Block Bloom Filter per zone: `fragment_id`, `zone_start`, `zone_length`, `has_null`, `bloom_filter_data`. Metadata `bloomfilter_item` (default "8192" items per zone), `bloomfilter_probability` (default "0.00057"), `null_bitmap`.
- Queries: `=` and `IN` are AtMost (false positives, never false negatives); `IS NULL` exact.
- No ranges. A point lookup still reads the matching zones.

### index/scalar/label_list.md
- For: list columns of labels or tags (`list<string>` etc.). Not for our scalar strings.
- File: `bitmap_page_lookup.lance` (a bitmap index over labels).
- Queries, exact: `array_has` / `array_contains`, `array_has_all`, `array_has_any`.

### index/scalar/ngram.md
- For: substring search in strings. Ours: `image` paths (find by folder or volume fragment), captions.
- Indexes all 3-character sequences after ASCII folding and lowercasing. File: `ngram_postings.lance` (`tokens` UInt32 hashed trigram, `posting_list`).
- Queries, all AtMost with recheck: `contains`; `regexp_like` / `regexp_match`; infix `LIKE` such as `%foo%bar%`.
- Not accelerated (falls back to rechecking every row): patterns with no derivable trigram, e.g. `a.b`, `.*`, case-insensitive matches, literal runs under three characters.

### index/scalar/fts.md (INVERTED / FTS)
- For: word search with BM25 ranking in text. Ours: captions; also the JSON column as a whole (json document type).
- Files: `tokens.lance`, `docs.lance`, `invert.lance`, `metadata.lance`; with partitions, prefixed `part_<id>_`. Every partition is searched per query; fewer is faster.
- Parameters (`InvertedIndexParams`): `lance_tokenizer` (`text` or `json`, inferred when absent), `base_tokenizer` ("simple"), `language` ("English"), `with_position` (false), `max_token_length` (40), `lower_case` (true), `stem` (true), `remove_stop_words` (true), `custom_stop_words`, `ascii_folding` (true), `min_ngram_length` (3), `max_ngram_length` (3), `prefix_only` (false), `block_size` (128; 256 experimental), and code-tokenizer flags `split_identifiers`, `split_on_numerics`, `preserve_original`, `index_operators`.
- Base tokenizers: `simple`, `whitespace`, `raw`, `code`, `ngram`, `icu`, `icu/split`, `jieba/*`, `lindera/*`.
- **Swedish is a supported stemming and stop-word language**: set `language` to Swedish. With the defaults (`language` "English", `stem` true, `ascii_folding` true) Swedish captions are stemmed as English and å, ä, ö are folded to a, a, o; set `ascii_folding` false to keep them distinct.
- JSON documents are tokenized into triplets `path,type,value` (types `str`, `number`, `bool`, `null`); query as `address.city:San Francisco`.
- Queries, all AtMost: `contains_tokens(column, 'search terms')`, `match`, `phrase` (needs `with_position: true`), `boolean`, `multi_match`, `boost`.
- Build: tokenize in parallel workers, spill part files, merge. Environment: `LANCE_FTS_NUM_SHARDS` (default: number of CPUs), `LANCE_FTS_PARTITION_SIZE` (256 MiB), `LANCE_FTS_TARGET_SIZE` (4096 MiB), `LANCE_FTS_FORMAT_VERSION`. Peak memory about `NUM_SHARDS * PARTITION_SIZE`.
- Size: without positions, smaller than the column, under 2x the column in total disk; with positions, temporary space can be 10x the column or more. The `ngram` tokenizer makes a larger index.
- Format versions 1 to 3 implemented; "format version 4 is specified here but is not yet implemented". Existing indexes keep their version when updated or optimized.

### index/scalar/json.md
- Indexes **one path** of a JSON column: decodes the value at the path, builds an ordinary scalar index (the target, e.g. BTree or Bitmap) on it. Writes only the target's files.
- Target types: `JSON_TARGET_DATA_TYPE_BOOLEAN`, `_INT64`, `_FLOAT64`, `_UTF8`. A JSON null or absent path is indexed as null.
- Used only by `json_get_bool`, `json_get_int`, `json_get_float`, `json_get_string`, and only when the path argument is a literal equal to the indexed `path`. Query types and exactness are those of the target index.
- `json_extract` is deliberately **not** routed to the index (it returns quoted JSON text). Filter with the typed accessors.
- One index per path; several paths need several indexes.

### index/scalar/fmindex.md **[newer?]**
- Substring, prefix, suffix and regular-expression search on raw bytes (Burrows-Wheeler based). Served: `CONTAINS(column, "query_string")`.
- No normalization: case folding must be applied by the user to the column before indexing and to the query.
- Bytes `\x00` and `\xFF` are remapped to space at build time.
- Parameter `num_segments` at creation. Appends: a later `create_index` builds a separate segment over the new fragments. Merging segments re-reads the text and rebuilds (`merge_segments`).
- Not among the `create_scalar_index` type names of pylance 13.0.0.

### index/scalar/minhash_lsh.md (experimental) **[newer?]**
- Near-duplicate text search by estimated Jaccard similarity of token shingles. Query: `nearest = MinHashQuery(text, column)`; returns `_distance` = 1 - estimated similarity. Would serve "which captions or transcriptions are copies of this one".
- Parameters: `num_hashes` (128; 1 to 4096, multiple of `num_bands`), `num_bands` (16; 1 to 256), `shingle_size` (3), `tokenizer`. Threshold `(1 / num_bands) ^ (num_bands / num_hashes)`: 0.71 at the defaults.
- Cost per row: `8 + 2 * num_hashes` bytes (signatures) + `12 * num_bands` bytes (bands) = 456 bytes at the defaults; about 1.46 GB for 3.2 M rows.
- Appends: uncovered rows are still searched, signed on the fly, unless `fast_search`; `optimize_indices` adds a segment. Merging reuses stored signatures (no re-tokenizing). Changing a parameter means a rebuild.
- Its signature file sets field metadata `"lance-encoding:structural-encoding": "fullzip"` and `"lance-encoding:compression": "none"` on a fixed-size list of uint16: a worked example of those keys.
- Experimental: "its files may change incompatibly without a separate vote". Not among the pylance 13.0.0 type names.

### index/scalar/list_element.md (experimental) **[newer?]**
- For `list<struct>` columns: one entry per list element, keyed on chosen fields of the element; a wrapper over a BTree on `Binary`.
- Serves `any_element(list_column, 'predicate')` when the predicate fixes the first key with `=`, `IN` or `IS NULL` (then further keys in order, then one range).
- We have no `list<struct>` column. Not among the pylance 13.0.0 type names.

### index/scalar/rtree.md
- 2D spatial index on bounding boxes (Hilbert-sorted, packed). Files `page_data.lance`, `nulls.lance`. Serves `St_Intersects`, `St_Contains`, `St_Within`, `St_Touches`, `St_Crosses`, `St_Overlaps`, `St_Covers`, `St_Coveredby` (AtMost) and `IS NULL` (Exact). Geometry columns only; not for us.

### Scalar index type names accepted by pylance 13.0.0 `create_scalar_index` **[measured 13.0.0]**
`BTREE`, `BITMAP`, `LABEL_LIST`, `INVERTED`, `FTS`, `NGRAM`, `ZONEMAP`, `BLOOMFILTER`, `RTREE`, or an `IndexConfig`.

### Which index for which of our columns
| Column | Index | Why |
|---|---|---|
| `image` (unique string) | `BTREE` | exact `=` and `IN`; `BLOOMFILTER` if size matters more than exactness; `NGRAM` in addition for path-fragment search |
| `pool`, `kind`, `mark` | `BITMAP` | low cardinality; `=`, `IN`, `IS NULL` |
| booleans | `BITMAP` | two values plus null |
| float scores | `BTREE` | ranges; `ZONEMAP` only if rows are clustered by score |
| JSON column | JSON index per queried path (target BTree or Bitmap), filter with `json_get_*`; or `INVERTED` with the json tokenizer for token search over the whole document |
| captions | `INVERTED` / `FTS` for words (`language` Swedish); `NGRAM` for substrings and regex |
| mostly-null columns | `IS NULL` is exact from `BITMAP`, `BTREE`, `BLOOMFILTER`, and from `ZONEMAP` when its null bitmap is present |

---

## System indexes

### index/system/frag_reuse.md
- Fragment Reuse Index (FRI): one system-index entry named `__lance_frag_reuse`. Lets compaction skip the index remap: it records old-address to new-address mappings, and indexes are translated when loaded (then cached).
- Why: by default compaction remaps all indexes, which conflicts with concurrent index optimization and tends to make the compaction fail.
- Cost: extra translation work each time an affected index is loaded; longer chains cost more. Trim the history once the indexes have caught up.
- FRI version 0: compaction only. Version 1 adds `transitions` and "stable partition" reclustering (row map `_fri/<map_id>/stable_partition.lance`); first use sets `FLAG_FRAGMENT_REUSE_INDEX` (1024), which older clients must refuse. **[newer?]**
- With stable row IDs: flag `FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS`; an index that stores stable row IDs must not coexist with an FRI entry.
- Fragments covered by no index are scanned, so correctness never depends on a mapping.

### index/system/mem_wal.md
- MemWAL Index: the one system index that holds all MemWAL metadata (shard specs, indexes to maintain, SSTable compaction progress, shard state snapshots). At most one per table; works on primary-key and append-only tables. The page is a stub that points to `table/mem_wal.md`. Not used by our batch writes.

---

## index/vector/index.md

### Parts
An index type is `{clustering}_{sub_index}_{quantization}`; a `FLAT` sub-index is left out of the name.
- Clustering: IVF only (k-means partitions; a search examines only the nearest partitions).
- Sub-index: `FLAT` (scan the partition), `HNSW` (graph inside each partition).
- Quantization: `PQ` (product), `SQ` (scalar, per dimension), `RQ` (RabitQ: random rotation + binary codes), `FLAT` (none).

### Types named in the spec
| Type | Stored per vector (auxiliary file) | At 4,096-d, 3 M rows |
|---|---|---|
| `IVF_PQ` | `__pq_code`: `num_sub_vectors * num_bits / 8` bytes | 256 sub-vectors x 8 bits = 256 B/row = 0.77 GB |
| `IVF_SQ` | `__sq_code`: `dimension` bytes | 4,096 B/row = 12.3 GB |
| `IVF_HNSW_SQ` | as SQ, plus graph (`__vector_id`, `__neighbors`, `_distance`) | 12.3 GB + graph |
| `IVF_RQ` | `_rabit_codes`: `ceil(code_dim / 8)` bytes, + `__add_factors`, `__scale_factors`, `__error_factors` (float32 each); for `num_bits > 1` also `__blocked_ex_codes` etc. | 512 + 12 B/row = 1.6 GB at 1 bit; rotation matrix `[code_dim, code_dim]` float32 = 64 MB |
| `IVF_FLAT` | `flat`: `list<float32>[dimension]` | 16,384 B/row = 49 GB (float32, twice our float16 column) |

Each row also stores a `uint64` row address. Sizes above are arithmetic from the schemas, not measured.

### Parameters named in the spec
- Distance (`distance_type`): "l2", "cosine", "dot".
- IVF: number of partitions (centroids tensor `[num_partitions, dim]`, float32).
- PQ: `num_sub_vectors`, `nbits` (bits per sub-vector code, 8 = 256 codewords); codebook `[256, num_sub_vectors, dim/num_sub_vectors]` float32.
- SQ: `num_bits`, `bounds`.
- RQ: `num_bits` (1 to 9 bits per dimension), `code_dim`, `packed`, `query_estimator` (`residual_query` or `raw_query`).
- HNSW: `max_level` (7), `m` (20), `ef_construction` (150), `prefetch_distance` (2).
- PCA pre-transform for `IVF_RQ` (experimental): projects vectors to fewer dimensions before IVF and RQ; metric must be L2; index metadata version 4; schema metadata key `lance:pca`. **[newer?]**

### Files
- `index.idx`: search structure. Schema metadata `"lance:index"` (`type`, `distance_type`), `"lance:ivf"`, `"lance:flat"` or `"lance:hnsw"`. IVF centroids in a global buffer.
- `auxiliary.idx`: quantized vectors, partition by partition. Schema metadata `"distance_type"`, `"lance:ivf"`, `"storage_metadata"`, `"lance:rabit"`, `"covering_field_ids"`.
- Both readable with `lance.LanceFileReader`.
- Format version 3 is the latest documented: stores `_rowaddr` (physical row address) instead of `_rowid`.

### float16
- The spec says nothing about float16 input. Stored centroids, codebooks and the `IVF_FLAT` copy are float32.
- **[measured 13.0.0]** on 6,000 x 256-d nullable float16, cosine: `IVF_PQ`, `IVF_SQ`, `IVF_FLAT`, `IVF_HNSW_SQ`, `IVF_HNSW_PQ`, `IVF_RQ` all build and answer `nearest` queries. Null rows are skipped (`filter_nan=True` is the default). Not tested at 4,096-d or at 3 M rows.

### Upkeep
- Same segment rules as every index: appended fragments are unindexed and searched flat until an index build or optimize covers them; an in-place update of the vector column removes those fragments from the bitmap.
- IVF centroids and PQ codebooks are trained at creation; the spec does not say when they are retrained.

### First index to try for 3 M x 4,096-d float16 (from the spec's sizes, not measured)
1. `IVF_PQ`: smallest stable option (about 0.8 GB at 256 sub-vectors), lossy; the oldest and best documented type. Measure recall against a flat scan on a held-out sample.
2. `IVF_RQ` at `num_bits=1`: about 1.6 GB, no codebook training; compare recall with PQ.
3. `IVF_SQ` / `IVF_HNSW_SQ`: 12.3 GB, higher fidelity, only if PQ and RQ recall is too low.
4. `IVF_FLAT`: 49 GB; exact inside partitions; not worth it at this size.

---

## Unclear or possibly newer than 13.0.0

Newer or unstable (do not rely on without a test on 13.0.0):
- File format `2.3` and everything tied to it: planar layout, sparse structure, `lance-encoding:structural-encoding=sparse`, `lance-encoding:alp`, `lance-encoding:compression=pcodec`, `Range`, `Delta`, block dictionary, `AxisDelta`. All "unstable".
- FM-Index, MinHash LSH (experimental), List Element index (experimental): not in pylance 13.0.0's `create_scalar_index` type list.
- Zone map version 1 (`negative_nan_count`) and zone write seeds version 1 (`FLAG_ZONE_WRITE_SEEDS`, "reserved and unsupported until runtime support is implemented").
- `covering_fields`, `FLAG_COVERED_INDEX_METADATA`, `FLAG_INDEPENDENT_COVERING_FIELDS`; carried columns in vector auxiliary files (`covering_field_ids`).
- Data overlay files and their index exclusion rule.
- Index segment `metadata` map and `FLAG_OBJECT_METADATA`.
- Row addresses instead of row ids in index files (`_rowaddr`; bitmap, BTree, ngram, R-tree format version 1; label list version 2; vector version 3; FTS version 4 "not yet implemented"). Which of these 13.0.0 writes was not checked.
- BTree on `_rowid` (field ID `-3`).
- FRI index version 1 (stable partition, `transitions`), `FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS`.
- PCA pre-transform for `IVF_RQ` (experimental); RQ `num_bits > 1` blocked ex-codes and the `raw_query` estimator.
- FTS `block_size` 256 (experimental), the `code` tokenizer, `icu` tokenizers, `total_tokens` metadata.

Unclear in the spec:
- Null rows in a fixed-width full-zip column. The spec says only the control word is written for a null; 13.0.0 writes the full width (measured). Whether 15.x on format 2.2 differs is not stated; only 2.3 promises sparse storage.
- Whether `lance-encoding:compression` should compress full-zip values under 32 KiB when set on the field. The table marks General as per-value in full zip; in 13.0.0 it had no effect on 8 KB vectors.
- What "2.2-era storage features" means in the 2.2 row; no minimum Lance version is given for 2.2 or 2.3.
- float16 as vector-index input: not mentioned anywhere in the vector page; works in 13.0.0 at small scale (measured).
- `IVF_HNSW_PQ`: accepted by pylance 13.0.0 but not in the spec's table of common combinations.
- Vector index upkeep: no statement on centroid or codebook retraining, or on how appended rows are assigned to partitions.
- JSON index: the page documents JSONB decoding; whether our JSON column (if stored as a plain string) qualifies was not checked.
- Per-index build cost and size are given only for FTS, MinHash LSH and (in memory) BTree; the other pages give none.
- The spec pages name mechanisms, not the Python calls that trigger them (`optimize_indices` appears only on the MinHash page; `merge_segments` only on the FM-Index page).
