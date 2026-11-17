Covers: which index fits which column, what an index does after writes, what a column costs on disk, encoding settings, vector index choice and file format versions (Lance format spec, docs 15.0.0-beta, read 2026-11-17; installed pylance 13.0.0, storage version 2.2).
Read it before choosing or building an index, setting a `lance-encoding:*` key, sizing a vector column or its index, or changing `data_storage_version`.

Marks: **[measured 13.0.0]** = a small test on the installed pylance, not the spec. **[newer?]** = newer than 13.0.0 or unstable: test on a throwaway table before relying on it. The tag in brackets after a rule is its spec page under `docs/src/`.

## Which index for which column

Type names `create_scalar_index` accepts **[measured 13.0.0]**: `BTREE`, `BITMAP`, `LABEL_LIST`, `INVERTED`, `FTS`, `NGRAM`, `ZONEMAP`, `BLOOMFILTER`, `RTREE`, or an `IndexConfig`.

A query answer is **Exact** (no recheck) or **AtMost** (candidates, rechecked against the data).

| Column | Index | Serves |
|---|---|---|
| `image` (unique string) | `BTREE`; add `NGRAM` for path-fragment search | `=`, `BETWEEN`, `IN`, `IS NULL`, all Exact |
| `pool`, `kind`, `mark`, booleans | `BITMAP` | `=`, `BETWEEN`, `IN`, `IS NULL` (a pre-computed null bitmap) |
| float scores | `BTREE`; `ZONEMAP` only when rows are written sorted or clustered by score | ranges |
| captions | `INVERTED` / `FTS` for words, `NGRAM` for substrings and regex | see below |
| JSON column | a JSON index per queried path, or `INVERTED` with the json tokenizer for token search over the whole document | see below |
| mostly-null columns | `IS NULL` is Exact from `BITMAP`, `BTREE`, `BLOOMFILTER`, and from `ZONEMAP` when its null bitmap is present | |

- **BTREE** stores every value plus a `UInt64` address per row: on `image`, a second copy of all 3.2 M paths. In memory: a few MiB of page metadata even at 1 billion values. Page size: schema metadata `batch_size`, default "4096" rows. (format/index/scalar/btree.md)
- **BITMAP** holds one bitmap per distinct value, so it grows with cardinality. (format/index/scalar/bitmap.md)
- **ZONEMAP** is tiny: one row per zone, `rows_per_zone` default "8192", about 390 rows for 3.2 M. `=`, `BETWEEN`, `IN` are AtMost; `IS NULL` is Exact with the null bitmap, otherwise AtMost. Little use on shuffled data. Types without ordering (whole `List`, `FixedSizeList`, `Struct`) get null bounds, so on a vector column it answers `IS NULL` only. Version 0 lacks `negative_nan_count` and its `max` may hold a `+NaN` marker; version 1 **[newer?]**. (format/index/scalar/zonemap.md)
- **BLOOMFILTER** is the small alternative to BTREE on `image` when only "is this path present" matters and a recheck is acceptable. `=` and `IN` are AtMost (false positives, never false negatives); `IS NULL` Exact; ranges are unserved; a point lookup still reads the matching zones. Metadata `bloomfilter_item` (default "8192" items per zone), `bloomfilter_probability` (default "0.00057"). (format/index/scalar/bloom_filter.md)
- **NGRAM** indexes every 3-character sequence after ASCII folding and lowercasing. Serves `contains`, `regexp_like` / `regexp_match`, infix `LIKE` such as `%foo%bar%`, all AtMost. Patterns with no derivable trigram fall back to rechecking every row: `a.b`, `.*`, case-insensitive matches, literal runs under three characters. (format/index/scalar/ngram.md)
- For column kinds we lack: `LABEL_LIST` (`list<string>` labels; `array_has`, `array_has_all`, `array_has_any`, Exact) and `RTREE` (geometry).
- Absent from the 13.0.0 type names **[newer?]**: FM-Index (substring and regex search on raw bytes, no normalization), MinHash LSH (experimental; near-duplicate text by Jaccard similarity), List Element (experimental; `list<struct>` columns). Zone write seeds version 1 is a proposal, "reserved and unsupported until runtime support is implemented".
- The spec gives build cost and size only for FTS, MinHash LSH and (in memory) BTree.

### INVERTED / FTS on captions (format/index/scalar/fts.md)

- **Swedish**: set `language` to Swedish (a supported stemming and stop-word language) and `ascii_folding` false. The defaults (`language` "English", `stem` true, `ascii_folding` true) stem Swedish as English and fold å, ä, ö to a, a, o.
- `InvertedIndexParams` and defaults: `lance_tokenizer` (`text` or `json`, inferred when absent), `base_tokenizer` ("simple"), `language` ("English"), `with_position` (false), `max_token_length` (40), `lower_case` (true), `stem` (true), `remove_stop_words` (true), `custom_stop_words`, `ascii_folding` (true), `min_ngram_length` (3), `max_ngram_length` (3), `prefix_only` (false), `block_size` (128), and code-tokenizer flags `split_identifiers`, `split_on_numerics`, `preserve_original`, `index_operators`.
- Base tokenizers: `simple`, `whitespace`, `raw`, `code`, `ngram`, `icu`, `icu/split`, `jieba/*`, `lindera/*`. **[newer?]**: `block_size` 256 (experimental), the `code` and `icu` tokenizers, `total_tokens` metadata.
- Queries, all AtMost: `contains_tokens(column, 'search terms')`, `match`, `phrase` (needs `with_position: true`), `boolean`, `multi_match`, `boost`.
- JSON documents are tokenized into triplets `path,type,value` (types `str`, `number`, `bool`, `null`); query as `address.city:San Francisco`.
- Build environment: `LANCE_FTS_NUM_SHARDS` (default: number of CPUs), `LANCE_FTS_PARTITION_SIZE` (256 MiB), `LANCE_FTS_TARGET_SIZE` (4096 MiB), `LANCE_FTS_FORMAT_VERSION`. Peak memory about `NUM_SHARDS * PARTITION_SIZE`. Every partition is searched per query: fewer is faster.
- Size: without positions, smaller than the column and under 2x the column in total disk; with positions, temporary space can be 10x the column or more. The `ngram` tokenizer makes a larger index.
- Format versions 1 to 3 are implemented ("format version 4 is specified here but is not yet implemented"); an existing index keeps its version when updated or optimized.

### JSON index (format/index/scalar/json.md)

- Indexes one path: decodes the value there and builds an ordinary scalar index (the target, e.g. BTree or Bitmap) on it. One index per path. Target types `JSON_TARGET_DATA_TYPE_BOOLEAN`, `_INT64`, `_FLOAT64`, `_UTF8`; a JSON null or absent path is indexed as null. Query types and exactness are the target's.
- Filter with `json_get_bool`, `json_get_int`, `json_get_float`, `json_get_string`, passing the path as a literal equal to the indexed `path`: only these reach the index. `json_extract` is deliberately left unrouted (it returns quoted JSON text).
- Unchecked: the page documents JSONB decoding; whether our JSON column qualifies if stored as a plain string.

## What an index does after appends and updates

An index is made of immutable segments, each covering a disjoint set of fragments. The engine uses the index on covered fragments, scans the rest and merges: results stay correct, uncovered fragments are slower. An index type or version the engine does not know is skipped the same way. (format/index/index.md)

| Write | Effect |
|---|---|
| Append | New fragments are in no segment and are scanned until an index build over them (a new segment) or a segment merge. **[measured 13.0.0]** 500 rows appended to 6,000 indexed: `num_indexed_rows` 6000, `num_unindexed_rows` 500, for all six vector index types. |
| In-place column update | The whole fragment leaves every index on that column and is scanned until re-indexed. Indexes on other columns are untouched: updating `mark` leaves the `image` and vector indexes whole. |
| Delete rows | The segment keeps its entries; deleted rows are filtered at query time. |
| Compaction | Row addresses change. Either coverage is dropped (worst for queries), or segments are rewritten with remapped addresses (the default; write amplification), or a Fragment Reuse Index records the mapping for read time. |

- Re-index after a bulk update: an in-place update of `mark` over many fragments un-indexes `mark` on all of them at once.
- **Fragment Reuse Index** (`__lance_frag_reuse`): compaction skips the remap, and indexes are translated when loaded, then cached. The default remap conflicts with concurrent index optimization and tends to make the compaction fail. Each load of an affected index costs translation, more for longer chains: trim the history once the indexes have caught up. Version 1 (stable-partition reclustering; sets `FLAG_FRAGMENT_REUSE_INDEX`, which older clients must refuse) **[newer?]**. (format/index/system/frag_reuse.md)
- **Stable row IDs** for indexes: no remap after compaction, one extra lookup per query; "This feature is currently experimental." An index that stores stable row IDs must not coexist with an FRI entry. BTree on `_rowid` **[newer?]**.
- **Vector indexes** follow the same rules. IVF centroids and PQ codebooks are trained at creation; the spec is silent on retraining and on how appended rows are assigned to partitions.
- The spec names mechanisms, not Python calls: `optimize_indices` appears only on the MinHash page, `merge_segments` only on the FM-Index page.
- **[newer?]**: `covering_fields` (columns a segment carries so a query skips the base-table take); data overlay files (rows under an overlay newer than the segment are re-evaluated on the flat path); the segment `metadata` map; row addresses in place of row ids in index files (`_rowaddr`: bitmap, BTree, ngram, R-tree version 1; label list version 2; vector version 3). Which of these 13.0.0 writes was not checked.
- The MemWAL system index is unused by our batch writes.

## What a column costs on disk

The layout is chosen per page by value width (format/file/encoding.md):

- **Mini block**, values under 256 bytes (numbers, booleans, short strings): random access reads one whole mini-block (under 32 KiB compressed, at most 4096 values by default).
- **Full zip**, 256 bytes or more (vectors, long captions): 1 read for fixed-width values, 2 for variable-width.
- **Constant**, every value identical or the page all null: no data.
- **Blob**, recommended from 1 MiB a value: 1 read a value, data out of line.
- **Vectors** are full zip, uncompressed (`Flat`): 2,048 x float16 = 4,096 bytes a row, 4,096 x float16 = 8,192 bytes a row.
- **A null vector row costs the full row width plus 1 byte whenever its page holds one valid row; an all-null page costs nothing.** **[measured 13.0.0]**, 4,000 rows x 4,096 float16, same sizes on 2.1 and 2.2 (dense size 32,768,000):

| Rows valid | Bytes on disk |
|---|---|
| 100 % | 32,768,866 |
| 10 % | 32,772,900 |
| 1 % | 32,772,900 |
| 0 % | 828 |

  - At 3.2 M rows: 26.2 GB a column at 4,096-d, 13.1 GB at 2,048-d, however many rows are null.
  - For a sparse vector, use its own table keyed by `image`, or write rows so the nulls are contiguous and fill whole pages (or fragments).
  - The spec says otherwise ("If the value is null or an empty list, then the control word is all that is serialized"); that does not hold for fixed-width values in 13.0.0. Whether 15.x on 2.2 differs is unstated; only 2.3 promises sparse storage.
- **Compression leaves these vectors as they are**: `lance-encoding:compression=zstd` on the field made the file 143 bytes larger **[measured 13.0.0]**. In full zip, general compression is automatic only for values of 32 KiB or more; whether the key should compress smaller full-zip values is unclear in the spec.
- **Nulls elsewhere** are stored with the values: a compressed buffer in each mini-block, a control word of 0 to 4 bytes before each full-zip value.
- **Short strings** (`image`, `pool`, `kind`, `mark`): mini block with `Fsst`. `string` and `large_string` are stored alike; the Arrow type asked for at read time decides the offset width.
- **Few-valued strings**: dictionary, decided per page, when unique values fit `num_values / dict-divisor`, estimated size is under `dict-size-ratio` of raw, the page has at least 100 values and cardinality is at most 100,000. The dictionary is loaded whole at open.
- **Long captions** (256 bytes or more): full zip, FSST per value, plus a length and an offset per row.
- **Other fixed-width**: `Rle` when runs / values is under the threshold (integers, f32, f64): good for sorted or few-valued columns. `ByteStreamSplit` (f32, f64, timestamps) only together with `lance-encoding:compression`, and shrinks nothing by itself. `General` (LZ4, zstd) is opt in.
- **Page size**: keep the 8 MB default; larger pages cost writer memory and leave reads as fast. (format/file/index.md)
- A page with an unknown encoding raises "unsupported encoding" rather than reading wrongly. (format/file/index.md)

## Encoding settings

Field metadata keys (schema field metadata, or writer options) (format/file/encoding.md):

| Key | Values | Default |
|---|---|---|
| `lance-encoding:compression` | `lz4`, `zstd`, `none`, `fsst`, `pcodec` (2.3 only) | `none` |
| `lance-encoding:compression-level` | zstd `0-22` (crate default 3); pcodec 0 to 12 (default 8); lz4 fixed | by scheme |
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

- Environment variables: `LANCE_ENCODING_DICT_TOO_SMALL` (100), `LANCE_ENCODING_DICT_DIVISOR` (2), `LANCE_ENCODING_DICT_MAX_CARDINALITY` (100000), `LANCE_ENCODING_DICT_SIZE_RATIO` (0.8), `LANCE_ENCODING_DICT_VALUES_COMPRESSION`, `LANCE_ENCODING_DICT_VALUES_COMPRESSION_LEVEL`, `LANCE_MINIBLOCK_MAX_VALUES` (4096, max 32768). For dictionary values: field metadata, then environment, then `lz4`.
- Packed struct stores a struct row-major: fewer reads a row, no single-field reads. 2.1 takes fixed-width children only; 2.2 adds variable-width children.

## Vector index choice

A type is `{clustering}_{sub_index}_{quantization}`, a `FLAT` sub-index left out of the name: clustering IVF; sub-index `FLAT` (scan the partition) or `HNSW` (graph in each partition); quantization `PQ`, `SQ`, `RQ` (RabitQ), `FLAT` (none). (format/index/vector/index.md)

Try in this order for 3 M x 4,096-d float16. Sizes are arithmetic from the spec's schemas, not measured; each row adds a `uint64` address.

| Order | Type | Per vector | Total | Note |
|---|---|---|---|---|
| 1 | `IVF_PQ` | `num_sub_vectors * num_bits / 8` bytes: 256 B at 256 sub-vectors x 8 bits | 0.77 GB | smallest stable option, lossy, oldest and best documented; measure recall against a flat scan on a held-out sample |
| 2 | `IVF_RQ` at `num_bits=1` | `ceil(code_dim / 8)` bytes + three float32 factors: 512 + 12 B | 1.6 GB, plus a 64 MB rotation matrix | no codebook training; compare recall with PQ |
| 3 | `IVF_SQ` / `IVF_HNSW_SQ` | `dimension` bytes: 4,096 B (+ graph) | 12.3 GB (+ graph) | higher fidelity; only if PQ and RQ recall is too low |
| 4 | `IVF_FLAT` | float32 copy: 16,384 B | 49 GB, twice our float16 column | exact inside partitions; too large here |

- **float16 input** is unmentioned in the spec; stored centroids, codebooks and the `IVF_FLAT` copy are float32. **[measured 13.0.0]** on 6,000 x 256-d nullable float16, cosine: `IVF_PQ`, `IVF_SQ`, `IVF_FLAT`, `IVF_HNSW_SQ`, `IVF_HNSW_PQ`, `IVF_RQ` all build and answer `nearest` queries, and null rows are skipped (`filter_nan=True` is the default). Untested at 4,096-d or at 3 M rows.
- `IVF_HNSW_PQ` is accepted by 13.0.0 but absent from the spec's table of common combinations.
- Parameters: `distance_type` "l2", "cosine", "dot"; IVF number of partitions; PQ `num_sub_vectors`, `nbits` (8 = 256 codewords); SQ `num_bits`, `bounds`; RQ `num_bits` (1 to 9 bits a dimension), `code_dim`, `packed`, `query_estimator` (`residual_query` or `raw_query`); HNSW `max_level` (7), `m` (20), `ef_construction` (150), `prefetch_distance` (2).
- **[newer?]**: RQ `num_bits > 1` (blocked ex-codes) and the `raw_query` estimator; PCA pre-transform for `IVF_RQ` (experimental, L2 only, schema metadata key `lance:pca`).
- Inspect an index's `index.idx` and `auxiliary.idx` (the quantized vectors) with `lance.LanceFileReader`.

## File format versions

One number covers container (major) and encodings (minor). (format/file/versioning.md)

| Version | Lance min | Adds |
|---|---|---|
| `0.1` (`legacy`) | write up to 0.34 | no longer writable |
| `2.0` | 0.16.0 | nulls for lists, fixed size lists, primitives |
| `2.1` | 0.38.1 | better integer and string compression; nulls in struct fields; faster random access on nested fields |
| `2.2` | none stated | "newer nested type/encoding capabilities (including map support) and 2.2-era storage features" (meaning not explained) |
| `2.3` (unstable) | none stated | sparse structural pages and other experimental encodings |

- Write our tables with `data_storage_version="2.2"` given explicitly: the aliases `stable` (default for new datasets) and `next` (latest unstable) move with the release.
- Keep `2.3` and `next` to throwaway tables: they carry no compatibility promise and a later build may fail to read their files.
- What 2.3 would add **[newer?]**: sparse structure for "mostly-null wide values" (`structural-encoding=sparse`), the fix for mostly-null vector columns; `pcodec`, the only float16 vector compression in the spec (mini-block only, rejected with `structural-encoding=fullzip`, unused for pages with null items); `ALP` for f32 and f64.
- A `FixedSizeList` column whose inner values are all null (2.1 and later): writers before Lance 11.1.0 wrote unreadable pages (`bits_per_value=0`); files from later writers need a reader of 11.1.0 or later (older readers panic on `Compression::Constant`). We write null rows, not null inner values, on 13.0.0, so this bites only if an older reader opens the files.
