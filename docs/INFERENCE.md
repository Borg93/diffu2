# Drawing synthetic archive pages on a KubeRay cluster

This guide is for running the page generator on a Ray cluster on Kubernetes (KubeRay) with several Blackwell GPUs per
node. It assumes no knowledge of this repository. Every value that depends on your site is written as `<placeholder>`;
replace them all (`grep -n '<' deploy/kuberay/*.yaml` lists the ones in the manifests). How Ray Data runs, retries
and fails the stages, and why the code is built the way it is, is in [RAY.md](RAY.md).

## 1. What it does

A run turns a file of page jobs into synthetic archive pages: each job names a real page layout (a PAGE XML file, the
template) or gives one inline, the text to write into its lines (or none: its lines keep their own), the style (archive,
years, paper, ink) and a sample page whose hand, ink and paper to copy. A fine-tuned MOSS-Image model draws the page
whole; a handwriting reader (TrOCR) reads every line back; the page's exact ALTO and PAGE XML labels are written from
the plan, with every line the reader could not confirm tagged `unclear`. Ray Data does the scheduling: it reads the run
file, plans pages on CPU workers, and hands them to a pool of GPU actors, each holding one copy of the model, the same
batch pattern as Ray Data LLM's batch inference (a dataset of rows, a `map_batches` stage of a callable class with
`concurrency` actors, then a sink). The driver commits the drawn pages to a Lance table as they arrive, so a stopped run
resumes where it stopped.

```
run.jsonl  one page job per line (template or layout, text, style, sample page, seed, id)
   |
   |  ray.data.read_text(run.jsonl)        Ray Data reads the run file, a line a row
   v
   flat_map(plan_rows)                     CPU tasks: write the text, if any, into the page and plan it whole;
   |                                       an opening becomes its two pages; pages already in the table are skipped
   v
   map_batches(DrawActor,                  GPU actor pool, one MOSS-Image pipeline and one TrOCR judge per actor:
               concurrency=GPUs x A,       draw the page from its sample, read every line back, write ALTO and PAGE
               num_gpus=1/A)               (A = --actors-per-gpu)
   |
   v
   lance_ray.write_lance(stream=True)      the driver commits every --commit-every pages
   |
   v
<out>/pages.lance        one row per page: the JPEG (Blob v2), ALTO, PAGE, text, per-line reads, seed, provenance
<out>/runs/<run>.json    each run's settings, counts and timing
```

The code is `packages/runner` (`diffu-run`, `diffu-run-export`); the drawing itself is the page pipeline
(`packages/page`) around our own MOSS-Image implementation (`packages/moss`).

## 2. Prerequisites

- A Kubernetes namespace you can create Jobs, RayJobs, PersistentVolumeClaims and Secrets in, and `kubectl`.
- The KubeRay operator, 1.6 or newer (the manifests here were checked against the 1.7.1 CRDs).
- GPU nodes with the NVIDIA device plugin (`nvidia.com/gpu` resources) and an NVIDIA driver for CUDA 13 (580 or
  newer): the image's PyTorch is built for CUDA 13.0 and brings its own CUDA libraries.
- Storage: one volume for the models (about 45 GB, read by every pod), one for the inputs (templates, sample images,
  run files, settings; read by every pod) and one for the outputs (written by the head pod only). Every pod must mount
  the models and inputs at the same paths: a model directory and a run file name their files by absolute path.
- A container registry the cluster pulls from, and a machine with `docker buildx` to build the image.
- Network access to huggingface.co once, while staging the models (section 3). The run itself never downloads.

## 3. The model artifacts

| Artifact | What it is | Version | Size | Staged at (example) |
|---|---|---|---|---|
| MOSS-Image-Base, `text_encoder/` `tokenizer/` `vae/` | the frozen Qwen3.5-9B text encoder, its tokenizer and the FLUX.2-class VAE that every MOSS-Image model shares | `OpenMOSS-Team/MOSS-Image-Base` at revision `85975793d8b3d432f1b6e27f352abaa4c2a4c0d3` | 16.8 GB + 0.02 GB + 0.25 GB | `/models/MOSS-Image-Base` |
| The fine-tuned DiT | our MOSS-Image transformer trained to draw from a sample page: `config.json`, `diffusion_pytorch_model.safetensors` (5.1 B parameters, bf16) and `export.json` (it says the DiT takes a sample image) | [`Gabriel/diffu-moss-r6b`](https://huggingface.co/Gabriel/diffu-moss-r6b) at commit `9da723d920d580abc8f462117f05c0b6f865c88b` (public), or a checkpoint folder you are handed | 10.2 GB | `/models/dit/diffu-moss-r6b` |
| The handwriting judge | the Swedish TrOCR model every drawn line is read back with | `Riksarkivet/trocr-base-handwritten-hist-swe-2` at revision `aa79fcb1850bf3155ebc442570d6c6bfc0ac8100` | 1.5 GB | `/models/trocr-base-handwritten-hist-swe-2` |
| The model directory | a 2 KB `modular_model_index.json` naming where the text encoder, tokenizer, VAE and DiT are (written by `diffu-moss-init`) | written on the cluster | - | `/models/moss-<dit-name>` |
| A hand's adapter (optional) | a PEFT LoRA adapter of Base's own DiT, one collection's handwriting ("A hand's adapter", below) | `Gabriel/diffu-moss-hand-<hand>`, four, public | 0.16 GB | `/models/adapters/<hand>` |

Only safetensors are loaded, no remote code runs (`trust_remote_code` is never set), and the image sets
`HF_HUB_OFFLINE=1`: a run reads the staged files and nothing else. The Base snapshot's own `transformer/` is not needed
when you draw with a fine-tuned DiT.

### Staging them on the cluster

Pick one way to get the files onto the models volume.

**A. A PersistentVolumeClaim filled by a staging Job (recommended).** `deploy/kuberay/volumes.yaml` creates the three
claims (models, data, outputs); `deploy/kuberay/stage-models.yaml` runs a Job in the inference image that downloads
the Base folders and the judge at their pinned revisions, fetches the DiT (`Gabriel/diffu-moss-r6b` as it is set, or
another model repository, see C), and writes the model directory. Fill in their placeholders, then:

```bash
kubectl -n <namespace> apply -f deploy/kuberay/volumes.yaml
kubectl -n <namespace> create secret generic diffu-hf-token --from-literal=token=<a read token>  # a private DiT only
kubectl -n <namespace> apply -f deploy/kuberay/stage-models.yaml
kubectl -n <namespace> logs -f job/diffu-stage-models
```

The data volume (`diffu-data`, mounted at `/data`) is yours to fill: the templates, the sample images, the run files
and `draw.json`, copied in with your site's tool (`kubectl cp` into a pod that mounts it, rsync, object storage).

**B. Object storage.** Copy the same folder layout into the volume with your site's tool, for example from a pod that
mounts the claim (`aws s3 sync s3://<bucket>/models/ /models/`, `rclone copy <remote>:models /models`), then write the
model directory as in "The model directory" below.

**C. A model repository.** The published DiT is `Gabriel/diffu-moss-r6b`, public, at commit
`9da723d920d580abc8f462117f05c0b6f865c88b`: the staging Job fetches it as it is set (`DIT_NAME` `diffu-moss-r6b`, so
the model directory is `/models/moss-diffu-moss-r6b`), with no token. A DiT of your own is uploaded once by the person
who trained it:

```bash
hf upload <org>/<dit-repo> <path to the checkpoint>/transformer . --private
```

and the staging Job, given its `DIT_REPO`, `DIT_REVISION` and `DIT_NAME`, downloads it with a read token from a Secret
(`hf download <org>/<dit-repo> --revision <commit> --local-dir /models/dit/<dit-name>`). Pin the commit, as the Base
and the judge are pinned.

Whichever way: the files must be readable by the image's user (UID 10001). A checkpoint copied with its permissions
kept can arrive as `-rw-------`; make it readable (`chmod -R a+rX /models`).

### The model directory

`diffu-moss-init` writes a small model directory naming the Base folders and the DiT by **absolute path**, so write it
on the cluster, with the volume mounted where the RayJob mounts it (`/models`); a directory written elsewhere is
refused at the start of a run ("names component folders that are not on this machine"). The staging Job does this;
by hand, in any pod of the image that mounts the volume read-write:

```bash
diffu-moss-init --base /models/MOSS-Image-Base --transformer /models/dit/<dit-name> --out /models/moss-<dit-name>
```

It prints whether the model takes a sample image (`takes a reference image`): a DiT trained with sample pages does,
and only such a DiT may be given pages with a `reference` (section 5).

### A hand's adapter (optional)

A hand LoRA draws in one collection's handwriting from the caption alone, no sample page: a PEFT LoRA adapter of
MOSS-Image-Base's own DiT (156 MB), loaded onto the DiT as each actor loads the pipeline. Four are published, public:
`Gabriel/diffu-moss-hand-pahlman`, `Gabriel/diffu-moss-hand-landskapshandlingar`,
`Gabriel/diffu-moss-hand-overstathallarambetet-1715` and `Gabriel/diffu-moss-hand-eight` ([MOSS.md](MOSS.md), "Hand
LoRAs"). A hand needs Base's `transformer/` staged too, and a model directory of Base's own DiT; stage both as the rest
is staged (in a pod with network access, the volume mounted read-write):

```bash
hf download OpenMOSS-Team/MOSS-Image-Base --revision 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3 \
  --include "transformer/*" --local-dir /models/MOSS-Image-Base
hf download Gabriel/diffu-moss-hand-pahlman --local-dir /models/adapters/pahlman
diffu-moss-init --base /models/MOSS-Image-Base --out /models/moss-base
```

The run then names the model directory and the adapter, `--model /models/moss-base --lora hand=/models/adapters/pahlman`
(section 7), and every page of it is drawn with that hand: one hand a run, and one `--out` holds the pages of one
model, adapter and settings. Its pages give no `reference` (Base takes none), and their `style` names the collection
as the hand was trained on it (the adapter's model card). In `draw.json`, `"fuse_loras": true` merges the adapter into
the DiT's weights at load, so a step runs no LoRA matmuls; MXFP8, a package and `compile_blocks` (section 12) take an
adapter only fused.

## 4. Building the image

`.docker/inference.dockerfile` builds one image for the head and the GPU workers: Ubuntu 24.04, a uv-managed Python
3.13, and the workspace installed from `uv.lock` (PyTorch 2.14 for CUDA 13.0, Ray 2.59, diffusers at the commit the
lock pins, Lance) as wheels, running as UID 10001, with no weights in it. From the repository root:

```bash
docker buildx build -f .docker/inference.dockerfile --platform linux/amd64 \
  --build-arg BUILD_DATE=$(date -u +%FT%TZ) --build-arg VCS_REF=$(git rev-parse HEAD) \
  --build-arg VERSION=$(git describe --always) \
  -t <registry>/diffu-inference:<tag> --push .
```

Use `--platform linux/arm64` for Grace (GB200) nodes. The build needs network access (it downloads the locked wheels)
and about 15 GB of disk. The image is about 7 GB (6.8 GB built for arm64), nearly all of it PyTorch and the CUDA
libraries.

To try it on one machine with a GPU: `docker run --rm --gpus all <registry>/diffu-inference:<tag> diffu-run --help`.

## 5. Writing a run file

A run file has one page job per line (JSON Lines), the input of `diffu_page.draw.jobs.PageJob`:

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | the page's name: unique in the run file and across everything written into one `--out`. It names the image (`<id>.jpg` in the labels), seeds the noise, and is what a resumed run skips by. A plain file name (no `/`). |
| `template` | one of `template` and `layout` | a PAGE or ALTO file whose layout to draw (its lines, regions and ruled lines are kept) |
| `layout` | one of `template` and `layout` | the page's layout inline: `{"width": ..., "height": ..., "lines": [...]}` in pixels. A line needs only its `box`, `[x0, y0, x1, y1]`; it may also give `text`, `kind` (default `body`, running text; or `heading`, `marginalia`, `page-number`, ...), `production` (`printed` or `handwritten`; default: drawn as handwriting), `baseline_y` (default the box's bottom) and `block` (lines of one block are read together, top to bottom; default one block). Optional `rules`, its ruled lines: `{"x0": ..., "y0": ..., "x1": ..., "y1": ...}`. A layout is one page: it takes no `side` and is never split. |
| `text` | no | the text to write into the running-text (`body`) lines, in order (a form's printed lines among its handwriting keep theirs); a blank line ends a paragraph. Lines beyond the end of the text are left out; other lines keep their own text. Without it every line keeps its own text, so each line of a `layout` must then hold one. |
| `style` | yes | `{"provenance": {"title": ..., "years": ...}, "look": {"greyscale": ..., "paper": ..., "ink": ...}}`: paper is one of grey, light grey, white, off-white, cream, yellowed; ink one of black, dark grey, faded grey, dark brown, brown, faded brown |
| `reference` | for a model that takes a sample image | `{"image": <a sample page of the hand>, "caption": "sample"}`. `caption` must be a form the DiT was trained on; the eight-hands fine-tunes know `named`, `hand` and `sample`. |
| `side` | no | `left` or `right` to draw one page of a template's two-page opening; without it an opening is drawn as both its pages, as `<id>_left` and `<id>_right` |
| `page_pixels` | no | the page's size in pixels (default: `--page-megapixels`, 1.0 MP, the eight-hands fine-tunes' training size) |
| `seed` | no | added to the id's own seed (default 0) |

The layout and the text are two separate choices. Text written into a template, or into a layout whose lines hold text,
takes as many characters a line as the page's own lines do (their median characters per pixel of line width, over the
lines of 10 characters or more). Text written into lines with no text to measure, a layout of boxes alone, takes about
3.7 characters per unit of a line's width over its height (`CHARS_PER_LINE_ASPECT`, the median over 24,807 handwritten
GT lines; half of them hold 2.8 to 4.8), so a 1600 x 80 px box takes about 74.

A worked example, `/data/runs/example.jsonl`, one line of each combination (the first draws the right page of an
opening; the second, with no side, both pages if its template is an opening, as `pahlman-0001_left` and
`pahlman-0001_right`):

```json
{"id": "jamtland-0001", "template": "/data/templates/jamtland/0105.xml", "side": "right", "text": "Anno 1649 den 6 Aprilis Tå laga tingh höltz medh Allmogen uthi Räfsundh ...", "style": {"provenance": {"title": "the court records of Jämtlands domsaga", "years": "1649"}, "look": {"greyscale": false, "paper": "cream", "ink": "black"}}, "reference": {"image": "/data/samples/jamtland/sample-01.jpg", "caption": "sample"}}
{"id": "pahlman-0001", "template": "/data/templates/pahlman/0015.xml", "style": {"provenance": {"title": "Carl Fredrik Påhlman's travel journals", "years": "1826"}, "look": {"greyscale": false, "paper": "off-white", "ink": "dark grey"}}, "reference": {"image": "/data/samples/pahlman/sample-03.jpg", "caption": "sample"}, "seed": 7}
{"id": "boxes-0001", "layout": {"width": 1400, "height": 2000, "lines": [{"box": [150, 200, 1250, 270], "kind": "heading", "text": "Anno 1650"}, {"box": [150, 320, 1250, 390]}, {"box": [150, 410, 1250, 480]}, {"box": [150, 500, 1250, 570]}]}, "text": "Tå laga tingh höltz medh Allmogen uthi Räfsundh ...", "style": {"provenance": {"title": "the court records of Jämtlands domsaga", "years": "1650"}, "look": {"greyscale": false, "paper": "cream", "ink": "black"}}, "reference": {"image": "/data/samples/jamtland/sample-01.jpg", "caption": "sample"}}
{"id": "boxes-0002", "layout": {"width": 1400, "height": 2000, "lines": [{"box": [150, 200, 1250, 270], "text": "Anno 1650 den 3 Maij"}, {"box": [150, 290, 1250, 360], "text": "kom för Rätten Olof Jonsson"}]}, "style": {"provenance": {"title": "the court records of Jämtlands domsaga", "years": "1650"}, "look": {"greyscale": false, "paper": "cream", "ink": "black"}}, "reference": {"image": "/data/samples/jamtland/sample-01.jpg", "caption": "sample"}}
```

They are, in order: a template with new text; a template keeping its own text; a layout of boxes with new text (its
heading keeps its own); a layout keeping its lines' own text.

A run file is usually written by a script from a list of templates or layouts, texts and samples; build each line
with `PageJob(...).model_dump_json()` and it is valid by construction. Name every template and sample by its absolute
path on the volumes (pages are planned and drawn on other nodes, in working directories of their own). A run is
refused before any model loads if a line is not a page job (both or neither of `template` and `layout`; a layout with
a `side`, no lines, a box off its page or a line left without text), an id is missing or repeated, a template or
sample image is named by a relative path or is not on the volume, or a sample is given to a model that takes none.
Keep run files to tens of thousands of lines: the head reads the whole file once to check it.

## 6. Draw settings

The draw settings are one JSON object, `--draw-settings /data/runs/draw.json`, of
`diffu_page.draw.pipeline.DrawSettings` fields; a field it leaves out keeps its default, and without the option every
default holds. For example:

```json
{"steps": 50, "guidance_scale": 3.0, "device": "cuda"}
```

| Field | Default | Meaning |
|---|---|---|
| `steps` | 50 | denoising steps per page: the main speed and quality knob |
| `guidance_scale` | MOSS-Image's own, 3 | how hard guidance pushes toward the caption |
| `reference_guidance_scale` | 1 | above 1 also pushes toward the sample page (a third pass a step, so slower) |
| `guidance_stop` | 1 | the share of the steps after which guidance stops |
| `sampling_shift` | MOSS-Image's own, 5 | the timestep shift |
| `max_line_cer` | 0.25 | a line the judge reads with a higher character error rate is tagged `unclear` in the labels |
| `lora_scale` | 1 | how strongly a hand's adapter (`--lora`, section 3) applies: 1 as trained, 0 not at all |
| `fuse_loras` | false | merge the adapter into the DiT's weights at load: no LoRA matmuls a step, but it can no longer be switched |
| `device` | `cuda` | `cpu` only for tests (with `--gpus 0`) |

The settings reach every actor whole: a field the page pipeline gains (such as the precision options of section 12)
works in a run without a change to the runner. A key that is not a field is refused, so a typo cannot silently fall
back to a default. `repair_passes` is refused: a run draws each page once (redrawing rejected lines is to become its own
stage). One `--out` holds pages of one model and one set of draw settings: a run into an `--out` drawn otherwise is
refused.

## 7. Submitting a run

`deploy/kuberay/rayjob.yaml` is a RayJob: KubeRay starts a Ray cluster (a CPU head and a group of GPU workers), runs
`diffu-run` on the head once the cluster is up, and deletes the cluster when the run ends. Fill in its placeholders
(image, claims, node selector, replicas, the run file), then:

```bash
kubectl -n <namespace> apply -f deploy/kuberay/rayjob.yaml
kubectl -n <namespace> get rayjob diffu-pages -w                 # JOB STATUS: RUNNING, then SUCCEEDED
kubectl -n <namespace> logs -f job/diffu-pages                   # the submitter: the run's own output
kubectl -n <namespace> port-forward svc/<head service> 8265:8265  # the Ray dashboard (kubectl get svc lists it)
```

The run's entry point, as the RayJob gives it:

```bash
diffu-run --jobs /data/runs/example.jsonl --draw-settings /data/runs/draw.json \
  --model /models/moss-<dit-name> --out /outputs/<run-name> \
  --htr-model /models/trocr-base-handwritten-hist-swe-2 --htr-revision "" \
  --gpus <total GPUs> --actors-per-gpu 1
```

It ends with one line, for example `run 20261101T120000123456Z: 1998 pages drawn, 2 failed, 0 already drawn before
it; 1210 pages an hour over 99.1 min, 23.4 s a page on an actor (median)` (the numbers here are made up).

A run in a hand's handwriting (section 3, "A hand's adapter") names Base's model directory and the hand's adapter:

```bash
diffu-run --jobs /data/runs/pahlman.jsonl --draw-settings /data/runs/draw.json \
  --model /models/moss-base --lora hand=/models/adapters/pahlman --out /outputs/pahlman \
  --htr-model /models/trocr-base-handwritten-hist-swe-2 --htr-revision "" --gpus <total GPUs>
```

## 8. Scaling knobs

| Knob | Where | What it does |
|---|---|---|
| GPU worker pods | `workerGroupSpecs[].replicas` | nodes (or node slices) of GPUs |
| GPUs per pod | `nvidia.com/gpu` in the worker's resources | KubeRay gives Ray as many GPUs |
| `--gpus` | entry point | GPUs the run draws on (default: every GPU the cluster has when it starts); set it to replicas x GPUs per pod |
| `--actors-per-gpu` | entry point | pipelines sharing a GPU, each a whole copy of the model (about 35 GB: the DiT 10 GB, the text encoder 17 GB, the VAE, the judge and the activations; to be measured). 1 by default; 2 may fit a 96 GB card, and helps only if one actor leaves the GPU idle (measure, section 11) |
| `--cpus-per-actor` | entry point | CPUs Ray keeps for each actor (default 1); pages are planned on the rest, so leave some |
| `--commit-every` | entry point | drawn pages per commit to the table (default 64): what a crash can lose, and how many small files the table gets |
| `--page-megapixels` | entry point | the size a page is drawn at, unless its job gives `page_pixels` (default 1.0) |
| `steps` | draw settings | time per page is about proportional to it |
| `--attention` | entry point | the attention kernel (section 15) |

Memory per GPU pod: each actor reads about 28 GB of weights through the host's memory while it loads; give a pod at
least 48 GiB per actor plus 16 GiB, and a `/dev/shm` of about 16 GiB for Ray's object store (the manifest does).

## 9. Outputs and how to read them

`<out>/pages.lance` is a Lance table (file format 2.2), one row per page:

| Column | Type | Meaning |
|---|---|---|
| `id` | string | the page's id |
| `image` | Blob v2 | the page as JPEG; null when it failed |
| `width`, `height` | int32 | its size in pixels |
| `alto_xml`, `page_xml` | large string | its labels, ALTO 4 and PAGE 2019; both name the image `<id>.jpg` |
| `text` | large string | its lines' texts in reading order, a line each |
| `lines` | list of {text, read, cer, numbers_differ} | each line's text and what the judge read in its box |
| `unclear_lines` | int32 | lines the labels tag `unclear` |
| `seed` | int64 | the noise seed it was drawn from |
| `template` | string | its template; null for a page given as a layout |
| `job` | large string | the page job as drawn (JSON): it draws the page again |
| `model` | string | the model directory that drew it |
| `run` | string | the run that drew it (`<out>/runs/<run>.json`) |
| `seconds` | float32 | drawing, judging and labelling it on its actor |
| `error` | string | why it failed; null when it was drawn |

As files, to look at or to hand to a tool that reads files (`<id>.jpg`, `<id>.alto.xml`, `<id>.page.xml`):

```bash
diffu-run-export --out /outputs/<run-name> --to /outputs/<run-name>/files --limit 20
diffu-run-export --out /outputs/<run-name> --to /tmp/two --id jamtland-0001 --id pahlman-0001_left
```

In Python (`pip install pylance==13.0.0`, or inside the image):

```python
import lance

pages = lance.dataset("/outputs/<run-name>/pages.lance")
print(pages.count_rows(), "rows;", pages.count_rows(filter="error IS NULL"), "drawn")
failed = pages.to_table(columns=["id", "error"], filter="error IS NOT NULL").to_pylist()
for batch in pages.to_batches(
    columns=["id", "image", "alto_xml"], filter="error IS NULL", batch_size=64, blob_handling="all_binary"
):
    for row in batch.to_pylist():
        jpeg, alto = row["image"], row["alto_xml"]  # bytes and str
```

On a Ray cluster, `lance_ray.read_lance("/outputs/<run-name>/pages.lance", columns=[...])` reads it as a Ray dataset,
for a later pass such as a judge. A long run leaves many small fragments (one per commit): compact the table once it
is done, `lance.dataset(path).optimize.compact_files()`.

## 10. Resuming after a failure

Run the same command again, with the same `--out`. The run reads the ids the table holds drawn and skips them, deletes
the rows of pages that failed and draws those again, and appends; nothing is drawn twice. What happens to the work in
flight when something breaks:

| What broke | What happens |
|---|---|
| a page cannot be planned or drawn (an unreadable template, GPU memory) | the page's row keeps the error and the run goes on; a rerun draws it again |
| an actor's process dies (a crash, a node lost) | Ray restarts it and draws its page again, up to 3 times |
| a CUDA fault (an illegal memory access: the device cannot be trusted) | the actor exits on purpose; Ray restarts it, as above |
| the head pod or the driver dies, or the RayJob is deleted | the pages committed so far stay; at most `--commit-every` pages are lost. Rerun. |

A rerun is refused if `<out>/runs/` shows earlier runs with another model, draw settings, page size or judge. Two runs
must not write into one `--out` at the same time.

## 11. Measuring throughput

Throughput on Blackwell has **not been measured yet**: all numbers so far are from a DGX Spark (GB10, unified memory),
which do not carry over. To measure it on your cluster:

1. Draw a short run of real pages, about 8 pages per actor, with the settings you mean to use.
2. Read its summary line, or `<out>/runs/<run>.json`: `drawn`, `started` and `finished` give pages an hour including
   the model load; `median_seconds` is the steady time per page on one actor, so the pool's steady rate is about
   `actors x 3600 / median_seconds` pages an hour.
3. Run it again with `--actors-per-gpu 2` (into a new `--out`) and keep whichever draws more pages an hour, if the
   card's memory holds both (`nvidia-smi` on a worker while it runs).
4. For where the time goes on one GPU, profile with the single-machine command on three pages whose lines give
   `page_pixels` (with `--jobs`, `diffu-generate` draws a page whole only when its line says so): `diffu-generate
   --jobs <run file> --model /models/moss-<dit-name> --out <dir> --profile <trace dir>` records pages 2 and 3 after
   a warm-up page; the trace's kernel names show the attention kernel.

A yardstick: SGLang's MOSS-Image integration reports 0.296 images a second at 1024 x 1024, 20 steps and batches of 4
on one H200 (BF16, eager). Our pages are not that workload: 50 steps by default, the sample page in every pass, and
the judge after each page.

## 12. Faster inference: MXFP8, compiled blocks, AOTInductor packages

Three options of `DrawSettings` make a MOSS-Image worker draw faster (`diffu_moss.inference`, applied by
`load_pipeline`). Each is off by default, and with all of them off a page is drawn byte for byte as before. A drawing
script and the Ray runner's `RunConfig.settings` reach them by settings alone (diffu-generate and the runner's CLI have
no flags for them yet); another model family refuses them.

| Setting | What it changes | Needs |
|---|---|---|
| `precision="mxfp8"` | The DiT's joint-block projections (q, k, v, o, the gated feed-forward's two matrices: 4.76 B of 5.08 B weights) hold FP8 E4M3 values in blocks of 32, each block sharing one power-of-two scale (OCP MX v1.0), and multiply on the tensor cores through `torch.nn.functional.scaled_mm` (`BlockWise1x32`, `SWIZZLE_32_4_4`, fp32 accumulation). Their weights take ~5.5 GB instead of ~10.2. | A Blackwell GPU: compute capability 10.0 or later (B200 10.0, RTX PRO 6000 12.0, GB10 12.1); refused below that before the model loads |
| `text_encoder_precision="mxfp8"` | The same for the text encoder's decoder-layer projections (6.9 B weights: ~10.1 GB instead of ~16.8; the delta rule's two 32-wide gates, the embeddings, norms and vision tower stay bfloat16) | As above |
| `compile_blocks=True` | The 42 joint blocks compiled once at load (regional `torch.compile`, one full graph, dynamic token count), after any MXFP8 swap, so each layer's input quantization fuses in; the first page pays for the compiling | Any device |
| `aoti_package=<path>.pt2` | The joint blocks run from a package built ahead of time with AOTInductor: nothing is compiled at load. Refused if it was built for another GPU architecture or torch version (before the model loads), or another precision, width or dtype | One package per GPU kind, torch version and precision (`diffu-moss-aoti`); not with `compile_blocks` |

MXFP8 is meant to run with `compile_blocks` or an `aoti_package`: run eagerly, each layer's input quantization is some
fifteen small kernels, which can cost more than the faster matmul saves. The text encoder is never compiled, so its
MXFP8 mostly saves memory (it encodes two prompts a page, the DiT runs two passes a step). With a hand's adapter
(section 3) the DiT's MXFP8, a package and `compile_blocks` rework the joint blocks' layers, which the adapter's layers
wrap: they are refused unless `fuse_loras` is set, and then work on the merged weights.

A package holds one compiled joint block and no weights (each block hands it its own at every call), so it serves all
42 blocks and any DiT of MOSS-Image's widths, a fine-tune's too. For a Ray cluster: build one per GPU kind on a node of
that kind, with the torch version the workers run, put it where every actor can read it, and give the actors
`DrawSettings(..., precision="mxfp8", aoti_package=Path("<package>"))`.

    python -m diffu_moss.aoti --model <MOSS model dir> --precision mxfp8 --out runs/aoti/moss_mxfp8_sm100.pt2

`python -m diffu_moss.bench_inference` measures the nine configurations (bf16, MXFP8 in the DiT, MXFP8 in both; each
eager, compiled and packaged) on a model directory at 1 MP: seconds per denoising step (CUDA events, warm), one
prompt's encoding, peak memory, load and first step, and the cosine of the first velocity and of the final latents to
bf16's at the same noise, one JSON line each (`--images` keeps every page to look at beside bf16's).

    python -m diffu_moss.bench_inference --model <MOSS model dir> \
        --package-bf16 runs/aoti/moss_bf16_sm100.pt2 --package-mxfp8 runs/aoti/moss_mxfp8_sm100.pt2 \
        --out runs/bench/inference.jsonl --images runs/bench/pages

Not yet run on a GPU. cuBLAS's MXFP8 GEMM, the compiled MXFP8 block and an MXFP8 package are tested only by tests that
skip without a Blackwell GPU. Inside an AOTInductor package, `scaled_mm` (torch's `_scaled_mm_v2`) has no C shim and
runs through AOTInductor's proxy executor; if that fails, `torch._scaled_mm` (v1, which has one) takes the same
operands.
> Measured speeds on Blackwell are not in yet: the options are tested on CPU, and the GPU benchmark runs next.
> In a run's `draw.json` (section 6) they are fields like any other draw setting.

## 13. The SGLang Diffusion route, and SGLang inside Ray Data

In plain words: SGLang Diffusion is another engine that can run MOSS-Image, as a scribe can work at another
workshop's desk. Out of the box that desk has no stand for the sample page: SGLang's MOSS-Image (sglang 0.5.23,
sgl-project/sglang#44325) draws from a prompt alone. Our plugin, `diffu_sglang_moss` (`packages/sglang_moss`), is
that stand. It adds the sample-page input exactly as our own pipeline reads it, so the fine-tuned scribe copies the
sample's hand at SGLang's desk too. On CPU with a tiny model, SGLang with the plugin draws the same pixels, bit for
bit, as our pipeline from the same prompt, sample and noise, with a sample page and without one
(`packages/sglang_moss/tests`).

What SGLang does **not** do is the rest of the page pipeline (sections 1 and 5): it does not plan pages, does not
compile a crop plan's caption into MOSS's Layout prompt, does not put the sample on the 16-px grid, does not repair
lines, and does not read lines back or write ALTO/PAGE. A request to SGLang carries a finished prompt and a sample page
already on the grid (both sides multiples of 16 px; an off-grid sample is refused, never resized). Use this route
when you want SGLang's serving and scheduling for the drawing itself; use `diffu-run` (sections 5 to 11) for whole
labelled pages.

### What the plugin changes

Three pieces, loaded into SGLang by `SGLANG_EXTERNAL_MODEL_PACKAGE=diffu_sglang_moss` (SGLang imports the package in
every process it starts; nothing of SGLang is patched or copied):

- **The sample page is a request field.** It is SGLang's usual image input: `image_path` (a path, a URL, or several
  paths) offline, the `image` upload of `POST /v1/images/edits` online. A request with one runs as SGLang's TI2I task,
  one without as T2I.
- **A stage encodes it** after the text encoder: the FLUX.2 VAE's mode, patched 2x2 and normalised by the VAE's
  batch-norm statistics, as `diffu_moss.latents` does. A model whose `transformer/export.json` does not say
  `takes_reference` is refused a sample ("this MOSS-Image model takes no reference image").
- **The DiT reads it** (`DiffuMOSSImageTransformer`, which SGLang loads in place of its own): the sample's tokens
  follow the prompt's, on the rotary reference axis at 1, modulated as at timestep 0, in both passes of guidance.
  Without a sample it runs SGLang's own forward.

Its sampling defaults are ours: the official demo's negative prompt (SGLang's default is empty) and the unconditional
pass skipping block 22 from the first step (SGLang's default starts at 1% of the steps). Not supported with a sample:
sequence parallelism (refused), and SGLang's MOSS-Image paths that its integration itself has not validated
(quantization, LoRA, torch.compile, Cache-DiT). CFG parallelism hands the sample to both branches, untested.

### Setting it up

SGLang runs in its own environment, not the workspace's (it pins diffusers 0.37 and transformers 5.19):

```bash
uv venv --python 3.12 <sglang-env>
printf 'kernels ; sys_platform == "never"\n' > <sglang-env>/overrides.txt   # leave out Hub kernel downloads
uv pip install --python <sglang-env>/bin/python "sglang[diffusion]==0.5.23" "torch==2.14.1" torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
    --overrides <sglang-env>/overrides.txt
uv pip install --python <sglang-env>/bin/python --no-deps -e packages/sglang_moss
```

`kernels` is a dependency of sglang that downloads compute kernels from the Hub at run time; SGLang imports it only
for an FA3 path, Cosmos 3 and a vision helper, none of which MOSS-Image uses, so the override leaves it out. Nothing
here needs `trust_remote_code`: the model folder below names every component by its library class, so the Hub's
`pipeline_moss_image.py` is never fetched or run. Set `HF_HUB_OFFLINE=1` and SGLang reads the folder alone.

### The model folder

SGLang loads a model in the Hub's layout (`model_index.json` and one folder per component). Write one from a MOSS
model directory or straight from a checkpoint:

```bash
diffu-moss-export-hf /models/moss-<dit-name> --out /models/hf-<dit-name> --copy
diffu-moss-export-hf <run>/checkpoints/step_000750 --base /models/MOSS-Image-Base --out /models/hf-<dit-name> --copy
```

The Base snapshot must hold `scheduler/` besides `text_encoder/`, `tokenizer/` and `vae/` (stage it with
`hf download OpenMOSS-Team/MOSS-Image-Base scheduler/scheduler_config.json --revision 85975793...`). The DiT's
`config.json` is rewritten to name `MOSSImageTransformer`, the rotary frequencies `rope.bands` the official
checkpoint stores are added in a 64-byte shard of their own, and `export.json` is kept beside the weights. Without
`--copy` the components and weights are symbolic links to where they are (on one disk, 27 GB not written twice);
with it the folder is whole, to move to object storage or another volume.

### Drawing offline, and serving

```python
from sglang.multimodal_gen import DiffGenerator  # in the SGLang environment, SGLANG_EXTERNAL_MODEL_PACKAGE set

generator = DiffGenerator.from_pretrained(model_path="/models/hf-<dit-name>", num_gpus=1)
result = generator.generate(
    sampling_params_kwargs={
        "prompt": "<the page's prompt>",
        "image_path": "<sample page on the 16-px grid>.png",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "seed": 0,
        "save_output": False,
    }
)
pixels = result.samples  # [3, H, W] in [0, 1]
generator.shutdown()
```

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE=diffu_sglang_moss HF_HUB_OFFLINE=1 \
    sglang serve --model-path /models/hf-<dit-name> --port 30000
curl -s http://localhost:30000/v1/images/edits -F prompt="<the page's prompt>" -F image=@<sample>.png \
    -F size=1024x1024 -F num_inference_steps=50 -F seed=0
```

`sglang generate --model-path ... --prompt ... --image-path ...` draws one image from the command line. SGLang's
MOSS-Image runs its flash-attention backend by default; `--attention-backend torch_sdpa` runs PyTorch's
`scaled_dot_product_attention`, as our pipeline does, which is how to compare the two.

### SGLang inside Ray Data

Ray Data's built-in SGLang processor (`ray.data.llm`'s `SGLangEngineProcessorConfig`) drives SGLang's language-model
engine with chat prompts: it is not a route for an image model. For MOSS-Image, the Ray Data stage is a `map_batches`
callable class that holds SGLang's diffusion engine, one per actor, built once when the actor starts. `DiffGenerator`
starts SGLang's worker process on the actor's GPU (Ray sets `CUDA_VISIBLE_DEVICES` for the actor, and the worker
inherits it), so give the plugin's variable to the actors' environment:

```python
import io

import numpy as np
import ray
import torch
from PIL import Image


class SGLangMossDraw:
    """One SGLang MOSS-Image engine per actor; each row a prompt, a sample page and a seed in, a PNG out."""

    def __init__(self, model: str) -> None:
        from sglang.multimodal_gen import DiffGenerator

        self.generator = DiffGenerator.from_pretrained(model_path=model, num_gpus=1)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list]:
        images = []
        for prompt, sample, seed, width, height in zip(
            batch["prompt"], batch["sample"], batch["seed"], batch["width"], batch["height"], strict=True
        ):
            result = self.generator.generate(sampling_params_kwargs={
                "prompt": str(prompt), "image_path": str(sample), "seed": int(seed),
                "width": int(width), "height": int(height), "num_inference_steps": 50, "save_output": False,
            })
            pixels = (torch.as_tensor(result.samples).float().cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            images.append(buffer.getvalue())
        return {"id": list(batch["id"]), "image": images}


ray.init(runtime_env={"env_vars": {"SGLANG_EXTERNAL_MODEL_PACKAGE": "diffu_sglang_moss", "HF_HUB_OFFLINE": "1"}})
pages = ray.data.read_json("<rows>.jsonl")  # id, prompt, sample (a path every node can read), seed, width, height
drawn = pages.map_batches(
    SGLangMossDraw, fn_constructor_kwargs={"model": "/models/hf-<dit-name>"},
    batch_size=4, num_gpus=1, concurrency=<GPUs>,
)
drawn.write_parquet("<out>")  # or lance_ray.write_lance, as diffu-run's sink does
```

The workers' image is the SGLang environment above (with the plugin installed), not diffu-run's: the two pin
different diffusers and transformers. Each actor draws its rows one `generate` call at a time, so throughput comes
from the number of actors. The pages it draws carry no labels and no read-back: the
planning before it and the judging after it stay with `packages/page`.

### Comparing SGLang with our pipeline on a GPU

`packages/sglang_moss/tests/compare_draw.py` draws one page with both from the same model, prompt, sample and noise,
one after the other (the two models are never in memory together), with PyTorch's attention in both, and writes
`compare.json` (identical or not, largest pixel error, PSNR) and the two images side by side to look at. On a GPU the
two need not be bit for bit: on a GPU SGLang loads its own Qwen3.5 text encoder (on CPU it falls back to
transformers', which ours runs), and its fused kernels take over some of the DiT's elementwise operations.

Measured on the DGX Spark (GB10), 2 Nov 2026, with `Gabriel/diffu-moss-r6b` (its first version, revision `726b9fe`) on a
held-out court-record page at 784 x 1280 px, 50 steps, with its sample page: the two pages differ by at most 0.00054 in
a pixel (on 0-1, about a seventh of one 8-bit level; PSNR 88.7 dB) and look the same; not bit for bit, as expected.
SGLang drew at 2.76 s a step (eager, bfloat16), its memory peaking at 29 GB.

**Ray Serve and vLLM-Omni.** Ray Serve LLM's SGLang server serves chat and completion models over an OpenAI-compatible
API, not image generation; it is the route if a reading model (section 14) is ever served online. vLLM-Omni serves
MOSS-Image text-to-image only.

## 14. Next step: a reading model as a judge in the same pipeline

Planned, not built. The judge today is TrOCR, line by line, inside each draw actor. The next judge reads each whole
drawn page with a vision-language reading model and compares what it reads with the page's ALTO/PAGE labels, as its
own Ray Data stage after drawing, with its own GPU pool: Ray Data LLM's batch processor (vLLM) with structured output,
as in Ray's "Batch inference with structural outputs" example. Its shape, for whoever builds it:

```python
from pydantic import BaseModel
from ray.data.llm import build_processor, vLLMEngineProcessorConfig


class PageReading(BaseModel):
    lines: list[str]  # the page's lines as the model reads them, in reading order


reader = build_processor(
    vLLMEngineProcessorConfig(model_source="<reading model>", concurrency=<GPUs for reading>, batch_size=<n>),
    preprocess=lambda row: {  # the drawn page and a prompt asking for its lines as JSON
        "messages": [{"role": "user", "content": [<the page image>, "Read every line of this page."]}],
        "sampling_params": {"structured_outputs": {"json": PageReading.model_json_schema()}, "temperature": 0},
    },
    postprocess=lambda row: {"id": row["id"], "reading": row["generated_text"]},
)
```

A further stage would compare `reading` with the labels' lines (character error rate per line, numbers read
otherwise) and flag the pages to drop or redraw; the result goes into the table as new columns
(`lance_ray.add_columns`), without rewriting the images.

## 15. Troubleshooting

- **A page's error is `OutOfMemoryError`.** The GPU ran out of memory on that page: run fewer actors per GPU
  (`--actors-per-gpu 1`), draw smaller pages (`--page-megapixels`), and check what else uses the card (`nvidia-smi`).
  The other pages go on; a rerun draws the failed ones again.
- **Actors are killed while loading, or the pod is OOMKilled.** The pod's memory limit is too small for the weights
  passing through host memory (section 8).
- **Attention.** The models call PyTorch's `scaled_dot_product_attention`; PyTorch picks a kernel per call (flash,
  cuDNN, memory-efficient, or the slow math one). Which one runs shows in a profiler trace's kernel names. To rule one
  in or out, `--attention flash|cudnn|efficient|math` allows that kernel alone (a page it cannot take fails with "No
  viable backend for scaled_dot_product_attention"); `auto` is the default. No attention kernels are downloaded at run
  time (no `kernels` package, no hub kernels).
- **"names component folders that are not on this machine".** The model directory was written where the volume is
  mounted elsewhere: write it again on the cluster (section 3).
- **"the adapter 'hand' does not fit this DiT".** The `--lora` folder is no adapter of MOSS-Image's DiT (a FLUX.3
  LoRA, a DiT of other widths): its layers are named or shaped otherwise. A hand's adapter is of Base's own DiT
  (section 3).
- **"precision only with the LoRAs fused".** MXFP8, a package or `compile_blocks` with an adapter: set
  `"fuse_loras": true` in `draw.json`.
- **"the HTR model ... is neither a local directory nor in the Hugging Face cache".** Stage the judge and pass
  `--htr-model <dir> --htr-revision ""`.
- **"N templates or reference images are not here".** The inputs volume is not mounted, or not at the paths the run
  file names, on the head; mount it at the same path on every pod.
- **The run waits and no page is drawn.** `kubectl get pods`: GPU workers still pending (no free GPUs, a missing
  toleration or node selector), or still pulling the image. In the Ray dashboard, Ray Data waits for its actor pool.
  If the pool is up but pages are never planned, `--cpus-per-actor` leaves the workers no CPU to plan on.
- **Permission denied reading weights.** Make the volume readable by UID 10001 (section 3).
- **The table is written slowly or has very many files.** Raise `--commit-every`; compact after the run.
- **The draw setting `compile_blocks`.** It compiles with Triton, which needs a C compiler the image leaves out; it is
  not measured on the cluster yet (section 12).
- **The dashboard or job API needs a token.** With KubeRay 1.6 or newer you can turn on Ray's token authentication
  (`authOptions`, commented in the manifest); KubeRay then hands the token to every pod and the job submitter.

## 16. Checklist

- [ ] KubeRay operator 1.6+ installed; GPU nodes advertise `nvidia.com/gpu`; driver 580+.
- [ ] Image built for the nodes' architecture and pushed: `<registry>/diffu-inference:<tag>`.
- [ ] Models volume holds `MOSS-Image-Base/{text_encoder,tokenizer,vae}` at revision `85975793...`, the DiT folder
      (with `export.json`), the judge at revision `aa79fcb1...`, all readable by UID 10001.
- [ ] Model directory written on the cluster with `diffu-moss-init`, printing "takes a reference image" (or, for a
      hand's adapter, Base's own DiT and the adapter staged: section 3).
- [ ] Inputs volume holds the templates, sample images, run file and draw settings, mounted at the same path on every
      pod.
- [ ] Run file lines are page jobs with unique ids; samples only for a model that takes them.
- [ ] `rayjob.yaml`: image, claims, node selector and tolerations, worker replicas and GPUs, memory and `/dev/shm`,
      `--gpus` = replicas x GPUs, the run file, `--out` on the outputs volume.
- [ ] A short run first (section 11): look at a few exported pages beside their real scans before a long run.
- [ ] After a stop: rerun the same command into the same `--out`.
