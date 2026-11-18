"""Builds index.html, the verdicts page (claude.ai/artifact/98423D2V376HLajHfbhpS2): every curation experiment of
docs/CURATION.md (E1 to E47) with its verdict, why it was run, what it found and what it taught us; five results as
bar charts; and the two archify diagrams as snapshots (snapshots.sh makes them and the interactive pages).

Publish index.html with the Artifact tool under the same URL. The texts and numbers below are copied from the
register by hand: change them here when a row of the register changes.

    bash snapshots.sh && python3 build.py "17 November 2026, 20:15" "step 3,620 of 4,374, 83 %" "1.75 to 1.79"
"""
import base64
import html
import sys
from pathlib import Path

HERE = Path(__file__).parent
STATE, PROGRESS, DEV_LOSS = sys.argv[1], sys.argv[2], sys.argv[3]

VERDICTS = {  # key: (label, icon, what it means)
    "worked": ("Worked", "i-check", "the method does its job and is in use, or ready to use"),
    "measured": ("Measured", "i-diamond", "a measurement that settled a question"),
    "partly": ("Partly", "i-half", "good for one job and not another, or a gain too small to be sure of"),
    "failed": ("Did not work", "i-cross", "dropped"),
    "running": ("Running", "i-play", "started, no result yet"),
    "queued": ("Queued", "i-ring", "agreed, not started"),
}

THEMES = [
    ("labels", "What the old labels are worth"),
    ("reader", "Training the reader"),
    ("cut", "Placing the cut on the fold"),
    ("describe", "Describing each scan with a vision-language model"),
    ("decide", "Deciding which scans to cut"),
    ("embed", "Embeddings, clusters and maps"),
    ("tables", "Tables"),
    ("checks", "Checks on the page and its labels"),
    ("table", "The table underneath"),
]

# id, theme, verdict, short name (board), title, [(heading, text), ...]
E = [
    (1, "labels", "measured", "Old labels as a reader", "The old labels scored as a page reader", [
        ("Why we did it", "Before training on the old YOLO and TrOCR labels we needed to know how good they are, on the same test pages every reader is scored on."),
        ("What we found", "10.7 % character error on 207 test pages that also lie in the pools. These are not the 200 pages the readers are scored on: only 11 are shared, and on those the old labels read at 11.2 %, Nemotron Parse at 7.6 %, LightOnOCR-3 at 3.4 %."),
        ("What we learned", "Our readers already read better than the labels they would learn from. The labels can only help as pretraining, with ground truth after."),
    ]),
    (2, "labels", "measured", "Calibration against truth", "Calibration: the old label against ground truth, line by line", [
        ("Why we did it", "To learn what kind of errors the labels hold and whether the old pipeline's confidence can filter them. 3,651 scans carry both an old label and a transcriber's truth."),
        ("What we found", "Error on matched lines: Svea 7.4 % (259 scans), andra arkiv 4.8 % (3,392 scans). 40 to 45 % of lines are exactly right; 1.4 to 2.7 % of true lines are missed. Confidence ranks pages by error (rank correlation −0.60 to −0.76)."),
        ("What we learned", "Confidence is blind to writing the pipeline never found, so it is one feature and never the filter alone. About 30 % of the differences are convention (case, punctuation, spelling), not misreading."),
    ]),
    (12, "labels", "partly", "Reader against old labels", "Our reader against the old labels on pool pages", [
        ("Why we did it", "Where our best reader and the old label disagree strongly, one of them is wrong. That could filter bad labels without any ground truth."),
        ("What we found", "Median disagreement 12 % (Svea) and 11 % (andra arkiv) of characters; 10 % and 8 % of pages disagree by more than half. Of 24 such pages looked at, at least 6 were bad cuts and not bad labels."),
        ("What we learned", "A disagreement above half marks a page not to train on as it is. The readings were made on pages cut the old way and on another sample: only 1,881 of the 139,937 trusted pages have one. The read is paused for training and has to be redone on the trusted pages."),
    ]),
    (3, "reader", "partly", "Unfiltered pretraining", "Control: pretrain on unfiltered Svea pool pages, then fine-tune", [
        ("Why we did it", "The cheapest test of the whole idea: do the old labels help at all, before any curation? 53,490 random Svea pages with their labels as they are, one epoch, then the usual fine-tune on ground truth."),
        ("What we found", "7.18 % character error against 7.55 % without pretraining. Better on 123 test pages, worse on 52. The 95 % interval of the gain is −0.13 to +0.84 points."),
        ("What we learned", "A small gain, not proven on 200 pages, and all of it on the pools' own archives: Svea hovrätt pages 8.63 to 6.25 %, andra arkiv's collections 6.09 to 5.36 %, the other collections unchanged."),
    ]),
    (4, "reader", "worked", "Two more GT epochs", "Two more epochs on the ground truth", [
        ("Why we did it", "The reader's dev loss was still falling when training ended, so more passes over the ground truth might help more than any pool data."),
        ("What we found", "7.55 to 6.73 % character error: the best reader so far, for about twice the GPU time."),
        ("What we learned", "Its dev loss was worse (1.167 against 1.155) while its test score was better. Dev loss is a poor guide here. Score the test pages."),
    ]),
    (22, "reader", "measured", "Readers by kind of page", "Readers scored by kind of page", [
        ("Why we did it", "One overall score hides where a reader fails. Once every ground-truth scan had its kind from the vision-language model, the 200 test pages could be split by kind."),
        ("What we found", "Nemotron Parse: running text 5.73 % (174 pages), tables 14.08 % (26 pages). LightOnOCR-3: 5.24 % and 30.82 %. Ten pages carry 44 % of all errors. Two test pages have a ground truth that covers only part of the page; without them the scores are 5.45 % and 7.07 %."),
        ("What we learned", "Tables are the weak kind and the next thing to train for. Some ground truth is partial: when two readers agree with each other and both read far more than the truth holds, the truth is the problem."),
    ]),
    (42, "tables", "measured", "Table pages looked at", "Table pages: the truth, the reader and the old labels", [
        ("Why we did it", "Tables read at 14 % against 6 % for running text. Before training for them we needed to know what a table is in our data, and whether the pools' table pages are worth learning from."),
        ("What we found", "Only 201 of 49,491 ground-truth training pages carry a table with rows and cells. The 26 table test pages are all transcribed as plain lines, and they hold 25 % of all errors. Old labels against truth on Svea tables: 19 % error on matched lines, a wrong number in 46 % of lines, 35 % of labelled lines matching no true line. On andra arkiv tables: 7 % error."),
        ("What we learned", "The reader reads our tables as lines and has seen almost no table structure. Part of the table error is reading order, numbers and dot leaders, and one of four pages looked at was no table at all. The pools' table labels are a poor teacher, Svea's most of all, so pool tables get no extra weight."),
    ]),
    (43, "reader", "queued", "YOLO + TrOCR on the 200 pages", "The old pipeline as one reader on the same 200 test pages", [
        ("Why it is planned", "The comparison that matters is on whole pages: YOLO and TrOCR run as one pipeline on the same 200 test pages, scored on character and word error as Nemotron Parse is."),
        ("Where it is", "Not started. The earlier 10.7 % (E1) was measured on 207 other pages, 11 of them shared, and is no comparison."),
        ("What it needs", "The old pipeline installed in an environment of its own and a GPU for a short run, after the training."),
    ]),
    (44, "reader", "measured", "Why the gain is small", "Why the pretraining helps so little", [
        ("Why we did it", "0.63 points from 139,937 extra pages is less than expected. Three checks on the stored readings and training curves."),
        ("What we found", "The pools are 72 to 88 % greyscale microfilm; the test pages are three-quarters colour. Greyscale test pages gained 17 % of their error (6.99 to 5.78 %), colour pages 5 % (7.76 to 7.35 %). The head start from pretraining shrinks during the fine-tune: the dev loss gap is 0.31 at step 250, 0.11 at step 1,000 and 0.02 at the end. 4,374 steps on pool pages bring the ground-truth dev loss to 1.78; 250 steps on ground truth alone reach 1.73."),
        ("What we learned", "The pools do not look like what the test asks for, the ground truth teaches the same things given time, and the pool's labels are a weak teacher. About 580,000 pool scans are colour; a set chosen to look like the target is the next thing to test, and the scan's saturation is now measured for every scan with the vectors."),
    ]),
    (38, "reader", "running", "Trusted-cut pretraining", "Pretrain on pages with a trusted cut, then fine-tune", [
        ("Why we are doing it", "This is the test the curation leads to: does pretraining on pool pages with a trusted cut beat the same training without it (7.55 %), the unfiltered control (7.18 %) and the best reader (6.73 %)?"),
        ("What we found so far", "One epoch on 139,937 pool pages, then 2 epochs on ground truth: 6.92 % character error, 18.61 % word error. Against the same recipe without pool pages that is a gain of 0.63 points (95 % interval +0.14 to +1.16), better on 123 pages and worse on 58. Against the unfiltered control the gain is 0.26 points and not shown. Against 4 epochs on ground truth alone it is not ahead."),
        ("What we learned so far", f"Pool pages help, most on the pools' own archives: 4.2 to 2.8 % on andra arkiv's volumes, 8.6 to 5.8 % on Svea's, 7.9 to 7.5 % elsewhere. Almost tripling the pages gave little, which points at the labels and not the amount. The second fine-tune, 2 more epochs, is at {PROGRESS} and ends in the evening of 18 November."),
    ]),
    (5, "cut", "failed", "Gutter network", "A small gutter network on thumbnails", [
        ("Why we did it", "Cutting by the old labels' line positions goes wrong on about 870,000 pages. A tiny network (0.2 M parameters) on a 384 × 256 thumbnail was cheap enough to run over all 3.15 million scans."),
        ("What we found", "It finds that a scan is an opening (99.3 % of held-out openings). But against folds marked by hand its cut is off by more than 1 % of the width on 36 % of ordinary openings. Cutting at the middle of the scan is better (23 %). It also says gutter for 32 of 33 scans that must not be cut."),
        ("What we learned", "It was taught the middle of the gap between the two pages' labelled text, which is not the fold, and it answers in steps of 1 % of the width. It was not scored against a marked fold until E31. The page table cut with it is not used."),
    ]),
    (13, "cut", "measured", "Looking at the image cut", "The image cut looked at by group, as thumbnails and at full size", [
        ("Why we did it", "Sample and verify before re-cutting 5.4 million pages with the gutter network."),
        ("What we found", "One bug fixed: 17,082 Svea scans the network was unsure of had been kept whole. On thumbnails most groups looked right, 12 of 12. At full resolution the cut lies inside a label box in 2.2 % of scans, and cuts through text were seen."),
        ("What we learned", "A cut 1 to 2 % of the width off looks right on a thumbnail. Cuts have to be judged at full resolution."),
    ]),
    (18, "cut", "partly", "Placer near the network", "Placing the cut at full resolution, near the network's answer", [
        ("Why we did it", "The network gets near the fold. A second step on the full-resolution image should put the cut on it."),
        ("What we found", "The column with the least writing was wrong: that is the inner margin beside the fold. The column that is dark down its whole height is the fold. Judged blind on 40 scans it beat the network 14 times, lost 2 and tied 24. It found a fold in only 68 % of scans."),
        ("What we learned", "The fold is a dark line. The 68 % was low because the search band was 3 % either side of a network answer that is itself further off; E31 fixed that. The old label boxes cannot judge a cut to half a percent of the width."),
    ]),
    (31, "cut", "worked", "Folds marked by hand", "Folds marked by hand: the yardstick", [
        ("Why we did it", "Every cutter had been judged by a stand-in: label boxes, thumbnails, another model. 200 scans of hard kinds were marked by click on the fold, in 17 minutes."),
        ("What we found", "Off by more than 1 % of the width: gutter network 60 %, the middle of the scan 37 %, the placer searching 6 % either side of the middle 8 %. Where the placer finds a fold line (87 % of scans) it is off on 2 %."),
        ("What we learned", "Search around the middle of the scan and not around the network's answer: nine marked folds in ten lie between 0.48 and 0.52 of the width. A cut can be trusted where the placer finds a fold line."),
    ]),
    (33, "cut", "worked", "150 random scans marked", "The same on 150 scans drawn at random", [
        ("Why we did it", "The 200 were hard kinds on purpose. Rates that hold for the pools need a random sample."),
        ("What we found", "Placer: 3 % of folds off by more than 1 %, none by more than 2 %. Where it finds a fold line (133 of 142 scans) one scan is off. Gutter network: 36 %. A first trusted-cut rule keeps 128 of 150 scans with 2 wrong."),
        ("What we learned", "The placer is good enough to build a training set on. The sample held only 8 scans that must not be cut, too few to show how often the first form misses them. E34 showed it."),
    ]),
    (27, "cut", "partly", "VL model as cut judge", "Can the vision-language model judge or place the cut?", [
        ("Why we did it", "If the model could judge a cut on a zoomed strip, it could score cutters on thousands of scans without hand marks."),
        ("What we found", "As a judge between two drawn cuts it agrees with the eye on 14 of 16 decided scans. As a locator its median error is 0.6 % of the width, no better than the gutter network."),
        ("What we learned", "The model can compare two cuts. It cannot say where the fold is. Hand marks turned out quicker and more certain as the yardstick."),
    ]),
    (26, "cut", "queued", "Trained cut placer", "A trained cut placer", [
        ("Why it is planned", "For openings where no dark fold line shows (wide gutters, loose sheets), a model on a high-resolution strip could learn the fold from the placer's confident answers."),
        ("Where it is", "Labels are collected from 19,660 scans; nothing is trained. 225 more scans are drawn for marking, the placer's failures among them."),
        ("Why it waits", "The simple placer covers about nine openings in ten, so this is now for the rest."),
    ]),
    (14, "describe", "worked", "VL form: pilot", "A vision-language model fills a form about each scan: pilot", [
        ("Why we did it", "Describe each scan before it is cut, so the cut and the filter can depend on what the scan is. Nemotron 3.5 Super VL answers a fixed form: kind, layout, written side, table across the fold, fold position."),
        ("What we found", "267 of 268 scans answered, at 2.7 scans a second. 11 of 12 looked at were right. Its fold position is a guess: 0.50 for 149 of 198 scans."),
        ("What we learned", "The model can say what a scan is. It cannot say where things are. Ask for JSON in the prompt: a schema forced by the server gave broken answers, both pages empty for nearly every opening."),
    ]),
    (15, "describe", "worked", "VL form: 45,000 scans", "The form on 45,000 scans", [
        ("Why we did it", "A sample large enough to see the rare kinds and to train a classifier that spreads the answers to every scan."),
        ("What we found", "44,932 answered. Both pages written 78 %, a page empty 18 %, single page 3.5 %, not a page 0.4 %; not handwriting 1.7 %."),
        ("What we learned", "At this speed all 3.15 million scans would take 13 days. So the plan is a described sample, and a classifier on embeddings for the rest."),
    ]),
    (19, "describe", "worked", "Define every option", "Does a sharper question fix the table answer?", [
        ("Why we did it", "The first form's table or ledger answer was right for only 8 of 18 scans looked at. Dense prose was called a table."),
        ("What we found", "With each option defined in one line the model is right on 33 to 35 of 35 scans. Thinking mode adds nothing."),
        ("What we learned", "The form was at fault and the model was not. Every option in a question gets a line saying what it means."),
    ]),
    (20, "describe", "worked", "Second pass, defined", "A second pass with the defined questions", [
        ("Why we did it", "To replace the first form's unreliable answers (table or running text, turned writing) on all 45,000 scans."),
        ("What we found", "Tables are 11 % of scans: Svea 17 %, ground truth 11 %, andra arkiv 5 %. Of the first form's 7,911 tables only 4,554 are tables. Some turned writing on 1.2 % of scans."),
        ("What we learned", "Which answers of the first form stand (not a page, single page, written side) and which do not (table, table across the fold, turned writing)."),
    ]),
    (24, "describe", "worked", "60,000 more scans", "60,000 more pool scans described", [
        ("Why we did it", "More examples of the rare kinds, and a larger pool to select training pages from."),
        ("What we found", "130,666 scans are now described and embedded, 100,083 of them from the pools. The trusted set doubled: 77,258 pool scans pass the rule."),
        ("What we learned", "Describing and embedding now limit the size of the training set: 130,666 of 3.15 million scans."),
    ]),
    (41, "describe", "measured", "Is the VL model needed everywhere", "Is the vision-language model needed on every scan?", [
        ("Why we did it", "The model would need over a month for 3 million scans. Its answers were compared with the classifier's on the 15,110 marked scans, and a small classifier was taught its table answer."),
        ("What we found", "Cut or not: the model's two answers find 65 % of the scans not to cut, and 4.1 % of what they would cut is of the wrong kind. The classifier finds 96 % and leaves 0.5 %. Table or running text: a small classifier agrees with the model on 97 % of 44,269 scans and finds 96 % of its tables."),
        ("What we learned", "The model is the teacher on a sample: it describes scans and names the tables. Classifiers trained on its answers and on the hand marks carry both to every scan. For cut-or-not the hand marks are the better teacher."),
    ]),
    (29, "describe", "queued", "Data Designer harness", "NeMo Data Designer as the harness for the vision-language model", [
        ("Why it is planned", "A ready-made harness for judging and describing: named options that are each described, parallel requests and traces, in place of our own scripts."),
        ("Where it is", "Not started."),
        ("First test", "Rebuild the cut judge of E27 in it and check that its forced output format does not break this model's answers, as a forced schema did in E14."),
    ]),
    (32, "decide", "failed", "Few-shot examples", "Few-shot examples for the cut-or-not question", [
        ("Why we did it", "The hope was that marked scans could serve as reference examples for the vision-language model. Tested on the 200 marked scans, each left out of its own examples."),
        ("What we found", "86 % right with definitions only, 88 % with six marked examples: 13 scans gained, 9 lost. Examples make the model slower to say do not cut: it finds 22 of 33 where it found 28."),
        ("What we learned", "Examples do not help this model on this question. Defining the three answers does: it holds back 16 good openings of 167 where the first form held back 32. The marks serve better as training data for a classifier (E36)."),
    ]),
    (34, "decide", "worked", "Bulk marks in a gallery", "Marking cut-or-not in bulk, in a gallery", [
        ("Why we did it", "A few hundred marks could not show how often the model is wrong on rare kinds. Every scan on the embedding map was marked by hand, cluster by cluster."),
        ("What we found", "14,985 scans: cut 13,454, odd layout 428, do not cut 419, single page 380, not a page 304. So 10 % must not be cut, and the first form calls 58 % of those an opening to cut."),
        ("What we learned", "Marking by cluster is fast because scans that look alike sit together. The page had to be rebuilt twice to be usable: 120 scans a screen, select first and then answer, the state kept in the browser."),
    ]),
    (35, "decide", "partly", "Marks by map region", "Marking by region of the embedding map", [
        ("Why we did it", "After marimo's bulk-labelling example: drag a box over a region of the map and mark all its scans at once."),
        ("What we found", "962 scans were marked this way. The clusters do separate the question: 4 of 50 clusters hold 68 % of the scans the first form calls single or table across the fold, and 36 clusters hold under 2 %."),
        ("What we learned", "It works. The gallery by cluster was the quicker page for marking everything."),
    ]),
    (36, "decide", "worked", "Classifier on the marks", "A classifier on the embeddings, trained on the marks", [
        ("Why we did it", "To spread the hand marks to every scan. A logistic head on the frozen Qwen3-VL-Embedding 8B vector answers cut, do not cut, single page, not a page or odd layout."),
        ("What we found", "With whole volumes held out: 94.1 % right against the first form's 85.5 %. It finds 95.4 % of the scans not to cut and holds back 4.8 % of the others. On 350 scans marked one at a time, outside its training: 39 of 41 found, 21 of 309 held back."),
        ("What we learned", "With thousands of marks a frozen embedder and a logistic head are enough. No embedder was fine-tuned; that stays an option for the odd-layout class, its weakest."),
    ]),
    (37, "decide", "worked", "Trusted-cut rule", "The trusted-cut rule applied to the pools", [
        ("Why we did it", "To select the first curated training set. A scan is cut only when independent checks agree."),
        ("What we found", "The classifier refitted on all 15,288 marks: 95.4 % right, finds 95.2 %, holds back 3.7 %. The rule keeps 77,258 of 100,083 described pool scans (77 %). Of 48 drawn at random and looked at with the fold drawn, 47 are cut on the fold."),
        ("What we learned", "The rule: the form says opening with no table across the fold, the defined question says cut, the classifier says cut, and the placer finds a fold line."),
    ]),
    (39, "decide", "worked", "Small-model classifier", "The cut classifier on a small vision model's vectors", [
        ("Why we did it", "Only 100,083 of 3.15 million pool scans had been through the checks, because the vision-language model and the 8B embedder are slow. Can the classifier run on vectors from a model a twentieth the size?"),
        ("What we found", "On 15,110 marked scans, whole volumes held out. Qwen 8B: 95.5 % right, finds 95.3 % of the scans not to cut. DINOv3 alone: 94.0 % right, finds 96.0 %, and holds back 5.6 % of good openings where Qwen holds back 3.9 %. Of the scans either would cut, about 0.5 % are not to be cut."),
        ("What we learned", "The small vision models are enough for cut-or-not. They embedded 45,000 scans at 35 a second on one GPU, so all 3.15 million pool scans take about 13 hours on two GPUs, and the vision-language model is not needed at that scale."),
    ]),
    (40, "decide", "queued", "Whole pools through the rule", "The whole pools through the cut rule", [
        ("Why it is planned", "Only 3 % of the pools has been through the checks. With a small model's vectors and the fold placer, every scan can be."),
        ("Where it is", "Queued to start by itself when the training ends: every pool scan with labelled lines is embedded on both GPUs, about 13 hours, and every fold is placed on the CPU. Checked on 40 scans: the new vectors match the ones the classifier was trained on."),
        ("What it should give", "About 2.4 million scans to cut, 4.4 million pages. One epoch of pretraining covers about 280,000 pages a day, so the size of the next set is a choice of GPU days."),
    ]),
    (7, "embed", "worked", "DINOv3 + EVoC clusters", "DINOv3 page embeddings clustered with EVoC", [
        ("Why we did it", "To map what the pools hold without any labels, and to find what is not handwriting."),
        ("What we found", "On 3,007 pages, every cluster looked at: reel targets and title cards, placeholder sheets, blank openings, ruled registers without labels and one series' hand each come out as their own cluster. About 8 % of pages are not handwriting."),
        ("What we learned", "Embeddings separate the kinds of page well enough to drop what is not handwriting by example."),
    ]),
    (8, "embed", "measured", "Embedders compared", "Which embedder groups pages that belong together", [
        ("Why we did it", "To choose the embedder for the map: TIPSv2, DINOv3 and EmbeddingGemma 2 on 49,998 pages, scored by whether a page's nearest pages come from its own volume."),
        ("What we found", "TIPSv2 0.726, DINOv3 0.653, EmbeddingGemma 2 0.42 to 0.43; chance is 0.03."),
        ("What we learned", "The vision models place a page by how it looks, which tells one volume from another. This yardstick fits only one of two jobs (see E9 and E21)."),
    ]),
    (9, "embed", "measured", "Qwen 2B and 8B", "The same for Qwen3-VL-Embedding 2B and 8B", [
        ("Why we did it", "To see whether a larger image-and-text embedder closes the gap."),
        ("What we found", "8B: 0.514 from the same volume. 2B: 0.381. Both behind DINOv3 and TIPSv2 on this yardstick."),
        ("What we learned", "Two kinds of embedding do two jobs and are not ranked in one table. Patch models (DINOv3, TIPSv2) show where the writing is and how a page looks. One-vector models (Qwen, EmbeddingGemma, Nemotron Embed) carry what a scan is, for clusters and classifiers."),
    ]),
    (16, "embed", "worked", "Head on patch embeddings", "A head on patch embeddings that repeats the model's answers", [
        ("Why we did it", "The vision-language model is too slow for every scan. A head on embeddings could carry its answers to all 3.15 million."),
        ("What we found", "On 44,932 scans with volumes held out: it finds 98.8 % of the scans that are not handwriting, and 74 % of what it flags is right. Single page: recall 0.91. Both pages written: precision 0.993."),
        ("What we learned", "The coarse description spreads well. Writing at the fold does not: 0.760 against 0.742 for always saying the commonest. For cut-or-not this head gave way to the classifier of E36, which learns from hand marks and not from the model's answers."),
    ]),
    (21, "embed", "worked", "Qwen 8B chosen", "Which one-vector embedder gives the best clusters of scans", [
        ("Why we did it", "For the job of saying what a scan is: Qwen3-VL-Embedding 8B, EmbeddingGemma 2 and Nemotron 3.5 Embed 8B on 15,000 scans, judged against the kind of scan."),
        ("What we found", "Neighbours of the same kind, chance 0.20: Qwen 8B 0.75, EmbeddingGemma 2 0.66, Nemotron Embed 0.60. Nemotron Embed fails on single pages: 0.14 against Qwen's 0.61."),
        ("What we learned", "Qwen3-VL-Embedding 8B is the scan embedder. The map, the bulk marking and the classifier all run on it."),
    ]),
    (23, "embed", "failed", "Tuning EVoC", "Tuning EVoC's settings", [
        ("Why we did it", "At about 50 clusters EVoC left half the scans in no cluster."),
        ("What we found", "Noise level, neighbours and smallest cluster size were varied for four embedders. No setting brings the unclustered share under about 40 %, and the agreement with the kinds of scan hardly moves."),
        ("What we learned", "EVoC's settings are not the lever. The unclustered scans have to be assigned afterwards (E25)."),
    ]),
    (25, "embed", "partly", "SCALAR and HiRank", "SCALAR for full coverage, HiRank for oddities", [
        ("Why we did it", "Every scan needs a cluster, for bulk marking and for balancing. And odd scans should be found."),
        ("What we found", "SCALAR leaves 7 % of scans unclustered where EVoC alone left 54 %, with the same agreement with the kinds. HiRank does not find what is not handwriting (AUC 0.18): signs and reel targets are many of a kind, so they are not outliers. 1,465 scans tie at its lowest score."),
        ("What we learned", "SCALAR works, reaches 3.15 million scans, and its 50 clusters carried the bulk marking. HiRank as run finds one-offs at best and needs a longer rank list."),
    ]),
    (45, "tables", "partly", "NVIDIA's table model", "NVIDIA's table-structure model on our tables", [
        ("Why we did it", "Our tables are the reader's weakest kind and the ground truth marks almost none as tables. Nemotron Table Structure v1 finds cells, rows and columns in an image of a table. It was trained on printed enterprise documents, and its card warns that it may not generalise."),
        ("What we found", "A first look by eye on the 26 table test pages, the model run on the whole page with no training on our data. On list-like pages it lays a row on nearly every written line (median 18 rows a page) and finds believable columns on many: item and amount, name and page numbers, two lists side by side. On dark microfilm and a densely ruled register it finds few rows. Cells are patchy."),
        ("What we learned", "Scored on the 228 tables the ground truth marks with rows and cells: it finds 89 % of the columns (88 % of those it finds are right) and 67 % of the rows (77 % right). Name lists and registers go well (rows 92 %, columns 95 % in one collection); tables whose rows are paragraphs of several lines go badly (rows 44 %). Columns are usable as they come, rows on lists, cells not. It has not been trained on our pages, and those 228 tables are labels to do so."),
    ]),
    (46, "tables", "failed", "NVIDIA's page-elements model", "NVIDIA's page-elements model as the table finder", [
        ("Why we did it", "The table model's card says to cut the table out first with Nemotron Page Elements v3, which finds tables, titles, headers and text on a page."),
        ("What we found", "On the 26 table test pages it calls 7 a table, 10 an infographic and the rest text. The table model does as well on the whole page as on the table it cuts out."),
        ("What we learned", "It is no table finder for our pages. Our own small classifier on the scan vectors finds 96 % of the tables the vision-language model names (E41), so that stays the table column."),
    ]),
    (47, "checks", "queued", "Nemotron OCR v2", "Nemotron OCR v2 as a line finder and reading-order model", [
        ("Why it is planned", "It finds text lines and their reading order at about 35 pages a second. That could find lines and check for writing without a label on every pool page."),
        ("Where it is", "Not started. Its languages are English, Chinese, Japanese, Korean and Russian, and no fine-tuning code is given, so its own reading of our hands is expected to be poor."),
        ("First test", "Its line boxes against the ground truth's line outlines and the old pipeline's lines. NeMo Curator now has stages built on it, to read for our pipeline as code."),
    ]),
    (6, "checks", "worked", "Orientation network", "An orientation network", [
        ("Why we did it", "The old pipeline invents fluent labels for writing that is turned. Such pages must be found."),
        ("What we found", "A 0.19 M-parameter network on a 256 × 256 thumbnail: 99.94 % on held-out pages. On 20,000 pool pages 0.17 % are turned with confidence."),
        ("What we learned", "Typed pages fool it, so it runs after what is not handwriting is set aside. It has not yet been run over the pools."),
    ]),
    (10, "checks", "worked", "Writing layer tuned", "The writing layer tuned on real line outlines", [
        ("Why we did it", "A layer that marks where writing is, to find writing the labels skip. Its first score (0.93) turned out to be against boxes and not outlines, so it was redone on the ground truth's polygons."),
        ("What we found", "Embedder, head and resolution all land within 0.01: an overlap of 0.71 to 0.73 out of a possible 0.84. DINOv3 with a linear head is enough."),
        ("What we learned", "The ground truth's real line outlines are in its PAGE files; its ALTO files hold boxes. The 16-pixel patch grid caps the overlap at 0.84."),
    ]),
    (11, "checks", "worked", "Missing labels noticed", "Does the writing layer notice missing labels?", [
        ("Why we did it", "That is its purpose: flag pages where the old pipeline left writing without a label, about 6 % of pages."),
        ("What we found", "With a quarter of a page's lines removed it ranks the page below a complete one with AUC 0.95. With a tenth removed: 0.88."),
        ("What we learned", "It works on misses that were made up. The old pipeline's real misses (a whole margin, a faint hand) are not yet scored, and the layer is not yet a column in the table."),
    ]),
    (17, "checks", "failed", "AnyUp, INSID3, ink mask", "AnyUp, INSID3, an ink mask and NeMo Curator", [
        ("Why we did it", "Four shortcuts: sharper features (AnyUp), finding writing from one example page (INSID3), a plain ink threshold, and NVIDIA's curation toolkit."),
        ("What we found", "AnyUp 0.803 against 0.855 for plain interpolation, worse on 30 of 30 pages. INSID3 0.72 against the layer's 0.85. The ink mask fails on stains and bleed-through. NeMo Curator has nothing for documents or handwriting."),
        ("What we learned", "Resolution is not the bottleneck for finding writing. A trained linear layer beats the methods that need no training."),
    ]),
    (28, "checks", "queued", "Consensus entropy", "Consensus entropy across readers", [
        ("Why it is planned", "Several readers read each page. Where they agree the reading can be trusted, and the most central reading becomes the label (Zhang et al., CVPR 2026). It would also find partial ground truth (E22)."),
        ("Where it is", "Not started."),
        ("Why it waits", "It needs LightOnOCR-3's readings of the calibration scans, which is a GPU job, and the GPUs are training."),
    ]),
    (30, "table", "worked", "One Lance table", "One Lance table with every feature as a column", [
        ("Why we did it", "Results were scattered over JSONL and NumPy files. The aim: Lance tables, JSON columns and joins, built from the Lance docs."),
        ("What we found", "The scan table holds 3,184,889 scans keyed by image. A new feature joins as a column without rewriting rows; an existing column is updated in place. A refresh of 79,000 described scans takes 12 seconds."),
        ("What we learned", "Read the docs in full first: Lance's merge refuses a column that exists and conflicts with every other write, so updates go through update_columns. Some result files remain to be moved in."),
    ]),
]

LESSONS = [
    ("Mark a yardstick by hand before trusting a stand-in", "350 fold clicks and 15,288 bulk marks settled in a day what label boxes, thumbnails and model answers could not.", "E13, E18, E31, E34"),
    ("Judge at full resolution", "Thumbnails hid cuts that were 1 to 2 % off. A sample of bad labels turned out to be partly bad cuts.", "E12, E13"),
    ("Simple image evidence beat a trained network", "The fold is a dark line near the middle of the scan. Searching for it is off on 3 % of openings; the network on 36 %.", "E5, E18, E31, E33"),
    ("The model says what a scan is; it cannot say where the fold is", "Define every option. Ask for JSON in the prompt. Few-shot examples and thinking mode added nothing.", "E14, E19, E27, E32"),
    ("Marks plus embeddings spread a human decision", "A frozen embedder and a logistic head reach 95 % once there are thousands of marks. The same works for the model's own answers, so the slow model describes a sample and small classifiers reach every scan.", "E34 to E37, E39, E41"),
    ("Two kinds of embedding, two jobs", "Patch models show where the writing is. One-vector models carry what a scan is. They are not ranked together.", "E8, E9, E21"),
    ("The old labels are a weak teacher", "They read at 10.7 % where our reader is at 6.7 %. Unfiltered, they gave 0.37 points, on the pools' own archives only.", "E1, E2, E3"),
    ("Score the test pages, and by kind", "Dev loss pointed the wrong way. Tables read at 14 % against 6 % for running text. Two test pages have a partial truth.", "E4, E22"),
    ("Every result is a column in one table", "The trusted-cut rule is one filter over one table. Reading the Lance docs in full changed how columns are joined.", "E30, E37"),
]

ALSO = [
    "A 10-token no-repeat guard when Nemotron Parse reads took its first score from 13.6 % to 7.55 %.",
    "LightOnOCR-3 reads running text a little better (5.2 % against 5.7 %) and tables twice as badly (30.8 % against 14.1 %). Unlimited-OCR 2 scored 63.7 % on whole pages.",
    "Cutting by the old labels' line positions fails three ways: a strip of the facing page on 621,386 pages, 154,874 openings thrown away, and 8,281 near-square openings left whole.",
    "A side job on a training GPU slowed training from 11 to 16–18 seconds a step. Side work now stays off the training GPUs.",
    "A follow-on job queued on a log line never started after a restart. Jobs now queue on the tmux window closing.",
    "The root disk filled once, from another project's Docker volumes, and stopped every job for twenty minutes.",
]

OPEN = [
    ("The second score of E38", "Pretraining followed by 4 epochs on ground truth, against 6.73 %. It comes in the evening of 18 November and says whether pool pages add anything to longer training on ground truth."),
    ("Filtering the labels", "Our reader has to read the 139,937 trusted pages. On the 1,881 that have a reading, about 6 % disagree with the old label by more than half."),
    ("Tables", "6 % of the trusted set and the reader's weakest kind. The pools' labels on tables are poor (E42), so better table reading has to come from ground truth, a fixed reading order, or the double pass on table regions, which is queued."),
    ("Openings without a fold line", "About one in ten. The trained placer (E26) is for these."),
    ("Scale", "The 139,937 pages come from the 100,083 pool scans checked so far, 3 % of the pools. By the same rule the pools hold about 4.4 million pages. With E39 every scan can be embedded in about 13 hours on two GPUs. Training time then sets the size: one epoch covers about 280,000 pages a day."),
    ("Still to look at by eye", "The table test pages, the test pages with one side empty, and a sample from each cluster."),
    ("Partial ground truth", "Found on two test pages. The check over the whole ground truth needs two readers' readings (E28)."),
]

CHARTS = [
    ("Character error on the 200 test pages", "percent of characters wrong; lower is better", 9.0, [
        ("LightOnOCR-3", 8.32, "8.32 %", "2 epochs on ground truth"),
        ("Nemotron Parse, 2 epochs on ground truth", 7.55, "7.55 %", "with the 10-token no-repeat guard"),
        ("the same after unfiltered Svea pretraining", 7.18, "7.18 %", "53,490 pool pages first (E3)"),
        ("the same after trusted-cut pretraining", 6.92, "6.92 %", "139,937 pool pages first (E38); gain 0.63 points, 95 % interval +0.14 to +1.16"),
        ("Nemotron Parse, 4 epochs on ground truth", 6.73, "6.73 %", "the best reader so far (E4)"),
        ("4 epochs after trusted-cut pretraining", None, "18 Nov, evening", "the second fine-tune of E38 is training"),
    ], "Unlimited-OCR 2 read the same pages at 63.7 % and is off this scale. The old YOLO and TrOCR pipeline has no score on these pages yet (E43)."),
    ("What the trusted-cut rule keeps", "described pool scans left after each check", 100083.0, [
        ("Described pool scans", 100083, "100,083", "Svea and andra arkiv, described and embedded"),
        ("The form says opening, no table across the fold", 95485, "95,485", "96,105 openings; 620 of them hold a table across the fold"),
        ("The defined question says cut", 87366, "87,366", "8,119 scans held back"),
        ("The classifier says cut", 80767, "80,767", "6,599 scans held back"),
        ("The placer finds a fold line", 77495, "77,495", "3,272 openings with no fold line"),
        ("Labelled, and outside ground-truth volumes", 75776, "75,776", "237 without labelled lines; 1,482 in volumes that hold ground truth"),
    ], "The 75,776 scans give 139,937 single pages."),
    ("Cuts more than 1 % of the width from the marked fold", "share of 142 openings drawn at random; lower is better", 40.0, [
        ("Gutter network", 36, "36 %", "a small network on thumbnails (E5)"),
        ("The middle of the scan", 23, "23 %", "no method at all: cut at 0.5"),
        ("Fold placer", 3, "3 %", "dark column within 6 % of the middle (E33)"),
    ], "Where the placer finds a fold line, 133 of the 142, one scan is off."),
    ("Scans that must not be cut: share found", "share of the scans marked not to cut; higher is better", 100.0, [
        ("The vision-language model's first form", 42.4, "42 %", "639 of 1,506 on the first 7,418 marks; holds back 0.5 % of good openings"),
        ("Classifier trained on 15,288 marks", 95.2, "95 %", "whole volumes held out; holds back 3.7 % of good openings (E37)"),
    ], "The classifier holds back 3.7 % of the openings that could be cut. The first form holds back 0.5 %."),
    ("Nemotron Parse by kind of page", "percent of characters wrong on the test pages; lower is better", 16.0, [
        ("Running text", 5.73, "5.73 %", "174 test pages"),
        ("Table or ledger", 14.08, "14.08 %", "26 test pages"),
    ], "LightOnOCR-3 on the same pages: 5.24 % and 30.82 %."),
]


def esc(text: str) -> str:
    return html.escape(text, quote=True).replace(" %", " %")


def size(name: str) -> tuple[int, int]:
    header = (HERE / name).read_bytes()[16:24]
    return int.from_bytes(header[:4], "big"), int.from_bytes(header[4:], "big")


def uri(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((HERE / name).read_bytes()).decode()


def pill(verdict: str) -> str:
    label, icon, _ = VERDICTS[verdict]
    return f'<span class="pill v-{verdict}"><svg class="ic" aria-hidden="true"><use href="#{icon}"/></svg>{label}</span>'


counts = {key: sum(1 for row in E if row[2] == key) for key in VERDICTS}
assert sum(counts.values()) == len(E) and sorted(row[0] for row in E) == list(range(1, len(E) + 1)), counts

tiles = "".join(
    f'<div class="tile v-{key}"><b>{counts[key]}</b><span><svg class="ic" aria-hidden="true"><use href="#{icon}"/></svg>{label}</span><small>{esc(meaning)}</small></div>'
    for key, (label, icon, meaning) in VERDICTS.items()
)

board = ""
for key, (label, icon, _) in VERDICTS.items():
    chips = "".join(
        f'<a class="chip" href="#e{n}" data-jump="e{n}"><b>E{n}</b>{esc(short)}</a>'
        for n, _, verdict, short, _, _ in sorted(E) if verdict == key
    )
    board += (f'<div class="row v-{key}"><div class="row-head"><svg class="ic" aria-hidden="true"><use href="#{icon}"/></svg>'
              f'<span>{label}</span><em>{counts[key]}</em></div><div class="chips">{chips}</div></div>')

charts = ['<div class="charts two">', '<div class="charts three">']
for at, (title, unit, top, rows, note) in enumerate(CHARTS):
    body = ""
    for name, value, shown, detail in rows:
        if value is None:
            bar = f'<div class="bar-track"><span class="bar-pending">no score yet: {esc(shown)}</span></div>'
        else:
            bar = f'<div class="bar-track"><div class="bar" style="--w:{value / top:.4f}"></div><span class="bar-val">{esc(shown)}</span></div>'
        body += (f'<div class="bar-row" tabindex="0" data-name="{esc(name)}" data-value="{esc(shown)}" data-detail="{esc(detail)}">'
                 f'<div class="bar-label">{esc(name)}</div>{bar}</div>')
    charts[0 if at < 2 else 1] += f'<div class="chart"><h3>{esc(title)}</h3><p class="unit">{esc(unit)}</p>{body}<p class="chart-note">{esc(note)}</p></div>'

charts = "</div>".join(charts) + "</div>"

lessons = "".join(f'<div class="lesson"><h3>{esc(title)}</h3><p>{esc(text)}</p><span class="refs">{esc(refs)}</span></div>' for title, text, refs in LESSONS)

ledger = ""
for key, name in THEMES:
    rows = [row for row in E if row[1] == key]
    entries = ""
    for n, _, verdict, _, title, parts in rows:
        cols = "".join(f"<div><h5>{esc(head)}</h5><p>{esc(text)}</p></div>" for head, text in parts)
        entries += (f'<article class="exp" id="e{n}" data-verdict="{verdict}" tabindex="-1"><header><span class="eid">E{n}</span>'
                    f"<h4>{esc(title)}</h4>{pill(verdict)}</header><div class=\"cols\">{cols}</div></article>")
    ledger += f'<section class="theme" data-theme-group><h3>{esc(name)} <span>{len(rows)}</span></h3>{entries}</section>'

filters = f'<button type="button" id="filter-all" data-filter="all" aria-pressed="true">All <span>{len(E)}</span></button>' + "".join(
    f'<button type="button" id="filter-{key}" data-filter="{key}" aria-pressed="false" class="v-{key}"><svg class="ic" aria-hidden="true"><use href="#{icon}"/></svg>{label} <span>{counts[key]}</span></button>'
    for key, (label, icon, _) in VERDICTS.items()
)

also = "".join(f"<li>{esc(item)}</li>" for item in ALSO)
open_items = "".join(f"<div><dt>{esc(head)}</dt><dd>{esc(text)}</dd></div>" for head, text in OPEN)

page = (HERE / "template.html").read_text(encoding="utf-8")
for slot, value in {
    "N": str(len(E)), "STATE": esc(STATE), "PROGRESS": esc(PROGRESS), "TILES": tiles, "BOARD": board, "CHARTS": charts, "LESSONS": lessons,
    "LEDGER": ledger, "FILTERS": filters, "ALSO": also, "OPEN": open_items,
    "PIPE_LIGHT": uri("snap_pipeline_light.png"), "PIPE_DARK": uri("snap_pipeline_dark.png"),
    "CUT_LIGHT": uri("snap_cut_light.png"), "CUT_DARK": uri("snap_cut_dark.png"),
    "PIPE_W": str(size("snap_pipeline_light.png")[0]), "PIPE_H": str(size("snap_pipeline_light.png")[1]),
    "CUT_W": str(size("snap_cut_light.png")[0]), "CUT_H": str(size("snap_cut_light.png")[1]),
}.items():
    assert "{{" + slot + "}}" in page, slot
    page = page.replace("{{" + slot + "}}", value)
assert "{{" not in page
(HERE / "index.html").write_text(page, encoding="utf-8")
print(len(page), "bytes;", counts)
