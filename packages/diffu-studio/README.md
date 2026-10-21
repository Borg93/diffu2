# diffu-studio

A Gradio app on the page pipeline (`diffu_page.pipeline.page_blocks`): upload a PAGE or ALTO template, give it a text
(or keep its own) and a style (archive, years, paper, ink), and every crop of the page is drawn, read back and
labelled; the crops and their ALTO labels download as one zip. The pipeline is loaded once, at start.

    diffu-studio --model <model dir> --lora v2=<ckpt>/model --out runs/studio

On the DGX Spark it is a GPU job like any other: start it through `jobs/spark_guard.sh`.
