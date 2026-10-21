# diffu-page

Archive pages as data, CPU only: a real PAGE or ALTO page read as its lines, new text refilled into it, the
training-scale crops it is cut into, each crop's FLUX.3 layout prompt, PAGE/ALTO labels, and the builders of the
training and evaluation sets (`python -m diffu_page.data.build_crops`, `pick_eval`, `steer_set`, `style_pairs`).
