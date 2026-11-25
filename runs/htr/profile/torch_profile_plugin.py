"""An ms-swift plugin (--external_plugins) registering the callback ``torch_profile``: torch.profiler over training
steps 4 and 5 (after warm-up), its Chrome trace and a kernel table written to $PROFILE_DIR on rank 0."""

import os
from pathlib import Path

import torch
from swift.callbacks.base import TrainerCallback
from swift.callbacks.mapping import callbacks_map

FIRST, LAST = 4, 5


class TorchProfileCallback(TrainerCallback):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.profiler = None
        self.out = Path(os.environ.get("PROFILE_DIR", "profile"))

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step + 1 == FIRST and int(os.environ.get("RANK", "0")) == 0:
            self.out.mkdir(parents=True, exist_ok=True)
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            self.profiler.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler is not None and state.global_step == LAST:
            torch.cuda.synchronize()
            self.profiler.__exit__(None, None, None)
            self.profiler.export_chrome_trace(str(self.out / "trace.json"))
            table = self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=60)
            (self.out / "kernels.txt").write_text(table)
            self.profiler = None


callbacks_map["torch_profile"] = TorchProfileCallback
