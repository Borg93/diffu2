"""Diffu model package (diffusers-first).

Only ``Diffu`` — the one symbol consumers import via the package path (``from diffu.model import
Diffu``, plus diffu-studio's ``monkeypatch.setattr("diffu.model.Diffu", …)``) — is re-exported;
everything else follows the root convention: import from the owning submodule.
"""

from .diffu import Diffu

__all__ = ["Diffu"]
