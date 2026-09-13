# Independent audit GO — claimed Direct failure -> recompute

Reviewer: /root/audit_resume_controller (Parfit), 2026-09-09.
Runtime D SHA256: 5b9e650b4608e542c59f1d41b150f942d6c6e22fc97b219727eadd600cb6c542
CPU regression: 289 passed, 19.53s; explicit integration PYTHONPATH.

GO for BrowseComp Qwen3-8B TP1 4P:4D c512, 300+1200 seconds.
Only fast-tool failure policy changes; old claim/DMA fences remain.
Unique ownership, P2D Direct/Host releases, D2P Host release, background
progress, TP rank0 ownership and explicit recompute conservation reviewed.
New regression covers retained timestamp after marker cleanup, late-tool
nonclassification, sent terminal/inflight, unstarted return, flag off,
route retry and TP group/fault regressions.
Truly missing/expired arrival evidence remains conservative Slow and must
be reported separately; no fabricated fast/slow classification.
Native HiCache/Mooncake off; aligned .80/.80/.80/.60 Decode and all P .80.
