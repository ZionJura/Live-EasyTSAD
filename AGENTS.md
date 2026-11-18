# Live-EasyTSAD collaboration

The primary checkout is on the operator's Mac. Integrate reviewed code changes there on `zzr-developing`. Each Linux host can run its own harness and use Codex for development in its own checkout.

- Read `EasyTSAD/Harness/README.md` before changing harness behavior. Use `scripts/harness doctor` and the documented tests to check changes.
- On a worker, branch from the synced `zzr-developing` into `maintain/<host>/<topic>`. Commit the change with its verification and fetch it back to the Mac for review and integration. Fetch updates to the base branch between runs and use fast-forward merges.
- Give each method reproduction one owning checkout at a time. The harness task database, conversation state and GPU leases are scoped to one checkout. Coordinate assignment before starting the same method on a different host.
- On zx1 and zx2, use the separate `/mnt/home/projects/Live-EasyTSAD-zx1` and `/mnt/home/projects/Live-EasyTSAD-zx2` checkouts. Both can read the shared dataset checkout. Treat datasets as read-only. Keep virtual environments, outputs, task state and result publication local to the owning checkout.
- Keep credentials and machine overrides in private files. The harness accepts an ignored `EasyTSAD/Harness/local_config.toml` with per-section overrides. When `TSAD_HARNESS_CONFIG` selects another file, it supplies the full configuration.
- Check GPU ownership before allocating a device. Treat unrelated workloads as in use. Follow the approved occupancy handoff and restoration procedure before changing either host's GPU occupancy program.
- Commit source, lockfiles, task specifications, source provenance and compact verified summaries. Keep raw scores, checkpoints, session transcripts, credentials and datasets outside Git.
