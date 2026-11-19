# V0.2a diagnosis and repair

The source is the Synergy Budget release at 6456b680f379c197c80cd80845b330414a99e104. Run from this worktree root in its pinned Windows GPU environment after `git lfs pull`. Source checkpoints, frozen banks and target sets are checked before use. The study uses the development guideline's directory spelling `rsi_v02a`.

H1 asks whether the same target becomes harder when hand force changes and whether full-action control can still solve failures. H2 asks whether matched-state correction using the original bank can produce a native terminal success. H3 compares adaptation and old-body retention under a common physical-interaction budget. H4 checks if a same-rank change and a new column can be saved, reloaded and controlled.

The existing repository already has task contracts, per-checkpoint hashes, immutable Reach banks and explicitly tracked LFS inputs. This study adds separate protocol, collision-safe attempt records, simulation accounting and a new study index. Historical experiment directories are read-only inputs. Search candidates and PPO runs use distinct random streams. The official phase order starts with `python -m experiments.rsi_v02a.screen`. Repeated invocations reuse completed stages; a new attempt receives a different ID. Interrupted PPO training starts a fresh attempt and keeps prior records.
