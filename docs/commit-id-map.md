# Commit ID map (history rewrite, 28 Nov 2026)

On 28 Nov 2026 the commit messages on `main` were rewritten to remove a `Co-Authored-By` trailer
naming the AI assistant used in this project (AI assistance is disclosed in the paper). **No file
content changed**: the final tree is identical, and commit dates are preserved. Every commit
therefore has a new ID.

MLflow run tags (`git_commit`) and older notes record the **old** IDs. Use this table to map them.

| Old ID | New ID | Commit |
| --- | --- | --- |
| `8cf43cf` | `2f62c7c` | Project skeleton |
| `20b0f6d` | `bf3ecdd` | phase0 - initial commit |
| `e83c9bc` | `2416260` | Phase 1: Open-Meteo PM2.5 data under DVC, prepare/featurize pipeline, EDA |
| `5c391ab` | `3eb968f` | Let DVC manage data .gitignore files (root rules hid .dvc files from DVC) |
| `cb20211` | `d8167fe` | Phase 1 notes and README update |
| `e5210c3` | `24af350` | Phase 2: walk-forward folds, shared evaluation, naive baselines |
| `bcc009a` | `88c6c66` | Phase 3: SARIMA and direct XGBoost models, log_run, pyfunc wrapper, plots |
| `e28459b` | `10d0fa7` | Update dvc.lock after Phase 3 code changes |
| `7840f75` | `cce1d40` | Phase 3 complete: MVP results, champion registered, docs |
| `2b661ca` | `7f28dba` | Phase 4: LSTM model (direct multi-output, early stopping), optuna dependency |
| `3a5d83e` | `17a2401` | Run LSTM tests in a separate process (torch/xgboost libomp clash) |
| `d433eaa` | `5e0e192` | Phase 4: Optuna sweep tracked in MLflow (nested trial runs, importance plots) |
| `743a341` | `b3ae9a2` | Tuned XGBoost config from Optuna sweep (trial 17, CV RMSE 8.149) |
| `9ece807` | `fb1c727` | Phase 4 complete: LSTM results, Optuna sweep findings, docs |
| `2c0d0df` | `eccf18d` | Phase 4 notes: clean LSTM reruns reproduce original metrics exactly |
| `0e805c6` | `a168254` | Phase 5: promote.py (champion/challenger, baseline, rollback) and ensemble model |
| `11aae55` | `01e0d35` | Phase 5 complete: registry demo (2 rejections, ensemble promoted), docs |
| `16763c4` | `9f5db7a` | Study protocol (pre-registered) + forecast cache for promotion-policy study |
| `a1878b0` | `4ff7392` | Study: DM test, bootstrap, policies, replay loop, station loader + tests (before any replay results) |
| `b071dbf` | `c12f4a2` | Station dataset (CPCB via Kaggle) under DVC, study DVC stages, paper draft and verified references |
| `4463cbd` | `73e9196` | Protocol deviation D1: add 5 cities (no LSTM) + eligibility rule, before any results |
| `41aa8fd` | `1d5d5dc` | Study D1: city datasets (Delhi, Bengaluru, Chennai; Kolkata/Mumbai fail eligibility), per-dataset candidates, replay variants, multi-city figure |
| `aaae96d` | `7227806` | Study results: 6 replays, hypothesis scoring (H1 13/15, H2 5/15, H3 12/15), paper figures and numbers |
| `6bb8f36` | `2195424` | Paper: results, discussion, conclusion from generated macros; protocol D2; study notes |
| `2f7ae5d` | `71ee210` | Extended (arXiv) paper version: switch, pipeline diagram, testbed + case study, generated tables, full-results and protocol appendices; D2 disclosure in both versions |
| `ad7bf89` | `d245b6e` | Paper layout: one combined full-results table, FloatBarrier before references, top placement for tables |
| `ffaa1d6` | `983d0a1` | Paper layout fixes: full-width static table and figures, single timeline figure per version, diagram arrow, title wrap |
| `667dc16` | `cf1d6d8` | Paper layout: diagram row gap, reproducibility section only in conference version (Appendix C covers it in extended) |
| `fdcd9b5` | `8305277` | Prepare for public release: MIT licence, README paper + data attribution sections, private to-do moved out of docs |
| `5aedc88` | `a32644d` | Remove project plan from the repository (kept locally, git-ignored); update doc references |
| `e094879` | `6b43bc2` | Paper: author name and email |
| `c8caa48` | `c801694` | Paper: affiliation |
| `466c8f6` | `c08db5e` | Licence: copyright holder Aishwarya Nalla |
