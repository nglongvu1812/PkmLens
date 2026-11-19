# PkmLens final local verification - 2026-11-18

FastAPI + React 18.3.1, Windows, GTX 1660 Ti 6 GB. Local academic demo; no physical-camera accuracy certification.

## Current checks
- Full Python suite: 451 passed, 19 subtests passed in 155.74 s (2026-11-19; earlier pre-CardCorpus-scale baseline was 370 passed, 19 subtests passed).
- Frontend typecheck and production build: passed.
- Studio focused dataset/training/API/health checks: 157 passed + 19 subtests; desktop smoke passed.
- Camera browser: actual bbox and trained pose YOLO through authenticated gateway, ByteTrack, warp, DINOv2/FAISS/OCR; correct Charizard ex #223 after three votes; manual correction, save, pause/resume, stop cleanup passed.
- Media UI fixtures: JPEG/PNG/WebP, WebM and H.264 MP4 decoded, pause/resume, natural end, source URL revocation and socket cleanup; configuration/encoding/permission races; desktop 1440px/mobile 390px. These fixtures do not measure AI accuracy.
- Earlier marketplace browser evidence covered browse, responsive filtering, listings and simulated buy/pay/ship/receive. Repeated duplicate listing photos enter pending_review; no bypass was added.

## Pose training and frozen test
The prepared pose dataset has 800 images (600/100/100), 1,748 objects (1,283/228/237), and 116 negatives (86/15/15). Source JPEG bytes are unchanged; labels use one pokemon_card class and four semantic TL/TR/BR/BL corners. Ultralytics accepted every label without repair. No independent annotated real-camera images exist.

Run data/training/pose-yolo26n-synthetic-v1: YOLO26n pose, 25 epochs, 480px, batch 4, GPU0.
- Validation P/R/mAP50/mAP50-95: 0.995424/0.995614/0.993728/0.993711.
- Held-out test, 100 images and 237 objects: 0.999770/1.000000/0.995000/0.994717.
- best.pt SHA256: 7a1fdb0724b671f46d1133d12f1f2fbef8f54c03669a8112d34baa24231b897b.
- Dataset content SHA256: f8405bc564d1ceb2353100c86f1322fcc57ce3b576a01f3281dd0c29575c39ce.
- Experimental activation verifies terminal status, pose metrics, containment/hash and actual checkpoint schema. Bbox remains default; pose is explicit opt-in; real_photo_validated=false.

## Browser timing and KPI limits
The camera fixture is a synthetic still scene presented as video. The original source is 10 FPS; a separate fixture preserves the same pixels at 30 FPS.
- Pose 10-FPS fixture: 101 frames, detector+track p50/p95 70.18/112.17 ms, JPEG-to-frame roundtrip 84/127 ms, preview 9.9 FPS, first confirmation 17.151 s.
- Pose 30-FPS fixture: 109 frames, detector+track p50/p95 77.32/122.08 ms, preview 30.4 FPS, AI about 6.3 updates/s, first confirmation 18.241 s.
- OCR/embedding identity is asynchronous and excluded from JPEG-to-frame roundtrip. These measurements include local concurrent work/cold state; they do not prove a physical-camera accuracy or switch-time KPI.
- Requested >=30 preview FPS was observed only on synthetic 30-FPS input. Requested >=98% recall was observed only on synthetic pose test. Exact-card >=95% and <500ms new-card switch are not established on independent camera data. First consensus here is much slower.

## Retrieval and calibration
Active DINOv2 remains pretrained_not_finetuned with a 127-card FAISS gallery. Offline triplet-loss training updates a projection over frozen DINOv2 and exports a separate versioned staging bundle. It does not claim backbone fine-tuning or production activation.
Benchmark data/evaluation/recognition-synthetic-v1/manifest.jsonl contains 24 oracle-corner crops of 12 test-source cards and six synthetic backgrounds; benchmark excludes detected localization and temporal voting. It is not an independent unknown-card dataset, nor calibration input. See data/benchmarks/synthetic-oracle-crops-v1/summary.json and results.jsonl.
Calibration remains inactive and probabilities uncalibrated. v2 eligibility preserves non-threshold vetoes (number-only override, OCR conflict, quality/open-set floor); fingerprints include non-tuned policy; legacy policy-less reports fail closed. No deployable calibration artifact exists without independent fit/check validation groups.

## ONNX and TensorRT
Project-local dependency targets add onnx 1.23.2 and ml_dtypes 0.7.0 without modifying CardLens. ONNX export (opset18, fixed 480px/batch1) succeeded. Real CPU parity compares matched boxes/corners/confidences; full evidence is reports/onnx-pose-verification.json.
Direct ORT CPU test on 100 synthetic test images/237 objects: pose P/R/mAP50/mAP50-95 0.999798/1.000000/0.995000/0.994705. Actual batch forced to 1 by static graph. See reports/onnx-heldout-test.json. Production activation/evaluate CLI deliberately accepts trusted .pt only; exporter parity does not switch runtime backends.
TensorRT module/trtexec/engine are unavailable. No TensorRT benchmark claim.

## Studio use
Choose Pose tab, select data/datasets/cards-pose-synthetic-v1 and local data/models/card-detector-best.pt, choose n/s scale and CPU/GPU settings. Custom pose architecture is built before transferring matching weights; no hidden model downloads. STOP finishes the current epoch; resume.pt preserves optimizer and same-run config/data hashes. Results provide test (default) or val evaluation saved in a new directory, measured charts, experimental activation and ONNX export.
CLI examples:
    python -m ml.pose_dataset validate data/datasets/cards-pose-synthetic-v1
    python -m ml.train_pose train --data data/datasets/cards-pose-synthetic-v1/pose.yaml --checkpoint data/models/card-detector-best.pt --output data/training/new-run --scale n --epochs 25 --batch 4 --imgsz 480 --device 0
    python -m ml.train_pose evaluate --data data/datasets/cards-pose-synthetic-v1/pose.yaml --checkpoint data/training/pose-yolo26n-synthetic-v1/weights/best.pt --output data/training/new-test --split test --batch 4 --imgsz 480 --device 0

## Remaining limits
No physical webcam/phone/real unknown-card benchmark, clean-machine installation, full screen-reader/load audit, or deployable calibration. CardCorpus imported cards lack an independent sourced price feed; missing prices remain unavailable. Model scores are similarities, not calibrated probabilities. No public deployment, real payments, CardLens edits or git commits. Workspace is not a git repository.


## Final additions and measured retrieval benchmark

- `data/benchmarks/synthetic-oracle-crops-v1`: 30 samples, 24 known oracle-corner crops from 12 test-source cards and 6 synthetic backgrounds. This isolates retrieval/OCR on perfectly supplied corners; it excludes detector, temporal tracking and physical image quality. It is not calibration evidence and the backgrounds are not unseen physical cards.
- See `data/benchmarks/synthetic-oracle-crops-v1/summary.json` for per-method retrieval top1, acceptance/coverage, unknown/reject matrix and measured stage timings. OCR is a large synchronous stage in the one-shot benchmark; do not treat its rate as full webcam frame rate. Runtime identity work is asynchronous to detector frames.
- Benchmark v2 eligibility preserves number-only override, OCR conflict, quality/open-set floors and missing-competitor vetoes. Fingerprints include exact non-tuned brightness/sharpness/glare/open-set policy. Tuned thresholds are outputs, not fingerprint inputs; legacy reports without policy evidence fail closed. Calibration stays inactive because no independent, grouped known/unknown fit/check set exists.
- Shared `STATUS.md` is a brief pointer to this file and the evidence reports; older phase history above is retained for traceability, while this final section records present state.

## Additional bounded local-file test
- Still image through real active pose/WebSocket: Charizard ex #223 reached 3/3 in 12.039 s. This repeatedly analyzes one JPEG; not independent frames. Detector+track p50/p95 66.88/92.75 ms; frame roundtrip 79/106 ms.
- A 30.072 s synthetic H.264 MP4 repeat did NOT reach consensus. Full-scene attempt: 189 tracks, OCR alternated `223`, `OBF223`, `OB223`, resetting votes. Card-cropped retry: 13 tracked frames, detector+track p50/p95 2,103.65/2,793.84 ms, roundtrip 2,111/2,802 ms. Playback decoded 898 frames and source cleanup passed; AI video identification failed in the allowed clip duration. See `reports/browser-ai-media-verification.json`.
- Synthetic-oracle retrieval benchmark: 24 known perfect-corner crops/12 sources and six synthetic backgrounds; retrieval top1=1.0, decision coverage .80 (pHash)/.767 (embedding methods), accepted accuracy=1.0. Six backgrounds are not unknown Pokémon cards and cannot estimate open-set FAR. OCR p50 2.09-2.30 s/p95 3.08-3.21 s; one-shot p95 3.30-3.41 s. Excludes detector/tracking and does not represent live temporal scanning. See `data/benchmarks/synthetic-oracle-crops-v1/summary.json`.

## CardCorpus full-scale catalog, paired gallery and benchmark verification (2026-11-19)
- SQLite `data/pkmlens.db`: 67,243 cards (67,243 linked CardCorpus printings, 104,103 source variant options, 67,085 front images, 158 missing images; snapshot SHA-256 `0203e5097e020129ab76e398f5c800e2f58c1536e246b455b2a375468f43b51b`), 8 users, 3 listings, 1 order, 210 price snapshots; `PRAGMA quick_check=ok`, 0 foreign-key violations. Verified backup: `data/backups/pkmlens-pre-source-variants-20261119-115257.db`.
- Paired gallery bundle `data/models/cardcorpus-scanner`: `mapping.json` SHA-256 `43c31696e18dbe08f5dac35540ad3dbe51bd462478c9c7fc9eaee3a69005eed1` maps 67,243 printings and 104,519 local + source-only variant options (464 source-only IDs). Direct readiness evaluation (`cardlens_ai.main.health()`) confirms `status="ok"`, `active="cardcorpus"`, `prepared=true`, `large_index_loaded=true`, `reference_images=102,185`, `mapped_reference_images=99,173`, `indexed_printings=65,297`, and `indexed_catalog_variants=101,902` out of `104,519`.
- `ml/benchmark.py` keeps `legacy` default (`identify(..., cardcorpus=None)`) and supports `--backend cardcorpus` with helper functions defined before `__main__` and canonicalized `model_versions` on quality-rejected samples (never leaking `"not_run"` trace metadata into quality-reject fingerprints).
- Focused catalog/runtime/benchmark suite: 164 passed (`tests/test_benchmark_stages.py` + `tests/test_benchmark.py`: 23 passed). Full pytest suite: 451 passed, 19 subtests passed in 134.99 s. Frontend `npm run typecheck`, `npm run build`, `python -m compileall`, `node scripts/browser-visual.cjs`, `node scripts/browser-camera-media.cjs`, `node scripts/browser-variant-selection.cjs`, `node scripts/browser-catalog-scale.cjs`, and `node scripts/browser-camera.cjs` (bbox, pose 10-FPS, pose 30-FPS) all passed.
- Live camera, multi-card & tilted-card optimization (2026-11-19):
  - Single-batch 10-strip `TextRecognizer` fast path (6 bottom strips + 4 header strips, `rec_batch_num=10`, `det_limit_side_len=224`, `use_cls=False`, gated footer DBNet fallback, Trainer/Energy banner token filtering, and joined set-code stripping `OBF223/197`, `OB223/197`, `0BF223/197`, `MEWE199/165` while preserving subset prefixes `GG`, `TG`, `SV`, `RC`, etc.) reads 127/127 gallery numbers in a single `text_rec` batch (0 footer DBNet fallbacks) and cut median one-shot `identify()` latency from **5,321.6 ms** to **449.0 ms** (~11.8x speedup).
  - Dedicated persistent GPU worker thread (`run_on_gpu` in `detection.py` + `large_gallery.py`) eliminates Windows per-thread 2.3 s CUDA/cuDNN init stalls across `asyncio.to_thread` workers; startup warmup in `scripts/serve.py` (`PKMLENS_WARMUP_ON_START=1`) pre-warms YOLO bbox, YOLO pose, DINOv3, and RapidOCR; live recognition skips unused base64 PNG encoding (`encode_images=False`).
  - Multi-card & tilted-card upgrades: border keypoint clamping in `detect_pose`, full-frame cached pose-assisted 4-corner rectification inside `rectify()` for tilted boxes (`aspect > 0.76`), 180-degree rotation ranking (`_recognition_rank`), same-name reprint distinct-competitor margin (`effective_margin`), high-similarity (`>= 0.85`) visual consensus eligibility when tiny collector numbers are unreadable (`ocr.number is None`) with zero OCR conflicts, round-robin priority scheduling (`len(tasks) < 6`), live multi-card overlays (`name #number · sim`), and one-click `"Lưu tất cả (N thẻ)"` session save.
  - Real Chrome verification: single-card fake-camera (`scripts/browser-camera.cjs`) confirmed `Charizard ex, Obsidian Flames #223` 3/3 in **3.89 s** (Bbox 10-FPS, AI 9.5 updates/s, detector+track p50/p95 `78.88 / 215.89 ms`), **3.83 s** (Pose 10-FPS, AI 11.5 updates/s, p50/p95 `67.25 / 224.97 ms`), and **4.39 s** (Pose 30-FPS, preview 30.0 FPS, AI **12.5 updates/s**, p50/p95 `64.47 / 166.16 ms`). Multi-card tilted synthetic scene (`test-00002.jpg`, 3 cards at arbitrary rotations including ~65° and ~135° upside-down) confirmed all 3/3 cards simultaneously (`Ivysaur #30`, `Pokémon Flute #86`, `Charmander #168`) in **6.14 s** (Pose mode with 3 rotated quadrilaterals) and **9.18 s** (Bbox mode with auto-pose rectification) and saved all 3 in one click. Offline multi-card tilted suite (`test-00001..00005.jpg`): 11/11 usable tilted cards reached `needs_confirmation` in both Pose and Bbox modes (`sim 0.948–0.988`).
  - Full pytest suite: 453 passed, 19 subtests passed. Frontend `npm run typecheck`, `npm run build`, `python -m compileall`, `browser-visual.cjs`, `browser-camera-media.cjs`, `browser-variant-selection.cjs`, `browser-catalog-scale.cjs`, and `browser-camera.cjs` all passed.
- Reference-image index coverage is not physical-camera or open-set recognition accuracy. Reference variants (`reference_variant`) are not visually verified (`variant=None`, `variant_verified=False`). Source-only variants without sourced price feeds remain unpriced. All three PkmLens services (5173/8100/8101) are running the updated code and respond HTTP 200.
