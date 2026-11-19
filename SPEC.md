# PkmLens V1

Status: implementation authorized through the conversation; build incrementally.

## Objective

Build a polished React 18 marketplace backed by FastAPI, continuous YOLO camera recognition and a separate desktop training tool. Use CardCorpus for references and suitable existing CardLens code for the backend.

## Structure and order

| Module | Responsibility |
|---|---|
| catalog-data | Reference images, card metadata, CSV, provenance and sourced prices |
| yolo-pipeline | Labeled synthetic scenes, standard dataset validation, train/validate/checkpoints |
| card-recognition | YOLO localization, rectification, retrieval, OCR and unknown handling |
| marketplace-api | Accounts, ownership, listings and demo transactions |
| web-app | Search/filter/detail/sell flows and live-camera interface |
| desktop-trainer | Local window for preparation, train configuration, logs, metrics and resume |

Build the web catalog and marketplace slice first, connect recognition next, then desktop training. Use SQLite and local assets. Do not mutate D:/Project/CardLens.

## Commands

- Frontend: `npm run dev`, `npm run build`, `npm run typecheck`.
- Backend: `python -m uvicorn cardlens_core.main:app --host 127.0.0.1 --port 8100` with project service paths.
- AI: `python -m uvicorn cardlens_ai.main:app --host 127.0.0.1 --port 8101`.
- Tests: `python -m pytest -q`.
- Setup/run: PowerShell scripts under `scripts/`.

## Acceptance

- React and react-dom stay on major 18; Vite proxies HTTP and authenticated WebSocket traffic to FastAPI.
- Responsive marketplace follows the supplied warm/orange reference, renders real card art, filters and detail views with loading/error/empty states.
- Accounts and seller-owned listings are stored in a separate PkmLens database.
- Camera continuously sends bounded frames; YOLO returns real boxes, tracks expire, and stopping/leaving releases the media stream and connection.
- Retrieval/OCR returns evidence and uncertain/unknown states. New references can be added without card-ID detector retraining.
- Metadata CSV contains image_path,card_id,name,set,number,rarity. Synthetic train/validation/test separation preserves source groups.
- Desktop UI can prepare/import a dataset, select GPU/CPU, train, show actual metrics, stop and resume without losing checkpoints.
- Validate API/ML changes with focused tests, run typecheck/build and inspect desktop/mobile browser output.

## Boundaries

Always validate external images, paths and ownership. Keep price source/date/variant explicit. Label synthetic data and demo fixtures. Never invent recognition metrics, commercial stock or seller trust claims. Real payments, authenticity grading, public deployment and training on collected user photos without consent are outside V1.
