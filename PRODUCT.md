# PkmLens

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

User-required FastAPI + React 18. Frontend uses TypeScript and Vite. Python handles YOLO, OCR, image retrieval and a separate local training application.

## Users

Pokémon card collectors who browse, identify and list cards. The project owner is a student preparing a Machine Learning course project and needs to operate model training independently.

## Product Purpose

A marketplace for multiple sellers with continuous live-camera identification. YOLO detects cards; image retrieval and OCR resolve the printing. Reference prices come from sourced price records, independently of seller asking prices.

## Operating Context

Windows local demo, CPU support and optional NVIDIA GPU. A separate desktop UI prepares datasets and runs training after the website is built. CardCorpus is the main catalog source. Start with 1–3 sets before scaling up downloads.

## Capabilities and Constraints

- React 18 is mandatory; do not replace it with Next.js or React 19.
- Live camera is mandatory; upload alone does not satisfy recognition.
- New cards outside the detector training set can be identified after adding references to the gallery without retraining card-ID classes.
- Cards absent from the catalog return available OCR text and an unknown state. Do not fabricate identity or price.
- The owner has no labeled YOLO dataset. Provide synthetic scenes with generated bounding boxes and allow importing standard YOLO datasets.
- Reuse suitable backend code from D:/Project/CardLens without modifying that project or using its accounts/database.
- Initial scope is an academic demo. Real payments and public deployment are not authorized.

## Brand Commitments

The owner requires a beautiful interface and supplied https://pokebooth.com/#/market plus desktop/mobile screenshots. The reference has a warm light background, orange accents, prominent search, desktop filter sidebar and product imagery. PkmLens must give live AI scanning a clear entry point. Do not copy PokéBooth branding, seller ratings or factual claims.

## Evidence on Hand

CardLens contains FastAPI marketplace and live scanner code, DINOv2/FAISS/OCR, 103 cached reference images and a YOLO synthetic-scene generator. Its yolo26n checkpoint reports synthetic validation only, not real-camera accuracy. The current machine has a CUDA-capable GTX 1660 Ti. Skills Ponytail, UI UX Pro Max and Impeccable are installed project-locally.

## Product Principles

1. Show usable cards and actions immediately.
2. Keep sourced catalog records, demo listings and real user listings distinguishable.
3. Let users inspect and correct uncertain AI results.
4. Report actual training metrics and the kind of data measured.
5. Preserve camera, keyboard and small-screen usability.
