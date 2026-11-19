# Spec delta: CardCorpus printing and variant coverage

Approved scope: 95,000+ sourced card+variant options, not 95,000 independent detector classes. Preserve local FastAPI/React 18, YOLO, tracking, pricing, existing card IDs, and the small-gallery fallback.

1. Import the local CardCorpus snapshot into the existing catalog, preserving printing ID, printed collector number, exact source language, provenance and variant IDs; preserve prior identities, listings and source quotes. Count actual mapped cards and distinct (card ID, variant ID) options.
2. Use the published CardCorpus reference embeddings only with their paired matcher and verified artifacts. Map reference rows to existing local card IDs. Deduplicate candidates by printing, keep OCR/quality rejection, and leave variant unverified even when a reference has variant metadata. Report indexed printings, reference images and catalog variants of indexed printings separately. Keep offline camera inference and the current gallery as fallback.
3. Save a variant only when explicitly selected by the user. Keep unselected variants and unsourced prices unknown. Do not substitute distinct finishes or languages.

Verify repeat import, checksums and row alignment, clean checkout/sell behavior in existing tests, real queries against the expanded index, and separately labeled latency and accuracy evidence. No changes to CardLens, detector weights, payment behavior, or public deployment. No claim of camera accuracy without independent data.
