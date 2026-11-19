---
name: PkmLens
description: A warm collector marketplace with card imagery and live-camera identification.
colors:
  primary: "#b63e0e"
  paper: "#fcf9f4"
  surface: "#ffffff"
  ink: "#292b2e"
  muted: "#6e6258"
  line: "#eee6db"
  orange-light: "#fff0e3"
typography:
  display:
    fontFamily: "Be Vietnam Pro, sans-serif"
    fontSize: "clamp(30px, 3vw, 44px)"
    fontWeight: 800
    lineHeight: 1.3
    letterSpacing: "-0.035em"
  headline:
    fontFamily: "Be Vietnam Pro, sans-serif"
    fontSize: "clamp(30px, 3vw, 43px)"
    fontWeight: 800
    lineHeight: 1.28
    letterSpacing: "-0.04em"
  title:
    fontFamily: "Be Vietnam Pro, sans-serif"
    fontSize: "16px"
    fontWeight: 700
    lineHeight: 1.4
    letterSpacing: "-0.02em"
  body:
    fontFamily: "Be Vietnam Pro, sans-serif"
    lineHeight: 1.75
  label:
    fontFamily: "Be Vietnam Pro, sans-serif"
    fontSize: "13px"
    fontWeight: 700
    lineHeight: 1.5
rounded:
  tag: "5px"
  art: "8px"
  control: "10px"
  tile: "13px"
  panel: "16px"
  feature: "20px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "20px"
  xl: "24px"
  section: "28px"
  wide: "40px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.surface}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "12px 21px"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "12px 21px"
  input-field:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "12px 14px"
  navigation-active:
    backgroundColor: "#fffcf8"
    textColor: "{colors.primary}"
  tag:
    backgroundColor: "#fff0d7"
    textColor: "#805319"
    rounded: "{rounded.tag}"
    padding: "4px 7px"
  card-tile:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.tile}"
  candidate-option-selected:
    backgroundColor: "#fff4e5"
    textColor: "#6c421d"
    rounded: "{rounded.art}"
    padding: "10px"
---

# Design System: PkmLens

## Overview

**Creative North Star: "Collector marketplace"**

PkmLens uses warm light surfaces, orange accents, charcoal actions and real card imagery. The owner-supplied PokéBooth desktop/mobile screenshots establish the marketplace direction: prominent search, familiar navigation, desktop filtering and a gallery of collectible cards. PkmLens retains its own name, copy and factual content.

This documents the implemented web UI in `frontend/styles.css`, `frontend/App.tsx`, `frontend/components/CardTile.tsx` and `frontend/components/RecognitionPanel.tsx`, including the final cascade overrides. The YAML owns the extracted reusable tokens; component-specific tints and responsive exceptions remain local. The sidecar adds seven component samples and metadata; its generated tonal ramps are preview aids, not additional shipping colors.

**Key Characteristics:**

- Warm cream page, white surfaces and restrained orange accents.
- Actual card art with legible printing, price and source information.
- Familiar search, filters and responsive card grids.
- A dark camera area paired with light, inspectable recognition results.

## Colors

The palette combines warm paper and brown-gray supporting text with dark charcoal for strong actions.

### Primary

- **Burnt orange** (`primary`) identifies active navigation, links, caret and checkbox accents. The final root declaration overrides the earlier brighter orange.
- **Soft orange** (`orange-light`) supplies selected view/track and icon-hover backgrounds. The brighter brand mark and mobile camera icon retain their existing local colors.

### Neutral

- **Warm paper** (`paper`) is the storefront canvas; **white surface** (`surface`) carries cards, panels and fields.
- **Charcoal** (`ink`) carries principal text and primary buttons; **warm muted** (`muted`) carries supporting text where inherited.
- **Cream divider** (`line`) separates navigation, cards, filters and evidence rows. Local brown text overrides remain in toolbar and recognition details.

Dark sage is confined to the camera media area. Amber and green recognition/connection states include text or other cues alongside color.

## Typography

Be Vietnam Pro is self-hosted from `/fonts/`, with supplied weights (400, 500, 600, 700, 800), `font-display: swap`, `font-synthesis: none` and a sans-serif fallback. The same family serves headings, Vietnamese body copy and controls.

- **Display:** the general page-heading baseline; individual page headings have their existing size overrides.
- **Headline:** the marketplace banner; at the wide breakpoint it becomes (46px). On phones the final rule is (23px), line-height (1.3), including widths below (359px); the later phone rule supersedes the earlier (22px) declaration.
- **Title:** card names use the extracted desktop role and become (14px) on phones. Set/source text finishes at (11px) on phones; tile tags finish at (10px).
- **Body:** paragraph rhythm follows the extracted line-height. There is no single explicit global body-size token; compact copy varies by component. Recognition evidence uses (12px), with explanatory and price-source copy generally (11–12px).
- **Label:** the main button role. Form labels instead use (13px, 600) before local overrides. Amounts are bold and larger than their price/source labels.

## Layout

The centered page container includes its gutters in a maximum width (1456px). Its horizontal padding steps from (40px) to (28px), (20px) and (16px). The layout uses the extracted spacing steps alongside component-specific values; it is not a strict uniform grid of spacing multiples.

| Viewport rule | Observed behavior |
| --- | --- |
| Default | Filter rail (230px), layout gap (28px), four card columns with (17px) gaps; camera beside a recognition column (315px). |
| At or below (1200px) | Gutters (28px), rail (205px), three card columns; recognition column (285px). |
| At or below (950px) | Rail (180px), card gaps (12px); recognition column (270px). |
| At or below (760px) | Gutters (20px), filter rail replaced by a modal dialog, three card columns; camera and recognition stack; search occupies its own header row. |
| At or below (540px) | Gutters (16px), two card columns with (12px) gaps, toolbar controls wrap; list view becomes one column. |
| At or below (359px) | View-toggle controls and the tile's compact saved-price badge are hidden; the final phone banner sizing still applies. |

On phones the banner is a compact block: description, caption and stars are hidden; a small pair of actual cards sits in an absolutely positioned art area (82px × 115px). The heading reserves the right-hand artwork width (82px). The banner no longer creates a separate tall art row.

At or below (760px), the root is a viewport-height grid (`100dvh`, `minmax(0,1fr) auto`, overflow hidden). The header, main and footer scroll within `.app-scroll` (`overflow-y: auto`, `overflow-x: hidden`, `min-height: 0`). The five-item bottom navigation occupies the second grid row, is positioned relatively, and has minimum height (66px) with safe-area-aware bottom padding. It does not overlay the content. The scroll container has bottom scroll padding (30px), main bottom padding is (40px), and interactive elements retain scroll margins (90px). Route changes reset both document and content-container scroll.

Catalog filtering and sorting (including reference-price sorting) run before server pagination; catalog pages contain up to (24) records and reset when the result scope changes. Price sorting asks for a narrower result set above (3000) matches. Sales have a distinct scope: the API returns up to the newest (100) active listings matching the requested search/set, then the browser filters and sorts that returned subset. Do not describe sales sort or pagination as server-side. These are current browsing limits, not a universal data-volume guarantee.

## Elevation & Depth

Warm fills and thin borders establish most separation. Soft brown shadows distinguish panels, the search field and card hover; card art uses contained imagery and light drop shadows rather than heavy frames.

- **Panel:** `0 9px 28px -13px #3f2a1724` (the existing `--shadow`).
- **Card hover:** `0 10px 24px -13px #6c432d36`, with a warmer border.
- **Search:** `0 4px 20px -12px #4e322739`.

Control colors transition over (180ms), shadows over (200ms), and card border/shadow over (250ms). Image transforms use the existing easing (`cubic-bezier(.16,1,.3,1)`), with tile motion (400ms) and banner-card motion (500ms). The reduced-motion query removes animations and transitions and restores automatic scroll behavior.

## Shapes

Controls have gently rounded corners; tags and art insets use smaller radii, cards and panels use broader curves, and the desktop banner uses the largest extracted radius. The mobile banner retains its existing (15px) radius. Thin warm borders provide definition. Circular save controls and avatars are established exceptions to the rounded-rectangle vocabulary.

## Components

### Buttons

Primary actions are white-on-charcoal; secondary actions are white with warm borders. Both use the extracted control radius and padding, with minimum height (46px). Primary hover becomes warm charcoal; secondary hover warms its background and border. Disabled buttons retain native semantics and opacity (.48).

Global keyboard focus uses an orange outline (3px) with offset (4px). Save buttons are (44px × 44px). The phone banner primary action, filter/sort controls and mobile camera stop button have minimum heights (44px); mobile camera footer icons are at least (44px × 44px). The later mobile rule supersedes their earlier phone shrink rule. These observations do not imply that every existing control has that size.

### Chips

Tags use warm amber with brown text, with a neutral variant for language/category. Active filter chips are removable buttons; rarity and condition tags are informational. Their appearance does not imply stock or seller trust.

### Cards / Containers

Card tiles combine a white bordered shell with a softly tinted art inset. Images use `object-fit: contain`; a failed image receives the existing icon/text fallback. The independent save button exposes pressed state. Card titles link to the printing or listing; the decorative art link is excluded from keyboard and accessibility navigation to avoid duplicating that destination.

Tiles distinguish seller asking prices from reference prices and render missing prices explicitly. A stale reference can show the compact saved-price badge. Detail and recognition views provide the fuller source/date/staleness context; that information is not present in every tile. Panels reuse the white surface, warm border, panel radius and ambient shadow.

### Inputs / Fields

Standard fields use white backgrounds, warm strokes, extracted control corners and minimum height (46px). Native inputs/selects remain the foundation. The prominent search field wraps the input and submit button in one outlined surface; its wrapper gains a focus-within outline (2px). Errors and unavailable states retain their existing text explanations.

### Navigation

Desktop navigation uses a bottom orange stroke and accent text for the current route. On mobile, the upper navigation scrolls horizontally while the bottom five destinations keep scanning prominent. Navigation labels remain visible beside or below icons. The filter dialog provides the small-screen counterpart to the desktop rail.

### Recognition and correction

The white recognition panel presents card art, named status, printing, image similarity, OCR number and frame evidence. Users can choose another candidate or search the catalog manually; selected candidate buttons have warm fill, border, checkmark and `aria-pressed`. Candidate controls have minimum height (54px), manual matches (48px), and phone correction focus uses its local outline (3px, offset 2px).

The camera discloses an experimental model near the start action and below the media area when applicable. Similarity and synthetic validation are not presented as real-camera accuracy. Reference quotes show source, date and saved price/FX status when supplied, with the NM assumption visible in recognition. Unknown results retain OCR and the explicit unidentified-save path. Session saving uses an explicit confirmation action.

## Do's and Don'ts

### Do:

- **Do** use the final accent and muted tokens with warm paper, white surfaces and charcoal actions.
- **Do** preserve real card art, containment, printing labels and the existing image/price fallbacks.
- **Do** preserve the compact phone banner and separate mobile content-scroll/navigation rows.
- **Do** retain visible keyboard focus, native control semantics, correction choices and reduced-motion behavior.
- **Do** keep reference-price source/date/staleness and recognition uncertainty visible where the implementation provides them.

### Don't:

- **Don't** revive superseded token values or the earlier tall mobile banner and fixed-overlay navigation rules.
- **Don't** promote camera sage to the storefront background or replace the confirmed warm marketplace direction.
- **Don't** copy PokéBooth branding or add unsupported seller ratings, stock, trust or recognition claims.
- **Don't** turn component-specific tints, preview color ramps or current demo data limits into new global design requirements.
