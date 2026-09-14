# AutoTrack — Design System (Master)

> **LOGIC:** page-specific overrides would live in `pages/[page-name].md` and take
> precedence over this file. AutoTrack's interface is a single page (three tabs of
> one Gradio app), so no page override exists — this file is authoritative.

**Generated:** 2026-09-14, via `ui-ux-pro-max` skill (`--design-system`), then
**hand-corrected** against this project's actual product/stack — see "Why this
deviates from the raw tool output" below. Applied to `interface/app.py`.

---

## 0. Product context (read before the tables)

- **Product type:** a single-page **interactive engineering tool** — upload a
  video / use a webcam / connect an IP camera, watch a computer-vision tracker
  run on it live. Not a marketing site, not a multi-page SaaS dashboard, not a
  docs portal.
- **Audience:** technical reviewers (portfolio/job-application context —
  Turkish aviation/defense sector), and the author demoing it live.
- **Stack:** **Gradio 6 (Python)**, not a listed stack in this skill
  (`react`/`nextjs`/`html-tailwind`/etc. do not apply). Styling is a CSS string
  passed to `gr.Blocks.launch(theme=..., css=...)` plus a `gr.themes.Soft(...)`
  base — there is no JSX/Tailwind class layer to target, so "component specs"
  below are plain CSS + Gradio theme tokens, not React/Tailwind snippets.
- **Existing visual identity to stay consistent with:** the project's README
  highlights section and a published case-study Artifact both already use a
  "field report" identity — IBM Plex Sans/Mono, a warm paper/ink palette, a
  bounding-box corner-bracket motif. The interface is the third surface in
  that same set and should read as the same product, not a restart.
- **A real, code-level constraint:** `viz/overlay_renderer.py`'s
  `BOX_COLOR_LOW = (80, 200, 80)` — the tracker's own bounding boxes render in
  **green**. A green UI accent isn't just a style preference here; it's the
  same color already present in every screenshot/video the tool produces.

## 1. Why this deviates from the raw tool output

Ran `--design-system` twice (`"computer vision AI analytics tool dashboard
technical"` and `"internal developer tool utility single page technical"`),
plus standalone `style`/`color`/`ux`/`icons` domain queries. Per the skill's
own instruction to verify fit before applying — three real mismatches:

| Tool output | Verdict | Reason |
|---|---|---|
| Pattern: **"Enterprise Gateway"** (Hero → Solutions by Industry → Solutions by Role → Client Logos → Contact Sales, CTA "Contact Sales") | **Rejected** | This is a B2B marketing-site pattern. AutoTrack has no sales funnel, no logo wall — it's a working tool. |
| Pattern (2nd query): **"FAQ/Documentation Landing"** (search bar, FAQ accordion, support CTA) | **Rejected** | Also not applicable — no support/docs use case. |
| Style: **HUD/Sci-Fi FUI, Cyberpunk UI** (neon glow, scanlines, glitch) | **Rejected** | Wrong mood for a defense-sector job application — the project's whole differentiator is rigorous, sober research documentation; a neon "hacker" skin would undercut that credibility. |
| Colors: **blue primary `#1E40AF` + amber accent `#D97706`** ("Data-Dense Dashboard" / generic enterprise-blue) | **Rejected as primary**, kept as a documented alternative | Breaks the established green identity across README/case-study/overlay (see §0). Two separate color-domain queries *did* independently converge on green as the accent for "developer/security tool" contexts, which is the one signal from the tool I did keep. |
| Typography: **Fira Code/Sans** (1st query) vs. **JetBrains Mono + IBM Plex Sans** (2nd query) | **Partially kept** | Both land in the same "technical/precise/dashboard" mood as the already-established IBM Plex Sans/Mono — the 2nd query's own pairing literally includes IBM Plex Sans. Kept the existing pairing for cross-surface consistency rather than switching fonts a 3rd time across the portfolio. |

What I **did** keep and apply: the "Data-Dense Dashboard" style's general
sensibility (minimal chrome, space-efficient, no ornamentation), the full
pre-delivery accessibility checklist, the spacing scale, and green-as-accent.

## 2. Color tokens

Every text/background pair below was computed (WCAG relative-luminance
formula), not eyeballed — script output kept in this file so it can be
re-checked instead of re-trusted.

### Light (default)

| Role | Token | Hex | Contrast (checked) |
|---|---|---|---|
| Background | `--at-bg` | `#F4F5F0` | — |
| Surface (cards/blocks) | `--at-surface` | `#FFFFFF` | — |
| Ink (body text) | `--at-ink` | `#1B2320` | 16.06:1 on surface, 14.65:1 on bg |
| Ink muted (labels/help text) | `--at-ink-muted` | `#5B6660` | 5.98:1 on surface, 5.45:1 on bg |
| Border | `--at-border` | `#D8DAD2` | decorative divider only, not a required affordance — WCAG's 3:1 non-text rule doesn't apply (see note below) |
| Primary / accent | `--at-green` | `#2F6E4D` | 6.07:1 both directions vs. white |
| Primary hover | `--at-green-hover` | `#255A3E` | darker than base, contrast only improves |
| Text on primary button | `--at-on-green` | `#FFFFFF` | 6.07:1 on `--at-green` |

### Dark (`prefers-color-scheme: dark` or explicit toggle)

| Role | Token | Hex | Contrast (checked) |
|---|---|---|---|
| Background | `--at-bg` | `#12140F` | — |
| Surface | `--at-surface` | `#1B1D17` | — |
| Ink | `--at-ink` | `#EDEFE9` | 14.68:1 on surface, 16.00:1 on bg |
| Ink muted | `--at-ink-muted` | `#9BA39A` | 6.56:1 on surface, 7.15:1 on bg |
| Border | `--at-border` | `#333B32` | decorative |
| Primary / accent | `--at-green` | `#38A873` | 5.68:1 vs. surface |
| Primary hover | `--at-green-hover` | `#2F9760` | — |
| **Text on primary button** | `--at-on-green` | `#12140F` (dark ink, not white) | 6.19:1 on `--at-green` |

**Why dark-mode button text flips to dark-on-green instead of white-on-green:**
measured first — white on the muted `#3F8A63` green only hit **4.18:1**,
under the 4.5:1 minimum for normal-weight button text (button labels here
are ~15-16px/600, under the 18.66px-bold "large text" WCAG cutoff, so the
lighter 3:1 threshold doesn't apply). Brightening the green until white text
passed pushed it toward a neon tone that didn't fit; flipping to dark text on
a slightly brighter green (`#38A873`) passes cleanly (6.19:1) **and** looks
more natural for this palette than white-on-neon would have.

**Border/non-text note:** WCAG 2.2's 3:1 non-text-contrast rule (1.4.11)
applies to UI components/graphics a user needs to *perceive to operate the
interface* — input boundaries, focus rings, icon glyphs. A quiet card divider
that isn't the sole cue for anything is conventionally exempt (same pattern
GitHub/Linear/Vercel use); the rule *is* enforced below for the one thing
that actually needs it — focus rings.

### Semantic (rare use — validation/warning text only, not decorative)

| Role | Light | Dark |
|---|---|---|
| `--at-red` (error/destructive) | `#A73E29` | `#C4593F` |
| `--at-amber` (warning — e.g. "Re-ID checkpoint missing, fell back to ImageNet") | `#8A6A1E` | `#C99A3A` |

## 3. Typography

- **Sans (UI text):** IBM Plex Sans — weights 400/500/600.
- **Mono (data — summary/status panels, code refs):** IBM Plex Mono.
- Loaded via `gr.themes.GoogleFont`, not a `<link>` tag (Gradio injects it
  into the app's own head correctly either way; `GoogleFont` is the
  documented mechanism for this Gradio version).
- **Sizes:** body/UI text 16px (Gradio's own default scale); monospace
  summary/status boxes bumped from 0.83rem → **0.875rem (14px)** — the
  original was legible but sat closer to the 12px floor than necessary for a
  block someone reads results out of, not glances at.
- **Headings:** one `h1` (the page title, "AutoTrack", ~1.55rem/25px) — no
  nested heading levels inside tabs, so there's no h1→h3 skip to check.

## 4. Spacing & radius

Kept the tool's Standard/6-density scale (this app is a focused tool, not a
cramped analytics grid, so "Standard" over "Dense"):

| Token | Value |
|---|---|
| `--at-space-xs` | 4px |
| `--at-space-sm` | 8px |
| `--at-space-md` | 16px |
| `--at-space-lg` | 24px |
| `--at-space-xl` | 32px |
| `--at-radius` | 12px (blocks/cards) |
| `--at-radius-sm` | 8px (buttons) |

## 5. Motion

- All custom hover/focus transitions: **180ms ease** (inside the
  checklist's 150-300ms band). Gradio's own internal component transitions
  are left as-is (out of reach without forking the library).
- No transform-on-hover that shifts layout (checklist anti-pattern) — hovers
  here are color/shadow only, never `translateY`/`scale`.
- `prefers-reduced-motion` — nothing here relies on motion to convey state
  (all feedback is also textual: progress %, status strings), so there is
  nothing that *needs* a reduced-motion override; the one transition
  (button/link hover color) is a fade, which reduced-motion guidance doesn't
  require suppressing.

## 6. Component specs (Gradio + CSS, not React)

### Buttons
- Primary (`variant="primary"`): `--at-green` fill, `--at-on-green` text,
  `--at-radius-sm` corners (6px, not Soft's default pill-ish 10px — see §8),
  1px darker-green border for definition instead of a drop shadow, hover →
  `--at-green-hover`, 180ms transition.
- Secondary (`variant="secondary"`, e.g. "Durdur"): neutral fill
  (`--neutral-100` / `--neutral-800` dark), no colored border — it's a
  regular action (stop the stream), not a destructive one, so it doesn't
  borrow `--at-red`.
- Minimum hit target: **measured 40px on the live page** (Gradio's `lg`
  buttons do *not* reach 44px by default, contrary to what a first pass of
  this doc assumed before actually checking) — fixed with an explicit
  `min-height: 44px` on `.gradio-container button`, re-measured after the
  fix.

## 8. Revision — "too AI/generic" correction

First pass of this redesign kept Gradio's `Soft` theme mostly as-shipped
(12-14px radii, soft drop shadows, emerald-on-white, geometric sans
headline) — functional, accessible, and *exactly* the visual signature of
every default Gradio/shadcn "AI tool" demo. Direct feedback mid-build: "çok
yapay zeka duruyor" (reads as too generic-AI). Correction, same token
system, different surface treatment:

- **Fraunces** (serif, `gr.themes.GoogleFont` doesn't cover a 3rd font slot,
  loaded via a plain `@import` in the CSS string instead) for the `<h1>`
  page title only — the single highest-leverage change, since a serif
  display headline is something the generic-AI-template pattern never uses.
- Radii cut hard: blocks 12px→6px, buttons 10/8px→6/5px, the header card
  14px→4px. Soft's shadows (`block_shadow`, `button_primary_shadow`, etc.)
  set to `none` across the board — flat fills + 1px borders instead of the
  "soft-UI" drop-shadow look.
- Corner-bracket motif kept and slightly enlarged (14px→16px arms) — it's
  the one genuinely project-specific signature already in use (references
  MOT bounding-box annotation, ties to the README/case-study identity) and
  reads *more* distinctive once it's not competing with rounded-everything.
- Primary green kept (still grounded in `viz/overlay_renderer.py`'s actual
  box color, §0) — the emerald-hue-as-generic-AI-signal comes from the
  *combination* with soft shadows/heavy rounding/geometric sans, not the
  hue alone, so swapping the accent color wasn't the fix; the surface
  treatment was.

### Icons
Three curated outline icons (hand-authored SVG, Phosphor-style — 1.6px
stroke, 20px viewBox, `currentColor`), one per tab, placed beside that tab's
intro line as a small section marker (Gradio's `gr.Tab` has no native `icon`
param in this version — checked via `inspect.signature`, not assumed):
- **Video yükle** → upload-into-tray glyph
- **PC kamerası** → camera-body glyph
- **IP kamera** → broadcast/antenna glyph
No emoji anywhere in the UI (checklist item — already true before this pass,
stays true after).

### Class picker (multiselect chips)
Existing Gradio `Dropdown(multiselect=True)` already satisfies the two
`ux`-domain findings that applied: chips wrap via flexbox instead of clipping
(verified — no fixed-height/overflow:hidden was ever applied to it), and the
component's own keyboard/ARIA handling is Gradio's, not custom-built (so the
"interactive chip needs a real button + aria-pressed" finding is Gradio's
responsibility, not this app's CSS layer).

### Status/summary panels
Monospace (`--at-mono`), `--at-surface` background, `--at-border` edge — a
quiet "readout" look distinct from the general Sans-serif chrome, consistent
with the case-study Artifact's own mono-for-data convention.

### Footer
Gradio's own footer (Runs / API / "Gradio ile oluşturulmuştur" / **Ayarlar**)
was *not* removed, despite an earlier draft of this pass hiding it outright
— opening it turned up a real, working **Görünüm Teması: Light/Dark/System**
toggle inside "Ayarlar." Hiding the footer would have hidden the one
in-app way to force dark mode regardless of OS setting. Kept, just shrunk
and muted (`opacity:0.6`, `--at-ink-muted` text) so it doesn't compete
visually with the app chrome.

### Focus states
Explicit `:focus-visible` outline added in `--at-green` (2px, 2px offset) on
interactive elements — this is the one place the 3:1 non-text rule is
actually enforced: `--at-green` vs. `--at-surface` measures 6.07:1 (light) /
5.68:1 (dark), both well past the 3:1 floor.

## 7. Pre-delivery checklist — each row actually re-checked on the live page

| Item | Status |
|---|---|
| No emojis as icons | ✅ SVG only |
| `cursor:pointer` on clickable elements | ✅ Gradio buttons/tabs default to this; not overridden |
| Hover transitions 150-300ms, no layout shift | ✅ 180ms, color/shadow only |
| Light mode text contrast ≥4.5:1 | ✅ computed, §2 |
| Dark mode text contrast ≥4.5:1 | ✅ computed, §2 (required fixing the button-text color, see §2) |
| Focus states visible | ✅ `:focus-visible` ring — confirmed via `el.matches(':focus-visible')` after real Tab-key navigation (a mouse click doesn't trigger `:focus-visible` by spec, so that's the correct test), returned the right green in both themes |
| `prefers-reduced-motion` | ✅ nothing motion-dependent to gate, see §5 |
| Responsive 375/768/1024/1440px | ✅ 768px: 2-column rows hold. 375px: collapses to 1 column, header/tabs/icons all reflow with no clipping |
| No horizontal scroll on mobile | ✅ initially **failed** — Gradio's own root container measured 386px in a 375px viewport (not from any custom element of ours); fixed with `overflow-x:hidden` on `html,body` and re-verified by attempting an actual `scrollTo` (stayed at `scrollX:0`), not just by re-reading the same `scrollWidth` metric that first caught it |
| Touch target ≥44px | ✅ initially **failed** — measured 40px on the real `size="lg"` button before assuming Gradio's default was compliant; fixed with `min-height:44px`, re-measured after |
| Label text actually uses the muted color token | ✅ initially **failed** — `THEME.set(block_label_text_color=...)` alone didn't work for any component with `info=` text (Gradio's Soft theme binds `span[data-testid="block-info"]` to the primary/accent color, not `--block-label-text-color`, for that specific span); fixed with an explicit CSS override, confirmed via `getComputedStyle` before and after |
| Dark-mode custom tokens (`--at-*`) actually flip in dark mode | ✅ initially **failed** — defining overrides on bare `:root` inside `@media(prefers-color-scheme:dark)` was dead code, because Gradio auto-rescopes custom `css=` selectors by prefixing them with `.gradio-container ... .contain`, and `X :root` can never match anything (the document root is never a descendant of another element); fixed by scoping the override to `.gradio-container` instead, and confirmed Gradio's *own* working mechanism is a JS-toggled `.dark` class on `<body>` (found by inspecting `document.body.className` after actually enabling dark mode, not by assumption) |

Four of the eleven rows above were **not** true on the first pass and only
became true after measuring, not after implementing — kept the failures
in this table instead of quietly fixing and rewriting history, since that's
the same discipline the rest of this project's README/research log runs on.
