# Design — Color, Theme, Typography

**[Guessing]** — no brand guidelines were given, so these are reasonable defaults for an EdTech translation tool. Treat this as a starting proposal to approve/change in Week 1, not a locked spec.

## 1. Design Principles

- Captions must be legible at a glance from across a lecture hall — prioritize contrast and size over decoration
- Minimal cognitive load: the live caption is the hero element, everything else is secondary chrome
- Calm, not flashy — this is a classroom tool, not a marketing site

## 2. Color Palette

| Role | Color | Hex | Use |
| --- | --- | --- | --- |
| Primary | Deep Indigo | `#3730A3` | Header, primary buttons, active states |
| Primary Light | Indigo 100 | `#E0E7FF` | Backgrounds, hover states |
| Secondary/Accent | Warm Amber | `#F59E0B` | Live/recording indicator, highlights, feedback prompts |
| Success | Emerald | `#10B981` | Positive feedback, connected status |
| Error | Rose | `#E11D48` | Errors, disconnected status, validation |
| Neutral Dark | Slate 900 | `#0F172A` | Primary text |
| Neutral Mid | Slate 500 | `#64748B` | Secondary text, timestamps |
| Neutral Light | Slate 50 | `#F8FAFC` | Page background |
| Surface | White | `#FFFFFF` | Cards, caption box background |

- Dark mode: invert Neutral Dark/Light and Surface roles; keep Primary/Accent as-is with slightly reduced saturation — flag as a Week 3+ nice-to-have, not MVP

## 3. Typography

| Use | Font | Notes |
| --- | --- | --- |
| UI / Body | Inter | Excellent multilingual glyph coverage (Latin, Devanagari via extended sets), good for a translation app |
| Live captions | Inter, larger weight | 18–24px minimum for readability from a distance; test with actual target-language scripts (Devanagari/Telugu glyphs can render narrower/wider than Latin — verify line height doesn't clip) |
| Headings | Inter, 600/700 weight | No separate display font needed — keep it simple |
| Monospace (if needed, e.g. session codes) | JetBrains Mono | Session join codes, timestamps in dev/debug views |

**Note**: If regional scripts (Devanagari, Telugu, etc.) render poorly in Inter, fall back to a Google Noto variant (e.g., Noto Sans Devanagari) for that specific text — don't force one font family across all scripts at the cost of readability.

## 4. Spacing & Layout

- 8px base spacing unit (Tailwind default scale is fine — don't invent a custom one)
- Caption box: fixed max-width (~800px) centered, generous padding, high line-height (1.6+) for readability
- Mobile/tablet breakpoint: stack language selector above caption box instead of side-by-side

## 5. Key UI States to Design For

- Listening/recording (pulsing amber indicator)
- Translating (subtle loading state per segment, not a full-page spinner — translation is continuous)
- Connected / reconnecting / disconnected (WebSocket status, visible but not alarming unless actually disconnected)
- Empty state (session not started yet)
- Feedback given (brief confirmation, non-blocking)
