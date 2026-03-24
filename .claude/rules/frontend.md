---
paths:
  - "app/web/**"
---
# Frontend Rules

## Color Palette
NEVER use generic Tailwind grays (`gray-*`, `neutral-*`) or pure black/white buttons.

Use these colors from the design system:

**Highlight (use sparingly):**
- Amber: `#D97706`, Pressed: `#B45309`, Tint: `#FEF3C7`

**Neutrals:**
- White: `#FFFFFF`
- Surface: `#FAFAF8`
- Muted: `#F3F3EF`
- Border: `#E8E8E3`
- Subtle: `#8A8A82`
- Secondary: `#5C5C56`
- Foreground: `#1A1A18`

**Semantic:** Success `#2F7D3E`, Error `#C73A3A`, Warning `#B78A0A`

Primary buttons: Foreground bg (`#1A1A18`) with white text. Surfaces: `#FAFAF8`. Borders: `#E8E8E3`.

## State Management
Never use localStorage for important state. Persist server-side via authenticated endpoints.
