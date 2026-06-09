```markdown
# Design System Specification: The Kinetic Intelligence Framework

## 1. Overview & Creative North Star: "The Digital Curator"
This design system is built to move beyond the "SaaS dashboard" trope. The Creative North Star is **The Digital Curator**—a philosophy that treats complex data as high-end editorial content. We reject the cluttered, boxy layouts of traditional analytics. Instead, we embrace a high-tech, cinematic aesthetic characterized by tonal depth, intentional asymmetry, and "breathing" data.

The system breaks the "template" look by utilizing wide gutters, overlapping glass layers, and a high-contrast typographic scale that guides the eye through dense information with the authority of a premium scientific journal.

---

## 2. Colors: Tonal Depth & The "No-Line" Rule
Our palette is anchored in deep, atmospheric neutrals that provide a canvas for vibrant, glowing accents.

### The "No-Line" Rule
**Explicit Instruction:** Designers are prohibited from using 1px solid borders to section content. Hierarchy must be defined through background color shifts or tonal transitions.
*   **Surface Transition:** Place a `surface-container-low` section against a `surface` background to define a zone.
*   **Inner Nesting:** Use `surface-container-highest` for high-priority interactive elements.

### The Glass & Gradient Rule
To achieve a "signature" look, we move away from flat surfaces:
*   **Signature Textures:** Main CTAs and Hero backgrounds should utilize a linear gradient from `primary` (#97a9ff) to `secondary` (#c99af8) at a 135° angle.
*   **Glassmorphism:** Floating panels (modals, popovers) must use `surface-container` colors at 70% opacity with a `24px` backdrop-blur. This softens the UI and prevents the "pasted-on" look.

### Core Token Map
| Token | Hex | Role |
| :--- | :--- | :--- |
| `surface` | #0c0e14 | The foundation; the deepest layer of the app. |
| `primary` | #97a9ff | Actionable elements and brand highlights. |
| `secondary` | #c99af8 | Secondary accents and cognitive grouping. |
| `tertiary` | #c3ffcd | Success states and positive data trends. |
| `error` | #ff6e84 | Critical alerts and negative data trends. |

---

## 3. Typography: Editorial Authority
We utilize a dual-typeface system to balance technical precision with high-end aesthetic appeal. 

*   **Display & Headlines (Manrope):** Chosen for its geometric modernism. Used for large metrics and section headers to provide a "signature" feel.
*   **Body & Labels (Inter):** Chosen for its exceptional legibility in dense data environments.

### Typography Scale
*   **Display-LG (Manrope, 3.5rem):** High-impact hero metrics.
*   **Headline-SM (Manrope, 1.5rem):** Primary section headers.
*   **Title-SM (Inter, 1rem, Medium):** Card titles and data group labels.
*   **Body-MD (Inter, 0.875rem):** General interface text and descriptions.
*   **Label-SM (Inter, 0.6875rem):** Micro-copy and chart legends.

---

## 4. Elevation & Depth: Tonal Layering
In this design system, elevation is a function of light and color, not structure.

*   **The Layering Principle:** Stacking surface tiers creates a "soft lift." A `surface-container-lowest` card placed on a `surface-container-low` background creates a natural recession, ideal for background data grids.
*   **Ambient Shadows:** For elements that must "float" (e.g., Command Palettes), use a 40px blur shadow at 8% opacity using the `on-surface` color. Avoid pure black shadows.
*   **The "Ghost Border" Fallback:** If a border is required for accessibility, use the `outline-variant` token at **15% opacity**. 100% opaque borders are strictly forbidden.

---

## 5. Components: The Elite Primitives

### Cards & Data Containers
*   **Rule:** Forbid divider lines. Separate content using vertical white space (use the `1.5rem` spacing token) or subtle background shifts.
*   **Styling:** 12px (`xl`) corner radius. High-priority cards use a subtle 1px "Ghost Border" top-edge highlight to mimic light hitting a glass edge.

### Buttons
*   **Primary:** A vibrant gradient background (`primary` to `secondary`) with `on-primary` text. No border.
*   **Secondary:** `surface-container-high` background with `primary` text.
*   **Tertiary:** Ghost style. No background, `on-surface-variant` text, shifting to `on-surface` on hover.

### Inputs & Search
*   **Visuals:** Inputs should be treated as "wells." Use `surface-container-lowest` with a subtle inset shadow to create a sense of depth.
*   **Focus State:** Transition the background to `surface-container-high` and apply a 2px `primary` glow.

### Data Visualization (Signature Charting)
*   **Palettes:** Use a sequence of `primary`, `secondary`, and `tertiary` for multi-series data. 
*   **Interactivity:** On hover, dim non-active series to 20% opacity rather than using heavy tooltips.

---

## 6. Do's and Don'ts

### Do
*   **Do** use asymmetrical layouts (e.g., a wide 8-column main view and a 3-column metadata sidebar) to create visual interest.
*   **Do** use `letter-spacing: -0.02em` on Display and Headline styles for a more "designed" editorial feel.
*   **Do** prioritize "negative space" as a functional tool to reduce cognitive load during data analysis.

### Don't
*   **Don't** use 100% opaque white text. Use `on-surface` (#e5e4ed) for headers and `on-surface-variant` (#aaaab3) for body text to reduce eye strain in dark mode.
*   **Don't** use standard "drop shadows" on cards. Use tonal background shifts instead.
*   **Don't** use sharp 0px corners. Every element must feel machined and polished, adhering to the `8px` to `12px` radius scale.