# Optigon brand assets

The Optigon mark and lockups. Flat SVG, 120-unit grid, round caps, no gradients.
Sibling set to ForgeDB and `typescript-to-rust`.

- **Mark:** a rounded hexagon of six neutral facets darkening away from one amber
  *lit* face — the winning implementation. The graphite ramp deliberately owns no
  hue, so it never collides with a domain colour.
- **Palette:** lit facet amber `#fbbf24` (the one brand colour) · graphite ramp
  `#a1a1aa` `#71717a` `#52525b` `#3f3f46` `#27272a` · ink `#0f1115`.
- **Type:** wordmark Space Grotesk 600; tagline JetBrains Mono 400, tracked +2.4.

## Files

| file | use |
|---|---|
| `optimark-primary.svg` | icon / favicon / package icon, any background |
| `optimark-mono.svg` | single-ink on light |
| `optimark-mono-inverse.svg` | single-ink on dark |
| `optigon-horizontal-{light,dark,mono}.svg` | mark + wordmark + descriptor |
| `optigon-stacked-{light,dark,mono}.svg` | centred, for square slots |
| `optigon-horizontal-{light,dark}.outlined.svg` | horizontal lockup with the wordmark converted to paths (see below) |

Mono lockups fill with `currentColor` — set `color` on the parent and they follow.

## Notes

The `-light` / `-dark` lockups are transparent-background and pick the wordmark
ink for that theme, so they pair with a `<picture>` + `prefers-color-scheme`
switch (as the root README does).

Lockup wordmarks are **live `<text>`** with system-font fallback stacks (Space
Grotesk → `system-ui`, JetBrains Mono → `monospace`), kept editable on purpose.
They render cleanly everywhere but are only pixel-identical to the intended
typefaces where those fonts are installed. The icon marks are pure geometry and
need no conversion.

For contexts that must render the intended typefaces without the fonts installed
(the root README hero, npm/crates package pages, social embeds), use the
`*.outlined.svg` copies: the wordmark (Space Grotesk 600) and tagline (JetBrains
Mono 400) are baked to `<path>` — HarfBuzz-shaped from the upstream OFL fonts, so
kerning matches a browser — with no `<text>` or `font-family` left. The editable
`<text>` originals are the source of truth; regenerate the outlined copies from
them after any wordmark edit.
