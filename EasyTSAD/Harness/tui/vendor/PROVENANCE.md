# Original DeepSeek Harness TUI

Source: https://github.com/deepseek-ai/deepseek-harness/tree/7248b5ec8f8769f882f12fd521504fa48e97bcf3/packages/ui/tui

Pinned revision: `7248b5ec8f8769f882f12fd521504fa48e97bcf3`. Original source is preserved under `deepseek-harness/` with its BSD-3-Clause LICENSE. Recovered from the complete deletion patch of commit `10bb9cbf4a22b5095bb9ff04d1425907af8f08af`.

The adapted renderer in `src/upstream/` removes the old Cordis/DSH runtime coupling. It does not run another agent runtime. The original pi-tui 0.80.7 patch is preserved under `patches/`.

## Adaptation boundary

| Area | Local adaptation |
| --- | --- |
| Display components | Extract original text, palette, Markdown, XML preview, transcript and file-diff rendering into `src/upstream/`; preserve upstream files separately above. |
| Message DTOs | Replace old DSH type imports with local display-only content/tool/diff types and normalize remote HTTP history in the client. |
| HintEditor | Retain original borderless prompt and add sanitization for complete or fragmented bracketed pastes, including OSC/CSI controls. |
| Remote tool presenters | Add client-only shell/file/search/fetch/TSAD presenters and extract explicit diffs from arguments, results or result metadata. No tools execute in these presenters. |
| Runtime and shortcuts | Do not load the original Cordis plugin, Agent lifecycle, model/resume selectors or autocomplete providers. Keep the remote Host as the single Agent runtime and the four basic interactions. |
| pi-tui dependency patch | Preserve the official `@earendil-works__pi-tui@0.80.7.patch` unchanged; it supplies borderless Editor prompt prefixes and correct continuation wrapping. |

## Dependency license sources

The exact published versions identify their source revisions through npm `gitHead`. Their repository license texts are preserved, unmodified after provenance headers, in:

- `licenses/pi-tui-0.80.7.txt`: MIT, Mario Zechner; npm gitHead `818d67457cdd6b60bce6b121d16b23141c252dd8`.
- `licenses/saxes-6.0.0.txt`: the complete source LICENSE, including ISC notices for saxes contributors and the original sax authors, plus its retained historical notice; npm gitHead `211fa0ebec9b628affc09219199639887174bfc3`.

Each file records the immutable source URL and the SHA-256 of the original license body. Include these notices with bundled distributions even where the npm package omitted a standalone LICENSE file.
