# Configuration

`paper_v1.json` is the frozen manuscript protocol. The command-line `paper`
mode reads only its `detector` object; the remaining sections document the
input, consolidation, and validation rules that must be applied by a complete
paper replay.

Changing this file creates a new experimental protocol. Do not overwrite it
when exploring alternative windows, null models, or matching tolerances.

`data.example.json` documents the authorized local-data mount used by the full
reproduction runner. Copy it to the ignored `data.local.json` before use.

`figure_sources.example.json` contains placeholders for the two separately
licensed optical scenes used by Figure 6. Copy it to the ignored
`figure_sources.local.json` and edit only that local file. No workstation path
or third-party image is committed.
