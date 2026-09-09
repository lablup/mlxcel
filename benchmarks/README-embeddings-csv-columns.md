# Column alignment in the embedding CSVs

`scripts/bench_embeddings.py` declares a 21-column header but two of its three
`writerow` calls omitted `mlx_commit`, so a measured row carried 20 fields.
Reading such a file by column name shifts everything from `hardware` rightward:
the hardware string lands under `mlxcel_version`, the mlxcel revision under
`mlx_commit`, and `notes` comes back as `None`.

Nothing errors on either side. `csv.writer` does not check the count, and
`csv.DictReader` fills the missing tail with `None`, so the defect is invisible
unless someone compares the row width against the header.

Fixed in `f42d127d`. Files written before it:

| File | Rows | Width | Aligned |
|------|-----:|------:|---------|
| `metal_m5max_embeddings_2026-09-04.csv` | 90 | 21 | yes, this predates the column rename |
| `metal_m5max_embeddings_2026-09-06.csv` | 102 | 20 | **no** |
| `metal_m1ultra_embeddings_2026-09-09.csv` | 102 | 20 | **no**, superseded by a re-run |
| `metal_m5max_embeddings_2026-09-09.csv` | 102 | 21 | yes |

The misaligned files are kept as measured. Their timing columns, everything up
to and including `load_ms`, are unaffected: the shift begins after them. Read
them positionally against the header above rather than by name, or take the
provenance from the surrounding commit.
