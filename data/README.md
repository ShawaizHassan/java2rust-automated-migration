# Data Notes

- **Schema:** each JSONL line contains `Java` and `Rust` objects and at least `bug_source_code` keys.
- **Large data:** keep out of Git. Use **DVC** (recommended) or **Git LFS**.
- **Samples:** tiny examples live in `data/samples/` for tests and CI.
