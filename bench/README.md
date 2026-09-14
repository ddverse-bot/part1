# bench

Benchmark and evaluation for octreg 1.0. It lives in the repository but is not part of the installed package, and it is still being built. The plan is in [../docs/design/spec_v1.0.md](../docs/design/spec_v1.0.md) (code_spec, bench/ and validation_spec).

`pairs/` will hold one TOML file per pair with file facts only (paths, sha256, spacing and its source, annotations, reference transforms), never method parameters.
