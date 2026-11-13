# Sources and licenses

- **MAESTRO:** https://github.com/penn-pal-lab/maestro at `e60d326233a238ddc496bf7a08acb7a278e9e66d`. Included prompt/reference runner carry the original MIT terms in `maestro/upstream/LICENSE` and upstream attribution in `maestro/upstream/NOTICE`. Its native robot stack is not included.
- **CABTO:** https://github.com/DIDS-EI/CABTO at `4cc89d501080015334f7e331471724b7cf6da7bc` (`last`). The pinned README declares MIT, while no root license file was present. Its planner source is retrieved by the SHA-verified fetcher, not bundled. `cabto_*/source_audit/LICENSE_NOTICE.md` preserves this distinction.
- **OpenPI:** https://github.com/Physical-Intelligence/openpi at `fdc03f527881cdfc8ae1a168ed6a20c60edbbbcc`; install upstream under its Apache-2.0 terms. Checkpoint distribution has its own model terms. OpenPI core code and checkpoint weights are not copied here.
- **LIBERO:** https://github.com/Lifelong-Robot-Learning/LIBERO; install the benchmark and assets under its upstream terms. No simulator source/assets are vendored.
- **Qwen3.5-9B:** https://huggingface.co/Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; obtain weights and follow the model repository's terms. No weights are redistributed.
- Benchmark adapters, native runtime and original assembly artifact are part of this project's reproduction code. Their exported-file provenance is recorded in `export_provenance.json`; the enclosing repository governs project-code licensing.
