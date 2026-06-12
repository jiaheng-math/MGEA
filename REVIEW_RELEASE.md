# Anonymous Review Release Scope

This archive is a scoped anonymous-review artifact for MGEA.

The package keeps:

- environment setup files
- dataset preparation scripts
- retrieval backend wrappers
- non-sensitive QA generation/evaluation utilities
- configuration examples

The package withholds:

- probe-aware routing feature extraction
- router training and out-of-fold probability generation
- main experiment orchestration
- marginal evidence-slot allocation
- deep evidence materialization for slot allocation
- conversion from routing/slot decisions to final generation inputs
- router-budget and slot-ablation plotting/summarization workflows

Those withheld components contain the main unpublished method and are preserved
as explicit placeholder modules in this review artifact. They are not silently
omitted or replaced with fake implementations.
