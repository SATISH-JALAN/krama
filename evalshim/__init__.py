"""Python shim satisfying rag-local-eval-loop's target contract for Krama.

Deliberately NOT named `eval`: that suite ships its own top-level `eval`
package, and it puts this project's root on sys.path, so an `eval/` here would
shadow it and break the suite's own imports. Krama already has an eval/ folder
of its own, which is exactly why nothing from the suite was copied into this
repo -- it is invoked with --rag-root instead.

These modules hold no logic. Every real computation happens in Krama's own
TypeScript, reached over HTTP via eval-adapter/serve.ts.
"""
