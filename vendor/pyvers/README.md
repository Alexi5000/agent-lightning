# Pyvers compatibility package

This is a maintained, metadata-only compatibility build of upstream pyvers
0.1.0, identified as `0.1.0+agl.1`. The three Python modules and MIT license
are byte-identical to the published wheel recorded in `provenance.json`.
The upstream module's `__version__` remains `0.1.0`; installed distribution
metadata identifies this local build.

The only dependency change expands `packaging>=25,<26` to `packaging>=25,<27`.
Pyvers uses only `packaging.version.parse` and its release/comparison API.
Compatibility tests exercise version boundaries, dispatch, backend selection,
and actual TensorDict operations with packaging 26.3. The normal VERL tests
and type checking remain required.

This permits patched virtualenv in the CPU development environment while
retaining TensorDict 0.10.0 and its macOS 14 wheels. No dependency override
is used. The local source is selected by uv; it does not change the runtime
dependencies of the published agentlightning wheel. Install from a complete
checkout when using the `verl-cpu` group.

The fork maintainers own this compatibility metadata and its tests. Remove
the local source when the supported VERL/TensorDict graph accepts a released
pyvers that supports packaging 26, after rerunning the same checks. Do not
update the vendored files or hashes without reviewing the upstream diff.
