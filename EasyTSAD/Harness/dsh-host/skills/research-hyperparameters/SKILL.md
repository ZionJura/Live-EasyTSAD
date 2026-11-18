---
name: research-hyperparameters
description: Extract the official TSAD model and runtime parameters into config.toml and a cited reproduction/hyperparameters.json artifact.
---

Read the downloaded paper and assigned `upstream/` source. Identify the official training entry point, effective defaults and configuration overrides. Research only the assigned method; the interface and environment branches run independently.

Write the method's `config.toml` and `reproduction/hyperparameters.json` with matching effective defaults. IMPLEMENT reads them and may correct a concrete mismatch with evidence. The JSON should give each effective parameter's value, origin (`paper`, `source`, or an explicitly recorded fallback), and source evidence with a page/section or repository file/line. Include model structure, preprocessing, window/stride, batch size, optimizer, learning rate, epochs, seed, scoring direction and alignment choices when applicable. Explain discrepancies between paper and source and record the selected behavior.

Use parameters from the official method and the requested protocol. Do not tune against test labels or test performance, invent epochs for an untrained method, edit the interface/environment branch outputs, or launch training. Return the written paths and any unresolved parameters.
