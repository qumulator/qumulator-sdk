# Contributing to qumulator-sdk

Thank you for your interest in contributing!

## Getting started

```bash
git clone https://github.com/qumulator-io/qumulator-sdk.git
cd qumulator-sdk
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

All tests mock the HTTP layer — no live API key is required.

## Pull requests

- One logical change per PR.
- All tests must pass.
- Add a test for any new behaviour.
- Keep public API surface consistent with the existing style in `circuit.py`
  and `resources.py`.

## Reporting issues

Open a GitHub issue. Include the SDK version (`pip show qumulator-sdk`),
Python version, and a minimal reproducible example.

## Licence

By contributing you agree that your changes will be released under the
[MIT Licence](LICENSE).
