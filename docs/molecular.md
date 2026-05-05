# Molecular Orbitals

Compute HOMO and LUMO frontier orbital energies from a SMILES string. Results include the
HOMO–LUMO gap and per-atom orbital density weights.

```python
# Resveratrol
result = client.homo.run("Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1")

print(result.homo_E_eV)    # e.g. -5.88 eV
print(result.lumo_E_eV)    # e.g. -1.40 eV
print(result.gap_eV)       # HOMO-LUMO gap in eV
print(result.heavy_symbols) # ['C', 'C', ..., 'O', 'O']
print(result.homo_density)  # per-heavy-atom HOMO weight
```

---

## Basis set and functional

```python
# Use a larger basis set and a different functional
result = client.homo.run(
    "c1ccccc1",   # benzene
    basis="6-31g",
    xc="pbe0",
)
```

---

## HomoResult fields

| Field | Type | Description |
|---|---|---|
| `homo_E_eV` | `float` | HOMO energy in eV |
| `lumo_E_eV` | `float` | LUMO energy in eV |
| `gap_eV` | `float` | HOMO–LUMO gap in eV |
| `homo_density` | `list[float]` | Per-heavy-atom HOMO orbital density |
| `lumo_density` | `list[float]` | Per-heavy-atom LUMO orbital density |
| `heavy_symbols` | `list[str]` | Element symbol for each heavy atom |
| `n_occ` | `int` | Number of occupied orbitals |
| `basis` | `str` | Basis set used (default: `sto-3g`) |
| `xc` | `str` | Exchange-correlation functional (default: `b3lyp`) |
