"""RDKit cheminformatics tools — SMILES validation, property calculation, fingerprints.

TODO (Phase 2):
    - Implement molecular property calculation (logP, TPSA, MW, HBD, HBA)
    - Add Lipinski Rule-of-Five filter
    - Implement Morgan fingerprint generation
    - Add molecular similarity search
    - Support 2D/3D coordinate generation
"""

from __future__ import annotations


async def validate_smiles(smiles: str) -> bool:
    """Validate a SMILES string using RDKit.

    Args:
        smiles: SMILES string to validate.

    Returns:
        True if the SMILES is valid, False otherwise.

    Raises:
        NotImplementedError: RDKit integration not yet implemented.
    """
    raise NotImplementedError("SMILES validation will use rdkit.Chem.MolFromSmiles in Phase 2.")


async def calculate_properties(smiles: str) -> dict[str, float]:
    """Calculate molecular properties for a SMILES string.

    Args:
        smiles: Valid SMILES string.

    Returns:
        Dict with keys: molecular_weight, logp, tpsa, hbd, hba, rotatable_bonds.

    Raises:
        NotImplementedError: Property calculation not yet implemented.
    """
    raise NotImplementedError("Property calculation will use RDKit Descriptors module in Phase 2.")


async def compute_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> list[int]:
    """Compute a Morgan fingerprint for a molecule.

    Args:
        smiles: Valid SMILES string.
        radius: Morgan fingerprint radius (default 2 = ECFP4).
        n_bits: Fingerprint bit length.

    Returns:
        Binary fingerprint as list of 0/1 integers.

    Raises:
        NotImplementedError: Fingerprint computation not yet implemented.
    """
    raise NotImplementedError(
        "Fingerprint generation will use RDKit AllChem.GetMorganFingerprintAsBitVect in Phase 2."
    )


async def lipinski_filter(smiles: str) -> dict[str, bool | float]:
    """Check Lipinski's Rule of Five for a molecule.

    Args:
        smiles: Valid SMILES string.

    Returns:
        Dict with individual rule results and overall pass/fail.

    Raises:
        NotImplementedError: Lipinski filter not yet implemented.
    """
    raise NotImplementedError(
        "Lipinski filter will be implemented with RDKit descriptors in Phase 2."
    )


async def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    """Compute Tanimoto similarity between two molecules.

    Args:
        smiles_a: First SMILES string.
        smiles_b: Second SMILES string.

    Returns:
        Tanimoto similarity coefficient (0.0 to 1.0).

    Raises:
        NotImplementedError: Similarity computation not yet implemented.
    """
    raise NotImplementedError("Tanimoto similarity will use RDKit DataStructs in Phase 2.")
