"""Taxonomy parsing utilities."""

from typing import Dict, Optional


def parse_taxon_string(taxon: str) -> Dict[str, str]:
    """
    Parse a taxonomic lineage string into structured taxonomy levels.

    Args:
        taxon: Taxonomic lineage string (e.g., "d__Bacteria; p__Pseudomonadota; c__Gammaproteobacteria")

    Returns:
        Dictionary with taxonomy levels (Domain, Phylum, Class, Order, Family, Genus, Species)
    """
    if not taxon:
        return {}

    # Define the standard taxonomic hierarchy prefixes
    level_prefixes = {
        "d__": "Domain",
        "p__": "Phylum",
        "c__": "Class",
        "o__": "Order",
        "f__": "Family",
        "g__": "Genus",
        "s__": "Species",
    }

    taxonomy = {}

    # Split the taxon string by semicolons and process each level
    levels = [level.strip() for level in taxon.split(";")]

    for level in levels:
        for prefix, level_name in level_prefixes.items():
            if level.startswith(prefix):
                # Extract the taxonomic name (remove prefix)
                # name = level[len(prefix):]
                # if name:  # Only add if name is not empty
                taxonomy[level_name] = level
                break

    return taxonomy


def format_taxonomy_display(taxonomy: Dict[str, str]) -> str:
    """
    Format parsed taxonomy for display purposes.

    Args:
        taxonomy: Dictionary with taxonomy levels

    Returns:
        Formatted taxonomy string for display
    """
    if not taxonomy:
        return "Unknown"

    # Standard hierarchy order
    hierarchy = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

    # Find the deepest level available
    deepest_level = None
    for level in reversed(hierarchy):
        if level in taxonomy and taxonomy[level]:
            deepest_level = level
            break

    if deepest_level:
        return f"{taxonomy[deepest_level]} ({deepest_level})"

    return "Unknown"


def get_taxonomy_level(taxonomy: Dict[str, str], level: str) -> Optional[str]:
    """
    Get a specific taxonomy level from the parsed taxonomy.

    Args:
        taxonomy: Dictionary with taxonomy levels
        level: Taxonomy level to retrieve (Domain, Phylum, Class, Order, Family, Genus, Species)

    Returns:
        Taxonomy name at the specified level, or None if not available
    """
    return taxonomy.get(level)


def get_deepest_taxonomy_level(taxonomy: Dict[str, str]) -> Optional[str]:
    """
    Get the deepest (most specific) taxonomy level available.

    Args:
        taxonomy: Dictionary with taxonomy levels

    Returns:
        The deepest available taxonomy level name
    """
    hierarchy = ["Species", "Genus", "Family", "Order", "Class", "Phylum", "Domain"]

    for level in hierarchy:
        if level in taxonomy and taxonomy[level]:
            return level

    return None


def calculate_taxonomy_consensus(
    taxonomies: list[Dict[str, str]], min_frequency: int = 3
) -> Dict[str, str]:
    """
    Calculate taxonomic consensus from multiple taxonomy results.

    Args:
        taxonomies: List of parsed taxonomy dictionaries
        min_frequency: Minimum frequency required for consensus

    Returns:
        Consensus taxonomy dictionary
    """
    from collections import Counter

    if not taxonomies:
        return {}

    hierarchy = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    consensus = {}

    for level in hierarchy:
        # Get all values for this level
        values = [tax.get(level) for tax in taxonomies if tax.get(level)]

        if not values:
            continue

        # Count frequencies
        count = Counter(values)
        most_common_val, freq = count.most_common(1)[0]

        # Only include in consensus if frequency meets threshold
        if freq >= min_frequency:
            consensus[level] = most_common_val
        else:
            # Stop at first level without consensus
            break

    return consensus
