import re
from typing import Optional, Tuple


def normalize_value_with_unit(value_str: str) -> Optional[Tuple[float, str]]:
    """
    Normalize a value with unit expression to a single float with its unit.
    
    Handles forms like:
      - ranges: "10-20 nm" -> average -> 15.0 nm
      - inequalities: "> 30 nm" -> 30.0 nm (returns central threshold)
      - approx: "~ 15 nm" or "≈15 µm" -> 15.0 nm/µm
      - plus/minus: "15 ± 2 nm" or "15+/-2 µm" -> 15.0 nm/µm
    
    Args:
        value_str: String containing a numeric value and unit
        
    Returns:
        Tuple of (normalized_value, unit) or None if no parsable value found
    """
    if not value_str or not isinstance(value_str, str):
        return None
    
    # Normalize the input string
    normalized = _normalize_characters(value_str.strip())
    
    # Try to extract value and unit using various patterns
    result = (
        _try_range_pattern(normalized) or
        _try_plusminus_pattern(normalized) or
        _try_inequality_pattern(normalized) or
        _try_approx_pattern(normalized) or
        _try_simple_pattern(normalized)
    )
    
    return result


def _normalize_characters(s: str) -> str:
    """Normalize various Unicode characters to standard ASCII equivalents."""
    # Normalize different dash/minus characters
    s = re.sub(r'[−–—‐‑‒–—―]', '-', s)
    
    # Normalize approximation symbols
    s = re.sub(r'[≈≃∼⁓]', '~', s)
    
    # Normalize plus-minus symbols
    s = re.sub(r'[±]', '+-', s)
    
    # Normalize greater/less than symbols
    s = re.sub(r'[＞]', '>', s)
    s = re.sub(r'[＜]', '<', s)
    s = re.sub(r'[≥≧]', '>=', s)
    s = re.sub(r'[≤≦]', '<=', s)
    
    # Normalize multiplication/times symbols
    s = re.sub(r'[×·⋅]', '*', s)
    
    # Normalize division symbols
    s = re.sub(r'[÷]', '/', s)
    
    return s


def _extract_unit(s: str, after_pos: int = 0) -> str:
    """Extract unit from string after a given position."""
    remainder = s[after_pos:].strip()
    # Extract common unit pattern (letters, potentially with ^, -, /, numbers for exponents)
    unit_match = re.match(r'([a-zA-Zµμ°]+(?:[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉\^\-\/0-9]*)?)', remainder)
    return unit_match.group(1).strip() if unit_match else ''


def _try_range_pattern(s: str) -> Optional[Tuple[float, str]]:
    """Handle range patterns like '10-20 nm' or '5 - 15 µm'."""
    # Pattern: number - number unit
    pattern = r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*([a-zA-Zµμ°]+.*?)$'
    match = re.search(pattern, s)
    if match:
        low = float(match.group(1))
        high = float(match.group(2))
        unit = match.group(3).strip()
        return ((low + high) / 2, unit)
    return None


def _try_plusminus_pattern(s: str) -> Optional[Tuple[float, str]]:
    """Handle plus/minus patterns like '15 ± 2 nm' or '15+/-2 µm'."""
    # Pattern: number +- number unit or number ± number unit
    pattern = r'(\d+\.?\d*)\s*(?:\+-|±)\s*\d+\.?\d*\s*([a-zA-Zµμ°]+.*?)$'
    match = re.search(pattern, s)
    if match:
        value = float(match.group(1))
        unit = match.group(2).strip()
        return (value, unit)
    return None


def _try_inequality_pattern(s: str) -> Optional[Tuple[float, str]]:
    """Handle inequality patterns like '> 30 nm' or '< 10 µm'."""
    # Pattern: [><=] number unit
    pattern = r'[<>]=?\s*(\d+\.?\d*)\s*([a-zA-Zµμ°]+.*?)$'
    match = re.search(pattern, s)
    if match:
        value = float(match.group(1))
        unit = match.group(2).strip()
        return (value, unit)
    return None


def _try_approx_pattern(s: str) -> Optional[Tuple[float, str]]:
    """Handle approximation patterns like '~ 15 nm' or '≈15 µm'."""
    # Pattern: ~ number unit
    pattern = r'~\s*(\d+\.?\d*)\s*([a-zA-Zµμ°]+.*?)$'
    match = re.search(pattern, s)
    if match:
        value = float(match.group(1))
        unit = match.group(2).strip()
        return (value, unit)
    return None


def _try_simple_pattern(s: str) -> Optional[Tuple[float, str]]:
    """Handle simple patterns like '15 nm' or '3.5µm'."""
    # Pattern: number unit
    pattern = r'(\d+\.?\d*)\s*([a-zA-Zµμ°]+.*?)$'
    match = re.search(pattern, s)
    if match:
        value = float(match.group(1))
        unit = match.group(2).strip()
        return (value, unit)
    return None


# Example usage and tests
if __name__ == "__main__":
    test_cases = [
        "10-20 nm",
        "5 - 15 µm",
        "> 30 nm",
        "< 10 µm",
        ">= 50 nm",
        "~ 15 nm",
        "≈15 µm",
        "15 ± 2 nm",
        "15+/-2 µm",
        "25 nm",
        "3.5µm",
        "100 mg/mL",
        "5−10 nm",  # Unicode minus
        "∼20 µm",   # Unicode tilde
        "0.29 µm",
        "invalid",
        "",
        None
    ]
    
    print("Testing value normalization:\n")
    for test in test_cases:
        result = normalize_value_with_unit(test)
        print(f"Input: {repr(test):25} -> Output: {result}")