CATEGORIZE_DSPY = """You are analyzing the experimental section of a high entropy alloys (HEAs) research paper. Your task is to determine whether the author has distinctly identified each fabricated material.

**Task**: Return True if ALL materials are distinctly identified, False otherwise.

**Material Identification Requirements**:
- **Chemical composition**: Specific elemental ratios (e.g., "Mn0.2CoCrNi", "Al0.3CoCrFeNi0.5")
  - General formulas like "MnxCoCrNi" are NOT acceptable
- **Explicit symbol**: Clear identifier like "A0", "HEA-1", "Sample-700C"
  - Descriptive phrases without explicit symbols are NOT acceptable

**Evaluation Rules**:

**Case 1 - Single processing route, single sample**:
- True: If chemical composition is provided
- False: If composition is missing

**Case 2 - Single processing route, multiple samples with different compositions**:
- True: If every sample has either specific composition OR explicit symbol
- False: If any sample lacks both composition and symbol

**Case 3 - Multiple processing routes, same composition**:
(Different methods, parameters, temperatures, durations, etc.)
- True: If every sample is explicitly labeled with unique identifiers
- False: If any sample lacks clear labeling

**Case 4 - Multiple processing routes, different compositions**:
- True: If every sample has either specific composition OR explicit symbol
- False: If any sample lacks both composition and symbol"""
