Phase vs. microstructure features
Real Example:
A paper might say: "The alloy exhibits a dendritic structure with FCC phase"
This means:

Microstructure: dendritic (the shape/morphology)
Phase: FCC (the crystal structure)

The dendrites themselves are made OF a phase (could be FCC, BCC, etc.), but "dendritic" itself is not a phase.
Other Similar Terms to Exclude from Phase:
These are microstructural features, not phases:

Dendritic, equiaxed, columnar (grain morphology)
Lamellar, eutectic (phase arrangement)
Fine-grained, coarse-grained (grain size)
Precipitates, inclusions (feature type - though the precipitate itself has a phase)

What to Extract as Phase:
Only crystal structures and intermetallic compounds:

BCC, FCC, HCP
σ-phase, Laves phase, B2, L12
Amorphous (if considering non-crystalline)

Bottom line: If you're extracting phase information, skip "dendritic structure" - it's describing the shape, not the crystal structure.

Original langchain version
langchain                                0.3.14
langchain-chroma                         0.1.4
langchain-community                      0.3.1
langchain-core                           0.3.29
langchain-openai                         0.2.2
langchain-text-splitters                 0.3.4


Possible optimization:
As method described in https://doi.org/10.1145/3589335.3651245.
Set up multiple partitions, use inforamtion entropy updated by bayesian optimization to optimize.
