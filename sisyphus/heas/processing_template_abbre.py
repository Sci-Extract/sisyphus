arc_melting = """arc melting: atmosphere: "", remelting times: "" """
induction_melting = """induction melting: atmosphere: "", remelting times: "" """
levitation_melting = """levitation melting: atmosphere: "", remelting times: "" """
annealed = """annealed: temperature: "", duration: "", atmosphere: "" """
aged = """aged: temperature: "", duration: "", atmosphere: "" """
homogenized = """homogenized: temperature: "", duration: "", atmosphere: "" """
solution_treated = """solution treated: temperature: "", duration: "", atmosphere: "" """
heat_treated = """heat treated: temperature: "", duration: "", atmosphere: "" """
quenching = """quenching: medium: "" """
cooling = """cooling: medium: "", rate: "" """
high_pressure_torsion = """high pressure torsion: pressure: "", turns: "", rotation speed: "" """
cold_rolled = """cold rolled: reduction: "", temperature: "" """
hot_rolled = """hot rolled: reduction: "", temperature: "" """
forged = """forged: temperature: "" """
hot_isostatic_pressing = """hot isostatic pressing: pressure: "", temperature: "", duration: "" """
mechanical_alloying = """mechanical alloying: duration: "", ball size: "", ball to powder ratio: "", rotation speed: "" """
spark_plasma_sintering = """spark plasma sintering: temperature: "", pressure: "", duration: "" """
hot_pressing = """hot pressing: temperature: "", pressure: "", duration: "" """
additive_manufacturing = """additive manufacturing: scan speed: "", laser power: "" """
gas_atomization = """gas atomization: pressure: "", gas velocity: "", gas: "" """

""""
Processing Type | Processing Name | Parameters
---|---|---
Liquid-State Processing (LSP) | arc melting | atmosphere (Ar/vacuum), remelting times, current, cooling method
Liquid-State Processing (LSP) | vacuum arc melting | remelting times, cooling method, atmosphere pressure
Liquid-State Processing (LSP) | vacuum induction melting | atmosphere, crucible type, melting duration, remelting times, temperature
Liquid-State Processing (LSP) | induction melting | atmosphere, crucible type, melting duration, remelting times
Liquid-State Processing (LSP) | directional solidification | withdrawal velocity, temperature gradient, cooling rate
Liquid-State Processing (LSP) | infiltration | pressure/capillary force, temperature, matrix phase
Liquid-State Processing (LSP) | electromagnetic stirring | magnetic field strength, stirring time, remelting times
Solid-State Processing (SSP) | mechanical alloying | milling duration, ball size, ball-to-powder ratio, rotation speed, atmosphere
Solid-State Processing (SSP) | gas atomization | gas type, gas pressure, gas velocity, nozzle diameter, superheat temperature
Solid-State Processing (SSP) | water atomization | water pressure, water velocity, nozzle configuration, particle size range
Solid-State Processing (SSP) | cold uniaxial pressing | compaction pressure, die material, dwell time
Solid-State Processing (SSP) | vacuum sintering | temperature, pressure, duration, heating rate, cooling rate, atmosphere
Solid-State Processing (SSP) | spark plasma sintering | temperature, pressure, duration, heating rate, current
Solid-State Processing (SSP) | hot pressing | temperature, pressure, duration, atmosphere
Solid-State Processing (SSP) | vacuum hot pressing | temperature, pressure, duration, vacuum level
Solid-State Processing (SSP) | hot isostatic pressing | temperature, pressure, duration, atmosphere
Heat Treatment | annealing | temperature, duration, atmosphere, cooling method
Heat Treatment | aging | temperature, duration, atmosphere
Heat Treatment | homogenization | temperature, duration, atmosphere
Heat Treatment | solution treatment | temperature, duration, atmosphere, quenching medium
Heat Treatment | heat treatment | temperature, duration, atmosphere
Cooling Methods | quenching | quenching medium (water/oil/gas), temperature
Cooling Methods | furnace cooling | cooling rate, atmosphere
Cooling Methods | air cooling | cooling rate, environment
Cooling Methods | supercooling | cooling rate, undercooling degree
Mechanical Processing | cold rolling | reduction ratio, number of passes, temperature
Mechanical Processing | hot rolling | reduction ratio, temperature, number of passes
Mechanical Processing | forging | temperature, strain rate, deformation degree
Mechanical Processing | extrusion | temperature, extrusion ratio, die design
Additive Manufacturing (AMT) | selective laser melting (SLM/LPBF) | laser power, scan speed, layer thickness, hatch spacing, volumetric energy density, atmosphere
Additive Manufacturing (AMT) | selective laser sintering (SLS) | laser power, scan speed, layer thickness, atmosphere
Additive Manufacturing (AMT) | direct energy deposition (DED) | laser power, scan speed, powder feed rate, layer thickness, hatch spacing, energy density
Additive Manufacturing (AMT) | laser cladding | laser power, scan speed, powder feed rate, substrate material
Thin-Film Deposition (TFD) | pulsed laser deposition | laser intensity, pulse duration, substrate temperature, vacuum level, target material
Thin-Film Deposition (TFD) | magnetron sputtering deposition | power (DC/RF), gas pressure, gas flow rate, substrate temperature, deposition time
Thin-Film Deposition (TFD) | plasma spray deposition | current, voltage, gas flow rate (Ar/H2), powder size, spray distance
"""