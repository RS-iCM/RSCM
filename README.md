# RSCM: Remote Sensing-integrated Crop Model

**Authors**: Chi Tim Ng at Hang Seng University of Hong Kong and Jonghan Ko at Chonnam National University
**Collaborator**: Jong-oh Ban at Hallym Polytechnic University
**Repository**: https://github.com/RS-iCM/RSCM

---

## Overview

RSCM is a modular, Python-based simulation framework that integrates remote sensing data with crop modeling processes. Designed for agricultural monitoring, RSCM enables dynamic simulation of crop growth, biophysical parameters (e.g., LAI, AGDM), and yield prediction under diverse environmental and management conditions. This framework facilitates the incorporation of machine learning algorithms to enhance parameter calibration and improve forecasting accuracy.

---

## Features

- Integration of leaf area index or remote sensing-derived vegetation indices with crop growth simulations
- Support for GDD-based phenological development tracking
- Yield estimation via partitioning functions
- Parameter optimization with the Powell method
- Modular design for adaptation to various crops and regions

---

## Requirements

- Python ≥ 3.8  
- numpy >= 1.22
- pandas >= 1.5
- matplotlib >= 3.5
- scikit-learn >= 1.1
- scipy >= 1.8

Install dependencies using:

```bash
pip install -r requirements.txt

```markdown
# RSCM: Remote Sensing-integrated Crop Model Software Framework

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RS-iCM/RSCM/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RSCM is an open-source, modular simulation framework designed to integrate satellite-derived vegetation indices (VIs) directly into process-based crop models via Bayesian Log-Log assimilation.


## Key Features
- [cite_start]**Bayesian Assimilation**: Automates parameter estimation ($L_0, a, b, c, rGDD$) using satellite time-series[cite: 143, 221].
- [cite_start]**Hybrid Architecture**: Combines Python's data flexibility with a high-performance C simulation core[cite: 150].
- [cite_start]**Multi-Crop Support**: Pre-configured parameters for Rice, Wheat, and Maize[cite: 182, 234].
- [cite_start]**Scalability**: Capable of processing regional-scale yield maps (millions of pixels) using shared-memory parallelization[cite: 780, 1198].

## System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows (10/11).
- **Python**: ≥ 3.8
- [cite_start]**Compiler**: GCC ≥ 7.5 (Linux/macOS) or MinGW-w64 (Windows) for the C-engine[cite: 1194].

## Installation
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/RS-iCM/RSCM.git](https://github.com/RS-iCM/RSCM.git)
   cd RSCM
```

Quick Start
To run a sample simulation for rice using 2021 field data:

Open the interactive notebook: RUN_Python_Rice_v1.ipynb.

Execute the cells to load the Bayesian priors, ingest MODIS-derived VIs, and generate the growth trajectory plots.

Repository Structure

/Data: Sample meteorological and VI datasets (Rice, Wheat, Maize).


/Source: C source code for the simulation engine and Powell optimization.


/Notebooks: Jupyter Notebook tutorials for plot-level and regional applications.

Citation
If you use this software in your research, please cite: Ng, C. T., Ko, J., Jeong, S., & Ban, J. (2026). RSCM: A Bayesian Remote Sensing-integrated Crop Model Software Framework for Yield Forecasting. SoftwareX.

License
This project is licensed under the MIT License - see the LICENSE file for details
