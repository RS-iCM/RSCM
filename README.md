# RSiCM: Remote Sensing-integrated Crop Model

**Authors**: Chi Tim Ng at Hang Seng University of Hong Kong and Jonghan Ko at Chonnam National University
**Collaborator**: Jong-oh Ban at Hallym Polytechnic University
**Repository**: https://github.com/RS-iCM/RSCM

---

## Overview

RSiCM is a modular, Python-based simulation framework that integrates remote sensing data with crop modeling processes. Designed for agricultural monitoring, RSCM enables dynamic simulation of crop growth, biophysical parameters (e.g., LAI, AGDM), and yield prediction under diverse environmental and management conditions. This framework facilitates the incorporation of machine learning algorithms to enhance parameter calibration and improve forecasting accuracy.

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

