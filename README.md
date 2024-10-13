# PASAD_Assignment
This Assignment implements intrusion detection method using PASAD.
=======

# PASAD Analysis Project

This project implements and compares various approaches to Process-Aware Stealthy Attack Detection (PASAD) across different datasets. It includes implementations of PASAD using various distance measures, threshold configurations, and clustering center calculations. The project also explores different Lag parameter configurations to optimize the PASAD detection.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Data Files](#data-files)
- [Contributors](#contributors)
- [License](#license)

## Introduction

PASAD (Process-Aware Stealthy Attack Detection) is a method for detecting attacks in Cyber-Physical Systems (CPS) by analyzing temporal sequences of sensor data. This project extends PASAD to various implementations and datasets, evaluating attack detection efficacy and performance.

## Project Structure

- **`pasad_ssa_analysis.py`**: Contains the function that implements PASAD for question Q1a. It processes the data and outputs detection results including scree plots, raw data plots, departure plots, attack detection, and runtime analysis.
  
- **`pasad_main.ipynb`**: Calls the `pasad_ssa_analysis.py` function to perform PASAD analysis on all Tennessee Eastman (TE) datasets. Outputs include:
  - Scree plots
  - Raw data and departure plots
  - Attack detection results
  - Runtime analysis

- **`pasad_SWaT.ipynb`**: Contains and calls the PASAD implementation for SWaT datasets, addressing question Q1 for the SWaT system.

- **`pasad_compare.py`**: Implements PASAD for comparing Untransformed (UT) and Unit-Untransformed (UUT) methods for question Q1b.

- **`pasad_compare.ipynb`**: Calls the `pasad_compare.py` functions, compares results between UT and UUT approaches, and evaluates their runtime performance.

- **`pasad_maxDep.ipynb`**: Implements PASAD with an increased threshold for maximum departure score, providing a bar chart representing the number of attacks detected across all TE datasets. This addresses question Q1c.

- **`pasad_madDep_center.ipynb`**: Implements PASAD with an increased threshold for the maximum departure score and computes the cluster center based on the middle point. 

- **`pasad_ssa_center.py`**: Contains the PASAD implementation where the center of the cluster is calculated using the midpoint instead of the centroid for question Q2.

- **`pasad_center.ipynb`**: Calls the `pasad_ssa_center.py` function and outputs results for all datasets. This includes scree plots, attack detection, departure plots, and runtime analysis.

- **`pasad_ssa_mahalanobis.py`**: Implements PASAD using Mahalanobis distance instead of Euclidean distance to analyze attack detection for question Q3.

- **`pasad_mahalanobis.ipynb`**: Calls the `pasad_ssa_mahalanobis.py` function and provides outputs for all datasets, including:
  - Scree plots
  - Raw data and departure plots
  - Attack detection
  - Runtime analysis

- **`PASAD_TE_COMBINATIONS.ipynb`**: Tests various values of the Lag parameter for PASAD to identify the optimal Lag configuration for attack detection.

- **`DataFiles/`**: Directory containing all datasets used in the project, including Tennessee Eastman (TE) and SWaT datasets.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Vrajb24/PASAD_Assignment.git
    cd PASAD_Assignment
    ```

2. Install the necessary dependencies (see [Dependencies](#dependencies)).

3. Open the Jupyter notebooks or run the Python scripts as described in the [Usage](#usage) section.

## Usage

### Running PASAD for TE datasets:
- To run PASAD analysis on the TE datasets and visualize the results, open and execute `pasad_main.ipynb`.

### Comparing UT and UUT methods:
- To compare PASAD results between UT and UUT methods, open and execute `pasad_compare.ipynb`.

### Running PASAD with various Lag parameters:
- To evaluate PASAD with different Lag parameter configurations, open and execute `PASAD_TE_COMBINATIONS.ipynb`.

## Features

- **PASAD Implementation**: Detection of attacks using PASAD on various datasets.
- **Comparison**: Analyze the difference between PASAD with different distance measures (Mahalanobis, Euclidean).
- **Threshold Tuning**: Evaluate the effect of increasing the departure score threshold.
- **Center Calculation**: Test different approaches to calculating cluster centers (centroid vs midpoint).
- **Performance Evaluation**: Measure runtime for each implementation.
  
## Dependencies

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `jupyter`

You can install the required dependencies using `pip`:
```bash
pip install numpy pandas matplotlib scikit-learn scipy jupyter
```

## Data Files

The `DataFiles/` directory contains the following datasets:

- **Tennessee Eastman (TE)**: Used for general PASAD analysis.
- **SWaT**: Used specifically for SWaT PASAD analysis.

## Contributors

- **Vraj Patel** - *Initial Work*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
>>>>>>> abb3dbd (everything I submitted)
>>>>>>> 59418cc (everything)
