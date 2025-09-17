# Code and Data for "Detection of a Higher Harmonic Quasi-normal Mode in the Ringdown Signal of GW231123"

[](https://opensource.org/licenses/MIT)
[Posteriors](https://doi.org/10.5281/zenodo.17142955)

This repository contains the source code and data to reproduce the results presented in the paper: *Detection of a Higher Harmonic Quasi-normal Mode in the Ringdown Signal of GW231123*.

The analysis identifies the (2,0,0) quasi-normal mode (QNM) in the ringdown of GW231123 with very high statistical confidence. We provide the scripts for two distinct time-domain analysis methodologies used in the paper:

1.  **TTD (Traditional Time-Domain)**: A standard Bayesian inference approach that stochastically samples the full parameter space of the ringdown model (`TTD_rd231123.py`).
2.  **Fs (F-statistic)**: A semi-analytical method that enhances efficiency by analytically marginalizing over linear parameters (amplitudes and phases), thereby reducing the dimensionality of the parameter space (`TDFs_rd231123.py`).

-----

## 1\. System Requirements

  - **Operating System**: The software has been tested on Ubuntu22.04 and Ubuntu24.04 systems.
  - **Software Dependencies**: All required Python packages and their specific versions are listed in the `py10.yml` file. Key dependencies include:
      - `python=3.10`
      - `bilby=2.4.0`
      - `pycbc=2.7.2`
      - `numpy`
      - `scipy`
      - `gwpy`
      - `lalsuite`
  - **Hardware**: No non-standard hardware is required. The analysis can be run on a standard multi-core desktop or laptop computer.

-----

## 2\. Installation Guide

We recommend using `conda` to create a dedicated environment, which ensures that all dependencies are handled correctly.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/whaitian/FsRD.git
    cd FsRD
    ```

2.  **Create and activate the conda environment:**
    This command will create a new environment named `gw10` and install all necessary packages from the `py10.yml` file.

    ```bash
    conda env create -f py10.yml
    conda activate gw10
    ```

<!-- end list -->

  - **Typical Installation Time**: The installation process typically takes about 10-15 minutes on a standard desktop computer, depending on network speed.

-----

## 3\. Demonstration

To verify that the installation is successful and to understand the basic workflow, you can run a quick demonstration.

### Running the Demo

The demo will run the F-statistic analysis for the `220+200` mode combination, starting the analysis `12 M` after the signal's polarization peak. This corresponds to the point of maximum evidence found in our paper.

Execute the following command from the root directory of the repository:

```bash
python TDFs_rd231123.py 221,201 12
```

### Expected Output

The script will create an output directory named `Fs_only2/221,201_12M_0-4s/`. Inside this directory, you will find:

  - A `.json` file (e.g., `221,201_12M_0-4s_p2_result.json`) containing the posterior samples and evidence calculations.
  - A `.log` file containing detailed information about the run.
  - Corner plots of the inferred parameters.

### Expected Runtime

  - **F-statistic method (`TDFs_rd231123.py`)**: Approximately 3-20 minutes.
  - **TTD method (`TTD_rd231123.py`)**: Approximately 30-100 minutes.

The exact time will depend on your system's performance.

-----

## 4\. Usage and Reproducibility

### How to Run the Software on Your Data

The analysis scripts are designed to be run from the command line, with arguments specifying the QNM modes and the analysis start time.

**General Command Structure:**

```bash
python <script_name.py> <mode_string> <start_time_M>
```

**Arguments:**

  - `<script_name.py>`:
      - `TTD_rd231123.py`: For the traditional time-domain (full sampling) method.
      - `TDFs_rd231123.py`: For the faster F-statistic (semi-analytic) method.
  - `<mode_string>`: A comma-separated string specifying the QNMs to include. The notation `lmn` corresponds to the overtone `n-1`. For example:
      - `221`: Includes only the fundamental (2,2,0) mode.
      - `221,201`: Includes the (2,2,0) and (2,0,0) modes.
      - `222`: Includes the (2,2,0) and (2,2,1) modes.
  - `<start_time_M>`: The analysis start time as a delay after the signal's polarization peak, in units of the redshifted remnant mass `M` (e.g., `8`, `10`, `12`).

**Example:**
To run the TTD analysis for the `220+200` mode combination starting at `12 M`, use:

```bash
python TTD_rd231123.py 221,201 12
```

### Reproducing the Paper's Results

To reproduce all quantitative results and figures from the manuscript, you will need to run the analysis scripts for all the mode combinations and start times discussed in the paper (see Figures 1 and 3).

The final posterior samples used to generate the figures in the paper are publicly available on Zenodo for convenience:
[**https://doi.org/10.5281/zenodo.17142955**](https://doi.org/10.5281/zenodo.17142955)

### Data

A pre-processed data file is provided in the `TD_data/` directory: `PyCBC_psd_acfs_GW231123_20-1024Hz_t8s-v0.npy`.

This file contains the necessary inputs for the analysis, including:

  - The whitened time-domain strain data for H1 and L1 detectors.
  - The estimated auto-covariance functions (ACFs) for the detector noise.

This data was generated using the `save_psd_acf.py` script, which downloads the public GWOSC data for GW231123 and performs the necessary processing (downsampling, filtering, and PSD/ACF estimation).

-----

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. We strongly recommend using an [OSI-approved license](https://opensource.org/licenses) for any derivative works.
