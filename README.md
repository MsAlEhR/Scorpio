# Scorpio

![GitHub language](https://img.shields.io/github/languages/top/MsAlEhR/Scorpio)
![GitHub license](https://img.shields.io/github/license/MsAlEhR/Scorpio)
![Docker Pulls](https://img.shields.io/docker/pulls/eesilab/scorpio)

Welcome to the Scorpio project! This repository contains advanced tools for training triplet networks using contrastive learning on diverse DNA sequences and data from promoter detection, phylogenomic analysis, antimicrobial resistance (AMR) detection, and any hierarchical information, which can improve downstream analysis and insights.

<p align="center">
  <img src="scorpio_logo.webp" alt="Scorpio Logo" width="250" height="250">
</p>

## Tutorials

The [GitHub Wiki](https://github.com/MsAlEhR/Scorpio/wiki) also contains tutorials to help you learn how to use Scorpio tools with real data.

## Installation

You can set up the environment for `scorpio` using either a conda environment or a Docker image. Follow the instructions below for your preferred method:

### Method 1: Conda Environment

1. **Create a conda environment named `scorpio` based on the environment file in the `src` directory:**

    ```bash
    conda env create -f src/environment.yml -n scorpio
    ```

2. **Activate the conda environment:**

    ```bash
    conda activate scorpio
    ```

3. **Run the setup script to add `scorpio` to your PATH:**

    ```bash
    ./src/setup.sh
    ```

### Method 2: Docker

1. **Download and run the Docker image:**

    ```bash
    docker pull eesilab/scorpio
    docker run -it eesilab/scorpio
    ```

After following the steps for either method, your environment should be set up and ready to use the `scorpio` tool.
