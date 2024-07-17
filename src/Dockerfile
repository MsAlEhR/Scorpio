# Use an official miniconda image as the base
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Install necessary packages and clean up
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment.yml file to the container
COPY environment.yml /app/environment.yml

# Install the conda environment from environment.yml and clean up
RUN conda env create -f /app/environment.yml && \
    conda clean -afy && \
    rm -rf /opt/conda/pkgs /tmp/* /var/tmp/*

# Set up environment variables and activate the environment
ENV PATH /opt/conda/envs/scorpio/bin:$PATH
SHELL ["conda", "run", "-n", "scorpio", "/bin/bash", "-c"]

# Copy necessary files
COPY setup.sh /app/setup.sh
COPY scorpio /app/scorpio
COPY createdb.py /app/createdb.py
COPY triplet_trainer.py /app/triplet_trainer.py
COPY inference.py /app/inference.py
COPY confidence_score.py /app/confidence_score.py
COPY utils.py /app/utils.py
COPY TripletModel.py /app/TripletModel.py

# Make the scripts executable and run the setup script
RUN chmod +x /app/setup.sh /app/scorpio /app/*.py && /bin/bash /app/setup.sh && \
    rm /app/setup.sh /app/environment.yml

# Copy and set up scorpio script
COPY scorpio /usr/local/bin/scorpio
RUN chmod +x /usr/local/bin/scorpio

# Set entrypoint
ENTRYPOINT ["scorpio"]
