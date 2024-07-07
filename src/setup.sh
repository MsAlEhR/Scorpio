#!/bin/bash

echo "Making scripts executable..."
chmod +x /app/scorpio
chmod +x /app/createdb.py
chmod +x /app/triplet_trainer.py
chmod +x /app/inference.py
chmod +x /app/confidence_score.py
chmod +x /app/utils.py
chmod +x /app/TripletModel.py

echo "Copying scripts to conda environment bin directory..."
mkdir -p /opt/conda/envs/scorpio/bin
cp /app/scorpio /opt/conda/envs/scorpio/bin/
cp /app/createdb.py /opt/conda/envs/scorpio/bin/
cp /app/triplet_trainer.py /opt/conda/envs/scorpio/bin/
cp /app/inference.py /opt/conda/envs/scorpio/bin/
cp /app/confidence_score.py /opt/conda/envs/scorpio/bin/
cp /app/utils.py /opt/conda/envs/scorpio/bin/
cp /app/TripletModel.py /opt/conda/envs/scorpio/bin/
chmod +x /opt/conda/envs/scorpio/bin/*

echo "Setup complete!"
