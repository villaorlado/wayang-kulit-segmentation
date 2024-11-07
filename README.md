# wayang-kulit-segmentation
Computational segementation of Wayang Kulit video recordings using a Cross-Attention Temporal Model

# Overview
This repository consists the code for 5 different steps
1. Querying YouTube Data API and Downloading YouTube videos
2. Extracting thumbnails from the videos
3. Labelling the videos
4. Training and Inference with the FACT action segmentation model
5. Render the resutls

# Installation
Run the command
```bash
pip install -r requirements.txt
```

# Quick guide for inference
1. Place your videos in *data/videos/*
2. Run the script
```bash
bash inference.sh
```
3. Check *data/prediction_visualisations* for results