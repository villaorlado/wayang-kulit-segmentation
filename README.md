# Computational segementation of Wayang Kulit video recordings using a Cross-Attention Temporal Model
This is the repository for "Computational segementation of Wayang Kulit video recordings using a Cross-Attention Temporal Model" to be presented at [Computational Humanities Research 2024](http://2024.computational-humanities-research.org/papers/paper141/).

We report preliminary findings on a novel approach to automatically segment Javanese [wayang kulit](https://cwa-web.org) (traditional leather puppet) performances using computational methods. We focus on identifying comic interludes, which have been the subject of scholarly debate regarding their increasing duration. Our study employs action segmentation techniques from a Cross-Attention Temporal Model, adapting methods from computer vision to the unique challenges of wayang kulit videos. We manually labelled 100 video recordings of performances to create a dataset for training and testing our model. These videos, which are typically 7 hours long, were sampled from our comprehensive dataset of 12,638 videos, created between 03 Jun 2012 and 30 Dec 2023. The resulting algorithm achieves an accuracy of 89.06 % in distinguishing between comic interludes and regular performance segments, with F1-scores of 96.53 %, 95.91 %, and 92.47 % at overlapping thresholds of 10 %, 25 %, and 50 % respectively. This work demonstrates the potential of computational approaches in analyzing traditional performing arts and other video material, offering new tools for quantitative studies of audiovisual cultural phenomena, and provides a foundation for future empirical research on the evolution of wayang kulit performances.

# Overview
This repository consists the code for 4 different steps
1. Extracting thumbnails from the videos
2. Labelling the videos
3. Training and Inference with the FACT action segmentation model
4. Render the resutls

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