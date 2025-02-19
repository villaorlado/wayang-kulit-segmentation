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

**If you only have the videos, do the following**
1. Place your videos in *data/videos/*
2. Run the script
```bash
bash video_inference.sh
```
3. Check *data/prediction_visualisations* for results
<em>Note: If you changed the parameters in Step 3.1, the folder names used in Steps 5, 6.1, 6.2 in **video_inference.sh** would be different. Do adjust accordingly.</em>

**If you already have the thumbnails, do the following**
1. Organise the thumbnails in the following manner 
* data/thumbnails/**thumbnails_dir_name**/**video_id_0**/**thumbnail_0_frame_nb**
* data/thumbnails/**thumbnails_dir_name**/**video_id_0**/**thumbnail_1_frame_nb**
* data/thumbnails/**thumbnails_dir_name**/**video_id_0**/**thumbnail_2_frame_nb**
* data/thumbnails/**thumbnails_dir_name**/**video_id_1**/**thumbnail_0_frame_nb**
* data/thumbnails/**thumbnails_dir_name**/**video_id_1**/**thumbnail_1_frame_nb**
* data/thumbnails/**thumbnails_dir_name**/**video_id_1**/**thumbnail_2_frame_nb**
2. Modify the folder names in Steps 4, 5.1, 5.2 in **thumbnails_inference.sh**
3. Run the script
```bash
bash thumbnails_inference.sh
```
4. Check *data/prediction_visualisations* for results

