# wayang-kulit-segmentation
Computational segementation of Wayang Kulit video recordings using a Cross-Attention Temporal Model

# Overview
This repository consists the code for 4 different steps
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
1. Place your video in *data/videos/*
2. From *src/2_thumbnails_extractor*, run
```bash
python3 thumbnails_extractor.py --data_dir "../../data" --seconds_between_frames 60 --resolution 320 240
```
3. Go to *data/model_weights*, download the model weights using
```bash
wget https://huggingface.co/shawnliewhongwei/wayangkulit-segmentation/resolve/main/split1_network.iter-32.net
```
4. Go to *src/4_fact*, run
```bash
python3 -m.src --cfg_path "../../data/model_logs/split1_args.json" --thumbnails_path "../../data/thumbnails_npy/thumbnails_60secsPerFrame_320px240px" --mapping_path "../../data/model_logs/class_mapping.txt" --weights_path "../../data/model_weights/split1_network.iter-32.net" --output_json_path "../../data/prediction_results/07nov_preds.json"
```
5. Prediction results would be ready in *data/prediction_results*.

# Training with Labelling
1. Place all your videos in *data/videos/*
2. From *src/2_thumbnails_extractor*, run
```bash
python3 thumbnails_extractor.py --data_dir "../../data" --seconds_between_frames 60 --resolution 320 240
```
3. Perform the labelling **TBC**
4. **TBC**

