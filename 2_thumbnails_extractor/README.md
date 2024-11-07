# Frames extraction
Given a directory of videos, we want to extract frames every X seconds with resolution((XX,YY)).

To run the script, run

```bash
python3 thumbnails_extractor.py --data_dir "../../data" --seconds_between_frames 60 --resolution 320 240
```