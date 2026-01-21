import torch
import os

# Monkeypatch torch.compile to avoid "Compiler: cl is not found" error on Windows without MSVC
# SAM3 uses torch.compile by default if not explicitly disabled or if the environment suggests it.
torch.compile = lambda x, *args, **kwargs: x

from ultralytics.models.sam import SAM3VideoPredictor

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "sam3.pt")
video_path = os.path.join(script_dir, "KakaoTalk_20251129_223033383.mp4")

# Create video predictor
overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, half=False, compile=False)
predictor = SAM3VideoPredictor(overrides=overrides)

# Track objects using bounding box prompts
results = predictor(source=video_path, bboxes=[[706.5, 442.5, 905.25, 555], [598, 635, 725, 750]], stream=True)

# Process and display results
for r in results:
    r.show()  # Display frame with segmentation masks