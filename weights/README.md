Place model files here:

```text
weights/paper_29.pth              # PressureVision++ checkpoint (download)
weights/hand_landmarker.task      # MediaPipe Hands model (shipped / auto-download)
```

PressureVision++ checkpoint:
https://www.dropbox.com/scl/fi/0r2koefy7bhr66dffc8z7/paper_29.pth?rlkey=wshcxm8iy8l1qo60oo7khdqjp&dl=1

```bash
curl -L -o weights/paper_29.pth "https://www.dropbox.com/scl/fi/0r2koefy7bhr66dffc8z7/paper_29.pth?rlkey=wshcxm8iy8l1qo60oo7khdqjp&dl=1"
```

If `hand_landmarker.task` is missing (or SSL download fails on macOS):

```bash
curl -L -o weights/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```
