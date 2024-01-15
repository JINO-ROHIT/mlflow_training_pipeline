# Mlflow Pytorch Pipeline

This is a simple training pipeline for a segmentation model(Unet + Effnet) implemented in pytorch along with mlflow tracking.

## Installation

1. Create a virtual environment

```bash
python3 -m venv hubmap
source hubmap/bin/activate
```

2. Install the requirements

```bash
make dev
```

3. Begin the training

```bash
make train
```

4. Launch the UI to see the tracked parameters and models

```bash
make ui
```
