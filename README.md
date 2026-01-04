# ACV_Final
114-1 ACV final project

## Environment setup

```bash
python3 -m venv acv_env
source acv_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Test it

### Stern wave detection with tradition and AI(YOLOv8 OBB) comparison

```bash
python3 stern_wave_yolo_n_tradition.py
```

### Boad direction code with tradition and AI(YOLOv8 OBB) comparison
```bash
python3 boat_direction_yolo_n_tradition.py
```

> **NOTE:**
>  Both stern wave detection and boat direction are using the same YOLOv8 OBB model
> best_mix_150.pt
