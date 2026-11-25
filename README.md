# Image Classifier (PyTorch + Gradio)

Train a ResNet-18 on your own image folders, then classify new images in a web UI.

## Project layout

```
.
├── requirements.txt
├── train.py              # trains the model -> custom_cnn.pth
├── app.py                # Gradio web UI for predictions
├── custom_cnn.pth        # trained checkpoint (included; overwritten by train.py)
└── dataset/
    ├── train/            # 2,000 images, 200 per class
    │   ├── tench/
    │   ├── english_springer/
    │   ├── cassette_player/
    │   ├── chain_saw/
    │   ├── church/
    │   ├── french_horn/
    │   ├── garbage_truck/
    │   ├── gas_pump/
    │   ├── golf_ball/
    │   └── parachute/
    └── val/              # 500 images, 50 per class
        └── (the same 10 class folders)
```

Each class is simply a subfolder name. The folder names become the labels the model
predicts, so `dataset/train/cats/` and `dataset/val/cats/` produce the class `cats`.

**A sample dataset is already included** so you can train immediately: a 10-class
subset of [Imagenette](https://github.com/fastai/imagenette), with 200 training and
50 validation images per class. You do not need to download anything to try the project.

## 1. Install dependencies

```bash
python -m venv .venv
# Windows (Git Bash / PowerShell path shown for bash):
source .venv/Scripts/activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Use your own images (optional)

The bundled sample data works out of the box. To train on your own photos, either
add new class folders alongside the existing ones or delete the existing folders
and replace them with your own. You need **at least two classes**, and both `train`
and `val` must contain **exactly the same folder names**.

1. Rename or add folders inside `dataset/train/`, one per class,
   for example `dataset/train/dogs/`, `dataset/train/cats/`.
2. Mirror the same folder names inside `dataset/val/`.
3. Drop image files (`.jpg`, `.jpeg`, `.png`, ...) into the matching folders.

```
dataset/train/dogs/dog1.jpg
dataset/train/dogs/dog2.jpg
dataset/train/cats/cat1.jpg
dataset/val/dogs/dog10.jpg
dataset/val/cats/cat10.jpg
```

A common split is ~80% of each class in `train` and ~20% in `val`. Aim for at
least a few dozen images per class for usable results.

To remove the sample data and start clean:

```bash
rm -rf dataset/train/* dataset/val/*
```

## 3. Train

```bash
python train.py
```

The script:

- loads `dataset/train` and `dataset/val` with `ImageFolder` + `DataLoader`
- applies augmentation (`RandomResizedCrop(224)`, `RandomHorizontalFlip()`) to
  training images and `Resize(256)` + `CenterCrop(224)` to validation images,
  normalized with ImageNet mean/std
- starts from pre-trained `resnet18` weights, freezes the base layers, and
  replaces the final layer with a new one sized to your number of classes
- trains for 10 epochs with `CrossEntropyLoss` and `Adam`, using CUDA, MPS or CPU
  automatically, and prints train/val loss and accuracy each epoch
- saves the best-performing weights plus the `class_names` list to `custom_cnn.pth`

If you change your class folders, just run `python train.py` again.

## 4. Launch the web UI

```bash
python app.py
```

Open the printed local URL (usually http://127.0.0.1:7860), upload an image, and
you will see a probability for every class. `app.py` rebuilds ResNet-18 with the
number of classes stored in `custom_cnn.pth`, loads the trained weights, and runs
inference under `torch.no_grad()`.

## Troubleshooting

- **`Checkpoint not found`** – run `python train.py` before `python app.py`.
- **`Class folders differ between train and val`** – make the folder names match
  exactly in both splits.
- **`No training images found`** – the class folders exist but are empty; add images.
- **Training is slow on CPU** – the bundled 2,000-image sample took about 30
  minutes for 10 epochs on CPU here (a GPU is much faster). Reduce `BATCH_SIZE` or
  `NUM_EPOCHS` at the top of `train.py`, or use a GPU.
