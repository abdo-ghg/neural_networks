import os9
import cv2 as c
import torch as t
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

IMG_SIZE      = 96
VALID_CLASSES = {"fear", "happy", "disgust", "sad", "surprise", "angry"}
label_map     = {cls: i for i, cls in enumerate(sorted(VALID_CLASSES))}
idx_to_label  = {v: k for k, v in label_map.items()}

comp_test = "/test/test"

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# ── Transforms ───────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

tta_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ── Dataset ──────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images    = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img

# ── Load test images ─────────────────────────────────────────
def load_test_images(path):
    images, filenames = [], []
    for img_file in os.listdir(path):
        if not img_file.endswith('.jpg'):
            continue
        img = c.imread(os.path.join(path, img_file), c.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
        filenames.append(img_file)
    return images, filenames

print("Loading test data...")
test_imgs, test_filenames = load_test_images(comp_test)
print(f"Total test samples: {len(test_imgs)}")

test_dl = DataLoader(TestDataset(test_imgs, transform=test_transform), batch_size=32, shuffle=False)

# ── Load model ───────────────────────────────────────────────
# Replace EmotionVGG with your actual model class
model = EmotionVGG().to(device)
model.load_state_dict(t.load("best_vgg.pth", map_location=device))

# ── TTA Inference ────────────────────────────────────────────
TTA_RUNS  = 5
model.eval()
all_preds = []

with t.no_grad():
    for batch_idx, imgs in enumerate(test_dl):
        imgs  = imgs.to(device)
        B     = imgs.size(0)
        probs = t.zeros(B, 6).to(device)

        for _ in range(TTA_RUNS):
            start = batch_idx * 32
            tta_batch = DataLoader(
                TestDataset(test_imgs[start:start + B], transform=tta_transform),
                batch_size=B, shuffle=False
            )
            tta_imgs = next(iter(tta_batch)).to(device)

            with t.cuda.amp.autocast():
                probs += t.softmax(model(tta_imgs), dim=1)

        all_preds.extend(probs.argmax(dim=1).cpu().tolist())

# ── Save submission ───────────────────────────────────────────
df = pd.DataFrame({
    "ID"   : test_filenames,
    "Label": [idx_to_label[p] for p in all_preds]
})
df.to_csv("submission.csv", index=False)
print(df.head(10))
print(f"\nSaved submission.csv → {len(df)} rows")