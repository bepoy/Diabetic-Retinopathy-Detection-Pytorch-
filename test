import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F

# Transformation class
def transform_coco(image, target):
    """Transforms the image and prepares the target for Faster R-CNN."""
    # Convert the image to a tensor
    image = F.to_tensor(image)

    # Validate and process target
    valid_boxes, valid_labels = [], []
    for obj in target:
        x, y, w, h = obj['bbox']
        if w > 0 and h > 0:  # Ensure valid box dimensions
            valid_boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
            valid_labels.append(obj['category_id'])

    target = {
        "boxes": torch.tensor(valid_boxes, dtype=torch.float32),
        "labels": torch.tensor(valid_labels, dtype=torch.int64),
    }

    return image, target

# Dataset loader
def get_coco_dataset(img_dir, ann_file):
    """Returns a CocoDetection dataset with the specified transformations."""
    return CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transforms=transform_coco
    )

# Model setup
def get_model(num_classes):
    """Loads a pre-trained Faster R-CNN model and adapts it for custom classes."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# DataLoader collate function
def collate_fn(batch):
    return tuple(zip(*batch))

# Training loop
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Trains the model for one epoch."""
    model.train()
    epoch_loss = 0

    for images, targets in data_loader:
        # Move images and targets to the device
        images = [img.to(device) for img in images]
        targets = [
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
            }
            for target in targets
        ]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f"Epoch [{epoch}] Loss: {epoch_loss / len(data_loader):.4f}")

# Paths to data and annotations
train_img_dir = "/home/bepoy/meow/train"
train_ann_file = "/home/bepoy/meow/train/annotations/annotations.json"
val_img_dir = "/home/bepoy/meow/val"
val_ann_file = "/home/bepoy/meow/val/annotations/annotations.json"

# Initialize datasets and loaders
train_dataset = get_coco_dataset(train_img_dir, train_ann_file)
val_dataset = get_coco_dataset(val_img_dir, val_ann_file)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Model setup
num_classes = 6  # Background + number of classes
model = get_model(num_classes)

# Device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 6
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    lr_scheduler.step()

    # Save the model's state after every epoch
    model_path = f"fasterrcnn_resnet50_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
