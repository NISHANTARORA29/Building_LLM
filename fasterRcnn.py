# train_detector.py
import os
import time
import json
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np

# -------------------------
# Helper: collate_fn
# -------------------------
def collate_fn(batch):
    return tuple(zip(*batch))

# -------------------------
# COCO-style dataset wrapper for torchvision models
# Expects: images/ and annotations in COCO JSON file
# -------------------------
class COCODataset(Dataset):
    def __init__(self, images_dir, annotation_json, transforms=None, keep_empty=False, with_masks=False):
        from pycocotools.coco import COCO
        self.images_dir = images_dir
        self.coco = COCO(annotation_json)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.keep_empty = keep_empty
        self.with_masks = with_masks

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img = Image.open(os.path.join(self.images_dir, path)).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        masks = []

        for a in anns:
            # COCO bbox: [x, y, width, height] -> convert to x1,y1,x2,y2
            x, y, w, h = a['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a.get('category_id', 1))
            areas.append(a.get('area', w*h))
            iscrowd.append(a.get('iscrowd', 0))
            if self.with_masks and 'segmentation' in a:
                masks.append(self.coco.annToMask(a))

        # If no annotations and keep_empty False, return a dummy and let loss handle it
        if len(boxes) == 0 and not self.keep_empty:
            # return an empty target (torchvision detection models accept 0-box images)
            target = {}
            target["boxes"] = torch.zeros((0,4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["image_id"] = torch.tensor([img_id])
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
            if self.with_masks:
                target["masks"] = torch.zeros((0, img.size[1], img.size[0]), dtype=torch.uint8)
        else:
            import torch
            target = {}
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["image_id"] = torch.tensor([img_id])
            target["area"] = torch.as_tensor(areas, dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)
            if self.with_masks:
                masks_np = np.stack(masks, axis=0) if len(masks) > 0 else np.zeros((0, img.size[1], img.size[0]), dtype=np.uint8)
                target["masks"] = torch.as_tensor(masks_np, dtype=torch.uint8)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

# -------------------------
# Transforms (to tensors, optional augmentation)
# -------------------------
class ComposeTransforms:
    def __init__(self, train=True):
        # Minimal; add augmentations as needed
        self.train = train
        self.to_tensor = T.ToTensor()

    def __call__(self, image, target):
        image = self.to_tensor(image)
        # torchvision object detection models expect tensors in [0,1]
        return image, target

# -------------------------
# Create model
# -------------------------
def get_model(num_classes, model_type='fasterrcnn', pretrained_backbone=True, pretrained=False):
    """
    num_classes: including background (so classes+1)
    model_type: 'fasterrcnn' or 'maskrcnn'
    pretrained: load pretrained COCO weights for the whole model (optional)
    """
    if model_type == 'fasterrcnn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=pretrained_backbone)
        # replace the head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    elif model_type == 'maskrcnn':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained_backbone=pretrained_backbone)
        # replace box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        # replace mask predictor with the right number of classes
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    else:
        raise ValueError("Unknown model_type")
    if pretrained:
        # if you want to load externally provided checkpoint later, do it in training loop
        pass
    return model

# -------------------------
# Training loop
# -------------------------
def train(
    images_dir,
    ann_file,
    output_dir,
    num_classes,
    model_type='fasterrcnn',
    epochs=12,
    batch_size=2,
    lr=0.005,
    device='cuda'
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    dataset = COCODataset(images_dir, ann_file, transforms=ComposeTransforms(train=True), with_masks=(model_type=='maskrcnn'))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = get_model(num_classes=num_classes, model_type=model_type).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    print("Start training")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        tic = time.time()
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        toc = time.time()
        print(f"Epoch [{epoch+1}/{epochs}] loss: {epoch_loss:.4f} time: {toc - tic:.1f}s")
        # save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    print("Training finished. Checkpoints saved to", output_dir)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help="images directory")
    parser.add_argument('--ann', required=True, help="coco annotations json")
    parser.add_argument('--out', default='./runs', help='output dir')
    parser.add_argument('--classes', type=int, required=True, help='num classes including background')
    parser.add_argument('--type', default='fasterrcnn', choices=['fasterrcnn','maskrcnn'])
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    train(args.images, args.ann, args.out, args.classes, model_type=args.type, epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=args.device)
