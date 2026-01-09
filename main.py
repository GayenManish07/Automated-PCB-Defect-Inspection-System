import os
import torch
import cv2
import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T

# arguments
def get_args():
    parser = argparse.ArgumentParser(description="PCB Defect Detection Final")
    parser.add_argument('--data_root', type=str, default='/home/manish/thesis/computer_vision/DeepPCB/PCBData', help='Path to PCBData')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_aug', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./output_comparison')
    return parser.parse_args()

# class configuration
CLASSES = {
    0: 'Background', 1: 'open', 2: 'short', 3: 'mousebite', 
    4: 'spur', 5: 'copper', 6: 'pin-hole'
}
SEVERITY_MAP = {
    'open': 'CRITICAL', 'short': 'CRITICAL', 
    'mousebite': 'MAJOR', 'spur': 'MAJOR', 
    'copper': 'MINOR', 'pin-hole': 'MINOR'
}
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# transformation for augmentations
def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    return T.Compose(transforms)

# class to read dataset
class DeepPCBDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs, self.anns, self.templates = [], [], []
        
        print(f"[INFO] Scanning {self.root}...")
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith('_test.jpg'):
                    img_path = os.path.join(dirpath, f)
                    
                    current_folder = os.path.basename(dirpath)
                    parent = os.path.dirname(dirpath)
                    ann_folder = os.path.join(parent, current_folder + "_not")
                    
                    ann_path_a = os.path.join(ann_folder, f.replace('.jpg', '.txt'))
                    ann_path_b = os.path.join(ann_folder, f.replace('_test.jpg', '.txt'))
                    
                    final_ann = ann_path_a if os.path.exists(ann_path_a) else (ann_path_b if os.path.exists(ann_path_b) else None)
                    temp_path = os.path.join(dirpath, f.replace('_test.jpg', '_temp.jpg'))
                    
                    if final_ann:
                        self.imgs.append(img_path)
                        self.anns.append(final_ann)
                        self.templates.append(temp_path if os.path.exists(temp_path) else None)

        if not self.imgs: raise RuntimeError("Dataset empty.")
        print(f"[SUCCESS] Loaded {len(self.imgs)} pairs.")

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        boxes, labels = [], []
        with open(self.anns[idx], 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()))
                if len(parts) == 5:
                    x1, y1, x2, y2, c = parts
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(c)
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        
        if self.transform: img = self.transform(img)
        return img, target

    def __len__(self): return len(self.imgs)

# model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# evaluation
def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def evaluate_model(model, data_loader, device):
    model.eval()
    print("\n--- Evaluation on Test Set ---")
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                gt_boxes = targets[i]['boxes'].numpy()
                
                keep = pred_scores > 0.5
                pred_boxes = pred_boxes[keep]
                gt_matched = [False] * len(gt_boxes)
                
                for pb in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    for gi, gb in enumerate(gt_boxes):
                        iou = calculate_iou(pb, gb)
                        if iou > best_iou: best_iou, best_gt_idx = iou, gi
                    
                    if best_iou >= 0.5:
                        if not gt_matched[best_gt_idx]: tp += 1; gt_matched[best_gt_idx] = True
                        else: fp += 1
                    else: fp += 1
                fn += sum(1 for m in gt_matched if not m)
                
    prec = tp / (tp + fp + 1e-7)
    rec = tp / (tp + fn + 1e-7)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-7)
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}\n")

# visualization and logging
def predict_and_draw(model, img_tensor, image_name="Unknown"):
    model.eval()
    with torch.no_grad(): prediction = model([img_tensor.to(DEVICE)])[0]

    img_np = img_tensor.mul(255).permute(1, 2, 0).byte().numpy().copy()
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = img_np.shape
    
    detected_data = [] 

    for i in range(len(prediction['boxes'])):
        score = prediction['scores'][i].item()
        if score > 0.5:
            box = prediction['boxes'][i].cpu().numpy().astype(int)
            label = CLASSES.get(prediction['labels'][i].item(), 'Unknown')
            severity = SEVERITY_MAP.get(label, 'UNK')
            cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
            
            detected_data.append([image_name, label, f"{score:.4f}", severity, cx, cy])
            print(f"  [Report] {image_name}: {label} at ({cx}, {cy}) | {severity}")

            color = (0, 0, 255) if severity == 'CRITICAL' else ((0, 255, 255) if severity == 'MAJOR' else (0, 255, 0))

            cv2.rectangle(img_np, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.circle(img_np, (cx, cy), 3, (0, 255, 0), -1)
            
            txt_top = f"{label} {score:.2f} {severity}"
            (wt, ht), _ = cv2.getTextSize(txt_top, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img_np, (box[0], box[1]-15), (box[0]+wt, box[1]), color, -1)
            cv2.putText(img_np, txt_top, (box[0], box[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

            txt_bot = f"({cx}, {cy})"
            (wb, hb), _ = cv2.getTextSize(txt_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            y_bot = box[3] + 15 if box[3] + 15 < h_img else box[3] - 5
            bg_y1 = box[3] if box[3] + 15 < h_img else box[3] - 15 - hb
            bg_y2 = box[3] + 15 if box[3] + 15 < h_img else box[3]
            
            if box[3] + 15 < h_img:
                cv2.rectangle(img_np, (box[0], bg_y1), (box[0]+wb, bg_y2), color, -1)
            cv2.putText(img_np, txt_bot, (box[0], y_bot), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0) if box[3] + 15 < h_img else color, 1)

    return img_np, detected_data

def generate_comparison_image(model, defective_tensor, template_path, save_path):
    filename = os.path.basename(save_path)
    print(f"Processing: {filename}")
    
    defective_drawn, csv_data = predict_and_draw(model, defective_tensor, filename)
    
    if template_path and os.path.exists(template_path):
        clean_tensor = T.ToTensor()(Image.open(template_path).convert("RGB"))
        clean_drawn, _ = predict_and_draw(model, clean_tensor, "Template")
    else:
        clean_drawn = np.zeros_like(defective_drawn)
    
    h, w, _ = defective_drawn.shape
    border = 10
    footer_h = 50 # extra space for labels
    
    combined = np.zeros((h + footer_h, w*2 + border, 3), dtype=np.uint8)
    
    combined[:h, :w, :] = clean_drawn
    combined[:h, w:w+border, :] = (255, 255, 255) # White Border
    combined[:h, w+border:, :] = defective_drawn
    
    cv2.putText(combined, "Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Inspection", (w+border+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # labels at the bottom
    text_y = h + 35
    cv2.putText(combined, "Undamaged", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Damaged", (w+border+10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imwrite(save_path, combined)
    return csv_data

# main
def main():
    torch.manual_seed(42); np.random.seed(42)
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # initialize csv
    csv_file_path = os.path.join(args.output_dir, 'inspection_results.csv')
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Defect Type', 'Confidence', 'Severity', 'Centroid X', 'Centroid Y'])
    
    # dataset
    full_ds = DeepPCBDataset(args.data_root, transform=get_transform(True))
    indices = torch.randperm(len(full_ds)).tolist()
    split = int(0.8 * len(full_ds))
    train_ds = torch.utils.data.Subset(full_ds, indices[:split])
    test_ds = torch.utils.data.Subset(DeepPCBDataset(args.data_root, transform=get_transform(False)), indices[split:])
    
    print(f"Split: {len(train_ds)} Train | {len(test_ds)} Test")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    # train
    model = get_model(7).to(DEVICE)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
    print(f"Training for {args.epochs} epochs...")
    for ep in range(args.epochs):
        model.train()
        losses = []
        for imgs, tgts in train_loader:
            imgs = [i.to(DEVICE) for i in imgs]
            tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            loss = sum(loss for loss in loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Epoch {ep+1}: Loss {sum(losses)/len(losses):.4f}")
        
    evaluate_model(model, test_loader, DEVICE)
    
    # saving result samples images and metrics in csv
    print("Saving 10 comparison samples and logging CSV...")
    sample_idxs = np.random.choice(len(test_ds), 10, replace=False)
    all_csv_rows = []
    for i, idx in enumerate(sample_idxs):
        img, _ = test_ds[idx]
        orig_idx = test_ds.indices[idx]
        temp_path = test_ds.dataset.templates[orig_idx]
        rows = generate_comparison_image(model, img, temp_path, os.path.join(args.output_dir, f"sample_{i+1}.jpg"))
        all_csv_rows.extend(rows)
    
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_csv_rows)
        
    print(f"Done. Images and CSV saved to {args.output_dir}")

if __name__ == "__main__":
    main()
