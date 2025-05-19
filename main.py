import torch
import segmentation_models_pytorch as smp
from preprocess import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import measure
from tabulate import tabulate
from fpdf import FPDF
from models import *


def iou_score(preds, masks):
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = (preds + masks).sum(dim=(1, 2, 3)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

def dice_score(preds, masks):
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-6) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6)).mean().item()

def pixel_accuracy(preds, masks):
    correct = (preds == masks).sum(dim=(1, 2, 3))
    total = torch.numel(preds[0]) * preds.shape[0]
    return (correct.sum().float() / total).item()

def boundary_f1_score(preds, masks):
    preds_np = preds.squeeze(1).cpu().numpy().astype(np.uint8)
    masks_np = masks.squeeze(1).cpu().numpy().astype(np.uint8)
    scores = []

    for p, m in zip(preds_np, masks_np):
        p_edges = measure.find_contours(p, 0.5)
        m_edges = measure.find_contours(m, 0.5)
        if not p_edges or not m_edges:
            continue
        # Approximate: use number of contours as edge overlap (not ideal but illustrative)
        score = min(len(p_edges), len(m_edges)) / max(len(p_edges), len(m_edges))
        scores.append(score)

    return np.mean(scores) if scores else 0.0

# Evaluation function
def evaluation(models, val_data, device=torch.device('cuda')):
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

    results = []

    for model in models:
        print("Debug: ", model.name)
        total_iou = 0
        total_dice = 0
        total_acc = 0
        total_boundary_f1 = 0
        num_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).float()
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)

                outputs = model(images)
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                total_iou += iou_score(preds, masks)
                total_dice += dice_score(preds, masks)
                total_acc += pixel_accuracy(preds, masks)
                total_boundary_f1 += boundary_f1_score(preds, masks)
                num_batches += 1

        avg_iou = total_iou / num_batches
        avg_dice = total_dice / num_batches
        avg_acc = total_acc / num_batches
        avg_boundary_f1 = total_boundary_f1 / num_batches

        results.append([
            model.name,
            round(avg_iou, 4),
            round(avg_dice, 4),
            round(avg_acc, 4),
            round(avg_boundary_f1, 4)
        ])
    headers = ["Model Name", "IoU", "Dice", "Pixel Acc", "Boundary F1"]
    return tabulate(results, headers=headers, tablefmt="fancy_grid")


def visualize_prediction(model, val_data,device = torch.device("cuda"), label = True):
    model.eval()
    if label == True:
        val_data = DataLoader(data.val, batch_size=2, shuffle=False)
        with torch.no_grad():
            for images, masks in val_data:
                images = images.to(device)
                
                masks = masks.to(device).float() 

                if masks.dim() == 3:  # (B, H, W)
                        masks = masks.unsqueeze(1)  # -> (B, 1, H, W)

                outputs = model(images)
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                for i in range(images.size(0)):
                    
                    img = images[i].cpu().permute(1, 2, 0).numpy()  # CxHxW -> HxWxC

                    fig, axs = plt.subplots(1, 5, figsize=(12, 4))
                    fig.suptitle(f"{model.name}: sample {i+1}", fontsize=16) 
                    axs[0].imshow(img)
                    axs[0].set_title("Input Image")

                    gt_mask = masks[i][0].cpu().numpy()
                    axs[1].imshow(gt_mask, cmap='gray')
                    axs[1].set_title("Ground Truth")

                    pred_mask = preds[i][0].cpu().numpy()
                    axs[2].imshow(pred_mask, cmap='gray')
                    axs[2].set_title("Prediction")


                    gt_overlay = merge(img, gt_mask, color=(0, 255, 0), alpha=0.4)
                    axs[3].imshow(gt_overlay)
                    axs[3].set_title("Ground Truth Overlay")


                    pred_overlay = merge(img, pred_mask, color=(255, 0, 0), alpha=0.4)
                    axs[4].imshow(pred_overlay)
                    axs[4].set_title("Prediction Overlay")

                    for ax in axs:
                        ax.axis('off')

                    plt.tight_layout()
                    plt.show()

                break
    else:
         with torch.no_grad():
            for img, _ in val_data:
                img = img.unsqueeze(0).to(device)


                outputs = model(img)
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                for i in range(min( 3, img.size(0))):
                    img = img[i].cpu().permute(1, 2, 0).numpy()  # CxHxW -> HxWxC

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    fig.suptitle(f"{model.name}: sample {i+1}", fontsize=16) 
                    axs[0].imshow(img)
                    axs[0].set_title("Input Image")


                    pred_mask = preds[i][0].cpu().numpy()
                    axs[1].imshow(pred_mask, cmap='gray')
                    axs[1].set_title("Prediction")



                    pred_overlay = merge(img, pred_mask, color=(255, 0, 0), alpha=0.4)
                    axs[2].imshow(pred_overlay)
                    axs[2].set_title("Prediction Overlay")

                    for ax in axs:
                        ax.axis('off')

                    plt.tight_layout()
                    plt.show()

                break


def merge(image, mask, color=(0, 0,255), alpha=0.5):
        """
        image: RGB image (H, W, 3)
        mask: 2D binary or grayscale mask (H, W)
        color: tuple, RGB color for overlay (red default)
        alpha: transparency factor (0.0 to 1.0)
        """
        overlay = image.copy()
        red_mask = np.zeros_like(image)
        red_mask[mask > 0] = color  # Only where mask is present

        # Blend the original image with the red mask
        cv2.addWeighted(red_mask, alpha, overlay, 1 - alpha, 0, overlay)
        return overlay

if __name__ == '__main__':
    
    import multiprocessing
    multiprocessing.freeze_support() 
    #parent_folder = "./output/rat"
    parent_folder = "./annotated/Human"
    folder_names = [name for name in os.listdir(parent_folder)
                    if os.path.isdir(os.path.join(parent_folder, name))]
    print(folder_names)
    test = False
    name = 'PSPNet_human'
    if test == True:
        data = Data_Generator(False, True)
        data.load_data(os.path.join(parent_folder, folder_names[0]))
    else:
        model, data = train(folder_names[1:],name = name , withAug = True)

    models = [load_model(f) for f in os.listdir("./models")
             if os.path.isfile(os.path.join("./models", f)) and f.endswith('.pth')]

    res = evaluation(models,data.val)
    with open("Visualize/compare.txt",'w') as f:
        f.write(res)
    #visualize_prediction(models[0],data.val, label=True)
    #visualize_prediction(models[1],data.val, label=True)
    