import torch
import segmentation_models_pytorch as smp
from preprocess import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scipy.ndimage import distance_transform_edt
import numpy as np

class BoundaryLoss(nn.Module):
    """
    "Boundary loss for highly unbalanced segmentation" - Kervadec et al.
    """
    def __init__(self):
        super(BoundaryLoss, self).__init__()
    
    def compute_level_set_function(self, gt_mask):
        """
        Compute level set function φG from ground truth mask.
        φG(q) = -DG(q) if q ∈ G (inside foreground)
        φG(q) = +DG(q) if q ∉ G (outside foreground)
        
        Args:
            gt_mask: Ground truth binary mask [B, 1, H, W]
        Returns:
            phi_G: Level set function [B, 1, H, W]
        """
        gt_np = gt_mask.cpu().numpy()
        phi_maps = []
        
        for i in range(gt_np.shape[0]):  # batch dimension
            mask = gt_np[i, 0]  # remove channel dimension [H, W]
            
            # Skip if slice contains only background (all zeros)
            if not mask.any():
                phi_maps.append(np.zeros_like(mask))
                continue
            
            # Compute distance transform for foreground (inside)
            dist_inside = distance_transform_edt(mask)
            
            # Compute distance transform for background (outside) 
            dist_outside = distance_transform_edt(1 - mask)
            
            # Level set function: negative inside, positive outside
            phi = np.where(mask, -dist_inside, dist_outside)
            phi_maps.append(phi)
        
        return torch.tensor(np.stack(phi_maps), dtype=torch.float32, 
                          device=gt_mask.device).unsqueeze(1)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Network predictions [B, 1, H, W] (logits)
            targets: Ground truth binary masks [B, 1, H, W]
        Returns:
            Boundary loss value
        """
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)
        
        # Compute level set function φG
        phi_G = self.compute_level_set_function(targets)
        
        # Boundary loss: ∫ φG(q) * sθ(q) dq
        boundary_loss = torch.mean(phi_G * predictions)
        
        return boundary_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Network predictions [B, 1, H, W] (logits)
            targets: Ground truth binary masks [B, 1, H, W]
        Returns:
            Dice loss value (1 - Dice coefficient)
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    Combined Boundary + Dice Loss with alpha scheduling
    """
    def __init__(self, alpha_schedule='rebalance', initial_alpha=0.01, alpha_increment=0.01):
        super(CombinedLoss, self).__init__()
        self.boundary_loss = BoundaryLoss()
        self.dice_loss = DiceLoss()
        self.alpha_schedule = alpha_schedule
        self.initial_alpha = initial_alpha
        self.alpha_increment = alpha_increment
        self.current_epoch = 0
        self.alpha = initial_alpha
    
    def update_alpha(self, epoch):
        """
        Update alpha according to scheduling strategy
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        
        if self.alpha_schedule == 'constant':
            # Keep alpha constant
            self.alpha = self.initial_alpha
        
        elif self.alpha_schedule == 'increase':
            # Increase alpha over time: LR + α*LB
            self.alpha = self.initial_alpha + self.alpha_increment * epoch
        
        elif self.alpha_schedule == 'rebalance':
            # Rebalance strategy: (1-α)*LR + α*LB (recommended)
            self.alpha = min(self.initial_alpha + self.alpha_increment * epoch, 1.0)
        
        else:
            raise ValueError(f"Unknown alpha_schedule: {self.alpha_schedule}")
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Network predictions [B, 1, H, W] (logits)
            targets: Ground truth binary masks [B, 1, H, W]
        Returns:
            Combined loss value
        """
        dice_loss = self.dice_loss(predictions, targets)
        boundary_loss = self.boundary_loss(predictions, targets)
        
        if self.alpha_schedule == 'rebalance':
            # (1-α)*Dice + α*Boundary - gradually shift from Dice to Boundary
            combined_loss = (1 - self.alpha) * dice_loss + self.alpha * boundary_loss
        else:
            # Dice + α*Boundary - add boundary term with increasing weight
            combined_loss = dice_loss + self.alpha * boundary_loss
        
        return combined_loss
    
    def get_loss_components(self, predictions, targets):
        """
        Get individual loss components for monitoring
        
        Returns:
            dict with 'dice', 'boundary', 'combined', 'alpha' values
        """
        dice_loss = self.dice_loss(predictions, targets)
        boundary_loss = self.boundary_loss(predictions, targets)
        combined_loss = self.forward(predictions, targets)
        
        return {
            'dice': dice_loss.item(),
            'boundary': boundary_loss.item(), 
            'combined': combined_loss.item(),
            'alpha': self.alpha
        }

def train(folder_names,  name = "Unet",parent_folder = "./annotated/Human" ,withAug = True, human = True):
    if human:
         name = name + '_human'
    dataset = Data_Generator(withAug)
    print("What:", folder_names)
    for f in folder_names:
        if os.path.isdir(os.path.join(parent_folder, f)):
            dataset.load_data(os.path.join(parent_folder, f))
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    model_type = name.split("_")[0]
    model = getattr(smp, model_type)
    model = model(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=1,                      
    )
    criterion = CombinedLoss(
        alpha_schedule='rebalance',  # 'constant', 'increase', or 'rebalance'
        initial_alpha=0.01,          # Starting alpha value
        alpha_increment=0.01         # How much to increase alpha each epoch
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(torch.__version__)                
    print(torch.cuda.is_available())       
    print(torch.cuda.get_device_name(0))     

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10  
    print(f"----------------------------------{name}----------------------------------")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1).float()  # Add channel dimension

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(loader):.4f}")
    torch.save(model.state_dict(),("human/" if human else "rat/" )+ name+ ("_aug" if withAug else "") + ".pth")
    model.name = name
    print(f"--------------------------------------------------------------------")
    return model, dataset

def load_model(path , device = torch.device("cuda"), parent_folder = "./models"):
    
    name = os.path.basename(path)
    model_type = name.split("_")[0]
    model = getattr(smp, model_type)
    
    model = model(
    encoder_name="resnet34",
    encoder_weights=None,  
    in_channels=3,
    classes=1
)
    model.name = os.path.splitext(name)[0]
    model.load_state_dict(torch.load(os.path.join(parent_folder, path)))
    model.to(device)
    model.eval()
    return model

if __name__ == '__main__':

    model = load_model("UnetPlusPlus_human_aug.pth",parent_folder = "./models" )
    with open("Visualize/UPP.txt", 'w') as f:
            f.write(str(model))
    