import torch
import segmentation_models_pytorch as smp
from preprocess import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def train(folder_names,  name = "Unet_human",parent_folder = "./annotated/Human" ,withAug = True):
    dataset = Data_Generator(withAug)
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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(torch.__version__)                
    print(torch.cuda.is_available())       
    print(torch.cuda.get_device_name(0))     

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10  

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
    torch.save(model.state_dict(),"models/"+ name+ ("_aug" if withAug else "") + ".pth")
    return model, dataset

def load_model(path , device = torch.device("cuda")):
    parent_folder = "./models"

    name = os.path.basename(path)
    model_type = name.split("_")[0]
    model = getattr(smp, model_type)
    
    model = model(
    encoder_name="resnet34",
    encoder_weights=None,   # Use None when loading pretrained weights manually
    in_channels=3,
    classes=1
)
    model.name = os.path.splitext(name)[0]
    model.load_state_dict(torch.load(os.path.join(parent_folder, path)))
    model.to(device)
    model.eval()
    return model

