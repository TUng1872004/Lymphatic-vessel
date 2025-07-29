import cv2
import os
import numpy as np
import albumentations as A
import json
from torch.utils.data import Dataset
import random


class Data_Generator(Dataset):
    def __init__(self, withAug = True ):
        self.data = []
        self.pic = []
        self.val =[]
        self.transform = A.Compose([
        A.Compose([
        A.Resize(
            height=256,
            width=256,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            area_for_downscale="image",  # Use INTER_AREA when downscaling images
            p=1.0
        )
    ]),
        #A.SmallestMaxSize(max_size=256, p=1.0),
        #A.RandomCrop(height=256, width=256),
        A.Compose([
            A.OneOf(
                [
                A.Affine(rotate=(0,180),border_mode=cv2.BORDER_REFLECT_101, p=1.0),  # Rotate 
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Compose([
                    A.RandomScale(scale_limit=0.2, p=0.5),  
                    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REPLICATE),
                    A.CenterCrop(height=256, width=256, p=1.0)
                ])
                ]
            ),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            #A.OneOf([
            #    A.GridDistortion(num_steps=5, distort_limit=(0.3, 0), p=1.0), 
            #    A.ElasticTransform(alpha=350, sigma=20, p=0.5),
            #    A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.5),
            #]),

            A.OneOf([
                A.GaussNoise(std_range=((5.0/255), (20.0/255)), mean_range=(0.0, 0.0), p=0.5),
                A.SaltAndPepper(p=0.5)
            ],p=0.7)
            ]),
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2() 
        ]) if withAug else None

        self.normalize = A.Compose([
                A.Resize(
                height=256,
                width=256,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                area_for_downscale="image",  # Use INTER_AREA when downscaling images
                p=1.0
            )
            ,
                #A.SmallestMaxSize(max_size=256, p=1.0),
                #A.RandomCrop(height=256, width=256),
                A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)),
                A.pytorch.ToTensorV2() 
        ])

    def annotation(self,json_path):
        if not os.path.exists(json_path):
                print(f"[Skipping] Missing JSON for: {json_path}")
                return None
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        height, width = data.get('imageHeight'), data.get('imageWidth')
        mask = np.zeros((height, width), dtype=np.uint8)

        for shape in data['shapes']:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], 1) 
        return mask

    def extract_frames(self, video_name, output, start, end, FPS=0.0005, human=True, save = False):
        human_vid = "./data/Human Lymphatics-selected/"
        rat_vid = "./data/Rat Lymphatics"
        video_path = human_vid if human else rat_vid

        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        full_path = None

        for ext in video_extensions:
            trial_path = os.path.join(video_path, video_name + ext)
            if os.path.exists(trial_path):
                full_path = trial_path
                break

        if full_path is None:
            print(f"Video file not found for {video_name} in: {video_path}")
            return

        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            print("Error: Cannot open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps

        start = max(0, start)
        end = min(duration, end)

        print(f"Video: {video_name}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total}")
        print(f"Video duration: {duration:.2f} seconds")
        print(f"Extracting from {start:.2f}s to {end:.2f}s")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📹 Resolution: {width}x{height}")

        if not os.path.exists(output):
            os.makedirs(output)

        interval = int(fps / FPS)
        print("Interval: ",interval)
        idx = int(start * fps)
        max_frame = int(end * fps)
        saved = 0
        frames = []
        while idx <= max_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            time_sec = idx / fps
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)
            milliseconds = int((time_sec - int(time_sec)) * 1000)

            if save:
                frame_name = f"{minutes:02d}_{seconds:02d}_{milliseconds:03d}.png"
                frame_path = os.path.join(output, frame_name)
                cv2.imwrite(frame_path, frame)
                saved += 1

            frames.append(frame)
            idx += interval

        cap.release()
        print(f"✅ Done: {saved} frames saved.")
        return frames

    def apply(self,frame, mask, trans = True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=gray, mask=mask) if trans and self.transform is not None else self.normalize(image=gray, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        return image, mask
    
    def test(self,frame, mask, trans):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = trans(image=gray, mask=mask) 
        image = augmented["image"]
        mask = augmented["mask"]
        return image, mask
    

    def __len__(self):
        return len(self.data)

    def load_data(self, image_dir, validate = False):
        image_paths = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        count = 0
        
        for i in image_paths:


            json_filename = i.replace(".png", ".json")

            image_path = os.path.join(image_dir, i)
            json_path = os.path.join(image_dir, json_filename)

            image = cv2.imread(image_path)
            mask = self.annotation(json_path)
            if mask is None:
                continue
            if validate == True:
                self.val.append(self.apply(image, mask, False))
            else:
                self.data.append(self.apply(image, mask, False))
                if self.transform is not None:
                    for i in range(10):
                        image_aug, mask_aug = self.apply(image,mask)
                        self.data.append((image_aug, mask_aug))
            self.pic.append((image, mask))
            count+=1


    def __getitem__(self,idx):
        return self.data[idx]
    def merge(self, image, mask, color=(0, 0,255), alpha=0.5):
        """
        image: RGB image (H, W, 3)
        mask: 2D binary or grayscale mask (H, W)
        color: tuple, RGB color for overlay (red default)
        alpha: transparency factor (0.0 to 1.0)
        """
        overlay = image.copy()
        red_mask = np.zeros_like(image)
        red_mask[mask > 0] = color  # Only where mask is present

        cv2.addWeighted(red_mask, alpha, overlay, 1 - alpha, 0, overlay)
        return overlay


if __name__ == '__main__':
    vid_name = "Human_Lymphatic_02-27-24_pressure_0Ca_scan_East2"
    img_dir = "./annotated/Human/" + vid_name
    loader = Data_Generator(withAug = False)
    loader.load_data(img_dir,n=1)

    '''
    A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=(0.2, 0), p=1.0), 
                A.ElasticTransform(alpha=2, sigma=75, p=0.5),
                A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.5),
            ])'''
    img, mask = loader.pic[0]
    
    cv2.imwrite("test/normal.png",img)
    '''
    if mask is not None:
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        if mask.ndim ==3 and mask.shape[2]==1:
            mask=mask.squeeze()
        cv2.imwrite("test/mask.png",mask)
    cv2.imwrite("test/merge.png",loader.merge(img, mask))
    '''



