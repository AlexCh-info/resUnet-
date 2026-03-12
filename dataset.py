import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
from pathlib import Path
import random

class Data(Dataset):
    def __init__(self, input_dir: Path, gt_dir: Path, defect_names: list, img_size=256, train=True):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.defect_names = defect_names
        self.img_size = img_size
        self.train = train

        # create pairs
        self.pairs = []
        gt_nm_sf = [(x.stem, x.suffix) for x in self.gt_dir.iterdir() if x.exists() and x.suffix.lower in ['.jpg', '.jpeg', '.png']]

        for stem , suf in gt_nm_sf:
            for defect in defect_names:
                input_path = self.input_dir / f'{stem}_{defect}{suf}'
                gt_path = self.gt_dir / f'{stem}{suf}'
                if gt_path.exists() and input_path.exists():
                    self.pairs.append((input_path, gt_path))
        print(f'Find {len(self.pairs)} pairs')

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        input_path, gt_path = self.pairs[item]

        input_img = Image.open(input_path).convert('RGB')
        idl_img = Image.open(gt_path).convert('RGB')

        if self.train:
            #Flip
            if random.random() > 0.5:
                input_img = F.hflip(input_img)
                idl_img = F.hflip(idl_img)

            #Rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                input_img = F.rotate(input_img, angle)
                idl_img = F.rotate(idl_img, angle)

            #Brightness/Contrast
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                input_img = F.adjust_brightness(input_img, factor)

        input_tensor = self.transform(input_img)
        idl_tensor = self.transform(idl_img)

        return input_tensor, idl_tensor

def get_dataloaders(input_dir: Path, gt_dir: Path, val_input_dir: Path, val_gt_dir: Path,
                    defect_names: list, batch_size=2, img_size=256,
                    num_workers=2, pin_memory=True):
    '''
    Create train and val dataloaders with memory optimization
    :param input_dir: train imgs path
    :param gt_dir: train gt imgs path
    :param val_input_dir: val imgs path
    :param val_gt_dir: val gt imgs path
    :param defect_names: list of defects names
    :param batch_size: batch size
    :param img_size: images height and width
    :param num_workers: number of workers
    :param pin_memory: economy of memory On/Off
    :return: dataloaders
    '''
    train_dataset = Data(input_dir, gt_dir, defect_names,
                         img_size=img_size, train=True)
    val_dataset = Data(val_input_dir, val_gt_dir, defect_names,
                       img_size=img_size, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return train_loader, val_loader