import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class dataset(Dataset):
    def __init__(self, root_path, split='train'):
        self.root_path = root_path
        
        if split == 'train':
            image_dir = os.path.join(root_path, "train_tp")
            mask_dir = os.path.join(root_path, "train_gt")
        elif split == 'validation':
            image_dir = os.path.join(root_path, "validation_tp")
            mask_dir = os.path.join(root_path, "validation_gt")
        elif split == 'test':
            image_dir = os.path.join(root_path, "test_tp")
            mask_dir = os.path.join(root_path, "test_gt")
        else:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        
        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Directories {image_dir} or {mask_dir} do not exist")
        
        self.images = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('jpg', 'tif'))])
        self.masks = sorted([os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir) if mask.endswith(('png'))])

        if len(self.images) == 0 or len(self.masks) == 0:
            raise FileNotFoundError(f"No images or masks found in {image_dir} or {mask_dir}")

        if len(self.images) != len(self.masks):
            raise ValueError("Number of images and masks must be equal")

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        if self.image_transform is not None:
            img = self.image_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)


        # Binarize the mask
        mask = (mask > 0.5).float()

        return img, mask

    def __len__(self):
        return len(self.images)