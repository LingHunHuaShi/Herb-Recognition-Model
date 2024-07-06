import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
import os
from PIL import Image


def create_image_list(root_dir, output_file):
    class_names = sorted(os.listdir(root_dir))
    with open(output_file, 'w') as file:
        for idx, class_name in enumerate(class_names):
            if class_name == '.DS_Store':
                continue
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name == '.DS_Store':
                    continue
                img_path = os.path.join(class_dir, img_name)
                # 检查图像格式并转换为RGB
                if img_name.lower().endswith(('.png', '.webp')):
                    try:
                        # 加载图像并转换为RGB
                        img = Image.open(img_path).convert('RGB')
                        img.save(img_path)  # 覆盖原图
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
                # 写入文件路径和标签
                file.write(f"{img_path},{idx}\n")


class HerbDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.image_labels = []
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(',')
                self.image_labels.append((parts[0], int(parts[1])))
            self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path, label = self.image_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    root_dir = './Herb_images'
    output_file = './image_list.txt'
    create_image_list(root_dir, output_file)
