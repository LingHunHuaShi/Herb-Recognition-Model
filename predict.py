import network.ResNet50 as ResNet
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

labels = ['丹参', '人参', '党参', '地黄', '川穹', '当归', '枸杞', '柴胡', '桂枝', '槐花', '甘草', '生姜', '白术',
          '白芷', '茯苓', '薄荷', '陈皮', '黄芩', '黄芪', '黄连']


def load_model():
    model = ResNet.ResNet(20, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("save/best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint['weight'])

    model.eval()
    print("Model Loaded, on device {}".format(device))
    return model.to(device)


def preprocess(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize(236, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    image = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式
    image = transform(image)
    image = image.unsqueeze(0)  # 增加batch维度
    return image


def predict(image_path):
    model = load_model()
    input_tensor = preprocess(image_path)
    with torch.no_grad():
        output = model(input_tensor)

    label_index = torch.argmax(output, dim=1)
    label = labels[label_index[0]]
    confidence = output[label_index[0]]
    return label, confidence


if __name__ == '__main__':
    image_path = 'input_sample/柴胡1.webp'
    label, confidence = predict(image_path)
    print(f"label: {label}, confidence: {confidence}")
