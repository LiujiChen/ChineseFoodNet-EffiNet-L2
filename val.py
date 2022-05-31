import torch
from PIL import Image
from torchvision.transforms import transforms

from model.wide_res_net import WideResNet

if __name__ == '__main__':
    model = WideResNet(8, 8, 0.0, in_channels=3, labels=208)
    model.load_state_dict(torch.load('./instance/model.pt'))
    model.eval()
    img = Image.open('./ChineseFoodNet/release_data/train/000/000085.jpg').convert('RGB')
    img = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52011104, 0.44459117, 0.30962785], std=[0.25595631, 0.25862494, 0.26925405])
    ])(img)
    img = img.reshape(1, 3, 128, 128)
    pre = model(img)
    print(pre)
    correct = torch.argmax(pre, 1)
    print(correct)

