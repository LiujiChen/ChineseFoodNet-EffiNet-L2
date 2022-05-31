from utils.ChineseFoodNetSet import ChineseFoodNetTrainSet, ChineseFoodNetTestSet
from PIL import Image
from torchvision.transforms import transforms
if __name__ == '__main__':
    img = Image.open('./ChineseFoodNet/release_data/train/000/000085.jpg')
    img = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(128)
    ])(img)
    img.show()