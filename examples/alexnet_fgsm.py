import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

from torchattack.attacks import FGSMAttack

model = models.alexnet(pretrained=True).eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
dataset = datasets.ImageFolder(
    "test_data", 
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

fgsm = FGSMAttack(model, mean=mean, std=std)

for x, t in loader:
    y = model(x).argmax()
    x_adv = fgsm.attack(x, t)
    y_adv = model(x_adv).argmax()
    print('groundtruth label: {:d}, initial prediction: {:d}, final prediction: {:d}'.format(t.item(), y.item(), y_adv.item()))
