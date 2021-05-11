import torch
import torchvision
import torch.nn as nn
from torchvision.models import vgg16
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
device = torch.device('cuda')
from torchvision.utils import save_image

VGG = vgg16(pretrained=True)


class VGG_Net(nn.Module):
    def __init__(self):
        super(VGG_Net, self).__init__()
        self.chosen_layers = ['0','5','10','17','24']
        self.model = vgg16(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for lay_num, layer in enumerate(self.model):
            x = layer(x)

            if str(lay_num) in self.chosen_layers:
                features.append(x)

        return features

loader = transforms.Compose([
    transforms.Resize((356,356)),
    transforms.ToTensor()
])
def load_image(image):
    image = Image.open(image)
    image = loader(image).unsqueeze(0)
    return image

content = load_image('/home/kartikeyac/Documents/Style_Transfer/content_image.jpg').to(device)
style = load_image('/home/kartikeyac/Documents/Style_Transfer/index.jpg').to(device)
gene = load_image('/home/kartikeyac/Documents/Style_Transfer/content_image.jpg').to(device)
gene = torch.tensor(gene, requires_grad = True)
alpha = 1
beta = 0.01
model = VGG_Net().to(device)
Optim = optim.Adam([gene],lr = 1e-3)
for epoch in range(1,10001):
    gen_feat = model(gene)
    orig_feat = model(content)
    style_feat = model(style)
    style_loss = orig_loss = 0
    for a,b,c in zip(gen_feat, orig_feat, style_feat):
        batch_size, channels, height, width = a.shape
        
        orig_loss += torch.mean((a - b) ** 2)

        G = a.view(channels, height * width).mm(
            a.view(channels, height * width).t()
        )

        S = c.view(channels, height * width).mm(
            c.view(channels, height * width).t()
        )

        style_loss += torch.mean((G - S) ** 2)

    total_loss = alpha * orig_loss + beta * style_loss
    Optim.zero_grad()
    total_loss.backward()
    Optim.step()

    if epoch % 1000 == 0:
        print(f'the total loss for epoch {epoch} is {total_loss}')
        save_image(gene, f'Generated_epoch_{epoch}.png')



