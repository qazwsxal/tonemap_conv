import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torch.nn.functional import mse_loss
from torch.optim import SGD
from tonemappers import *
EXPOSURE = 4.0
plt.ioff()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


curr_tm = UC2().to(device)
new_tm = ACES().to(device)

pil_image = Image.open("Laser_1_Blue.png")
convert_tensor = transforms.ToTensor()

in_image = convert_tensor(pil_image).transpose(-1,0).to(device)
out_image = torch.ones_like(in_image)/2
out_image.requires_grad = True
optim = SGD((out_image,), lr=1e-3)

curr_tm_im = curr_tm(in_image * EXPOSURE)
plt.imshow(curr_tm_im.cpu().numpy())
plt.show(block=False)
plt.pause(0.001)

fig,ax = plt.subplots()
for i in range(10000):
    new_tm_im = new_tm(out_image*EXPOSURE)
    loss = (curr_tm_im - new_tm_im).pow(2).sum()
    loss.backward()
    optim.step()
    optim.zero_grad()
    if (i % 1000) == 0:
        print(loss.item())
ax.imshow(new_tm_im.detach().cpu().numpy())
plt.show()