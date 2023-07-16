import argparse
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from models.cyclegan import Generator

def main():
    parser = argparse.ArgumentParser(description='Converts a an old damaged image into a new colored image')
    parser.add_argument('-r', '--reverse', action='store_true', default=False, help='Reverses the process (corrected image --> old image)')
    parser.add_argument('-i', '--intermediate', action='store_true', default=False, help='Output the intermediate states')
    parser.add_argument('-x', '--single-domain', action='store_true', default=False, help='Use the single domain-to-domain GAN')
    parser.add_argument('input_file', nargs='+', help='input file(s)')
    args = parser.parse_args()
    r = args.reverse

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = torch.load("ganban.pth", map_location=device)

    flow = []
    if args.single_domain:
        flow.append(Generator(3).to(device))
        flow[-1].load_state_dict(models["fixedcolor2broken" if r else "broken2fixedcolor"])
        flow[-1].eval()
    else:
        flow.append(Generator(3).to(device))
        flow[-1].load_state_dict(models["fixed2broken" if r else "broken2fixed"])
        flow[-1].eval()
        flow.append(Generator(3).to(device))
        flow[-1].load_state_dict(models["color2gray" if r else "gray2color"])
        flow[-1].eval()
    if r:
        flow.reverse()

    tf = transforms.Compose([
        #transforms.Resize((256, 256), Image.Resampling.BICUBIC),
        transforms.Resize(256, Image.Resampling.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # can make this faster if you use dataloader i think
    for f_image in args.input_file:
        filename = os.path.splitext(f_image)[0]
        input_image = Image.open(f_image).convert("RGB")
        img_tensor = tf(input_image).unsqueeze(0).to(device)

        if args.intermediate:
            save_image(img_tensor, f"{filename}.0.png", normalize=True)

        with torch.no_grad():
            for i, gen in enumerate(flow):
                img_tensor = gen(img_tensor)
                if args.intermediate and i != (len(flow) - 1):
                    save_image(img_tensor, f"{filename}.{i + 1}.png", normalize=True)
        save_image(img_tensor, f"{filename}.out.png", normalize=True)

if __name__ == "__main__":
    main()
