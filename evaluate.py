import argparse
import os

import PIL.Image as Image
import torch
from tqdm import tqdm

from model_factory import ModelFactory


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def test(
        model, 
        model_name, 
        model_path,
        outfile, 
        test_dir,
) -> None:
    """Main Function."""
    # cuda
    use_cuda = torch.cuda.is_available()

    # load model and transform
    state_dict = torch.load(model)
    model, data_transforms = ModelFactory(model_name, model_path).get_all()
    model.load_state_dict(state_dict)
    model.eval()

    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    output_file = open(outfile, "w")
    output_file.write("Id,Category\n")
    for f in tqdm(os.listdir(test_dir)):
        if "jpeg" in f:
            data = data_transforms(pil_loader(test_dir + "/" + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            if use_cuda:
                data = data.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            output_file.write("%s,%d\n" % (f[:-5], pred))

    output_file.close()

    print(
        "Succesfully wrote "
        + outfile
        + ", you can upload this file to the kaggle competition website"
    )
