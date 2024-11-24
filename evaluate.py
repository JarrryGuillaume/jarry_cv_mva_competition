import argparse
import os

import PIL.Image as Image
import torch
from tqdm import tqdm
from torchvision import datasets
from main import validation
from model_factory import ModelFactory


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def test(
        model_name, 
        model_path, 
        outfile, 
        data, 
        batch_size=256, 
        num_workers=4, 
        hidden_size=30,
) -> None:
    """Main Function."""
    # options
 
    test_dir = data + "/test_images/mistery_category"

    # cuda
    use_cuda = torch.cuda.is_available()

    # load model and transform
    model, data_transforms = ModelFactory(model_name, model_path, hidden_size=hidden_size).get_all()
    model.eval()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(data + "/val_images", transform=data_transforms["val"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_loss, val_accuracy = validation(model, val_loader, use_cuda)
    print(f"Validation loss : {val_loss}, Validation accuracy : {val_accuracy} %")

    output_file = open(outfile, "w")
    output_file.write("Id,Category\n")
    for f in tqdm(os.listdir(test_dir)):
        if "jpeg" in f:
            data = data_transforms["val"](pil_loader(test_dir + "/" + f))
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
