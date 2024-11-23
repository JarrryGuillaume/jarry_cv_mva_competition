import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from IPython.display import clear_output
from model_factory import ModelFactory


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    args = parser.parse_args()
    return args

import matplotlib.pyplot as plt

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    log_interval: int,
) -> float:
    """Default Training Loop."""
    model.train()
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        # Optionally log detailed info (comment out for cleaner output)
        if batch_idx % log_interval == 0:
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.data.item(),
                    )
                )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / len(train_loader.dataset)
    return avg_loss, accuracy


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> float:
    """Default Validation Loop."""
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    avg_loss = validation_loss / len(val_loader)
    accuracy = 100.0 * correct / len(val_loader.dataset)
    return avg_loss, accuracy


def main(data_folder="../mva_competition", 
            experiment_folder="../experiment",
            model_name="basic",
            model_path=None, 
            batch_size=32,
            num_workers=4,
            momentum=0.5,
            epochs=100,
            seed=42,
            lr=1e-3,
            saving_frequency=5,
            log_interval=10, 
            fine_tune=True, 
            optimizer="SGD", 
            tuning_layers=None):
    """Default Main Function."""
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(seed)

    # Create experiment folder
    if not os.path.isdir(experiment_folder):
        os.makedirs(experiment_folder)

    # load model and transform
    model, data_transforms = ModelFactory(model_name, model_path, tuning_layers, fine_tune=fine_tune).get_all()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(data_folder + "/train_images", transform=data_transforms["train"]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(data_folder + "/val_images", transform=data_transforms["val"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Setup optimizer
    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer == "AdamW": 
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(1, epochs + 1):
        # Training loop
        train_loss, train_accuracy = train(model, optimizer, train_loader, use_cuda, epoch, log_interval)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation loop
        val_loss, val_accuracy = validation(model, val_loader, use_cuda)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch : {epoch}, Training accuracy : {train_accuracy} %")
        print(f"Epoch : {epoch}, Validation accuracy : {val_accuracy} %")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_file = f"{experiment_folder}/best_{model_name}.pth"
            torch.save(model.state_dict(), best_model_file)

        # Save model periodically
        if epoch % saving_frequency == 0:
            model_file = experiment_folder + f"/{model_name}_{epoch}.pth"
            torch.save(model.state_dict(), model_file)
    

if __name__ == "__main__":
    main()
