"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms
from model import Net
import torch
import torchvision
import os
from torch.utils import model_zoo

class ModelFactory:
    def __init__(self, model_name, model_path: str, num_classes=500):
        self.model_name = model_name
        self.model_path = model_path
        self.num_classes = num_classes
        
        if torch.cuda.is_available(): 
            self.use_cuda = True
            self.map_location = None
        else: 
            self.use_cuda = False
            self.map_location = torch.device('cpu') 
        
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic":
            return Net()
        elif "resnet50" in self.model_name:
            print("Using the ResNet50 architecture.")
            model = torchvision.models.resnet50(pretrained=False)
            checkpoint = torch.load(self.model_path, map_location=self.map_location)

            state_dict = checkpoint["state_dict"]
            if not self.use_cuda:
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict=state_dict)

            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, self.num_classes)

            if self.use_cuda:
                model = torch.nn.DataParallel(model).cuda()
            return model

        elif "vgg16" in self.model_name:

            print("Using the VGG-16 architecture.")
            assert os.path.exists(self.model_path), (
                "Please download the VGG model yourself from the following link and save it locally: "
                "https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK"
            )

            model = torchvision.models.vgg16(pretrained=False)
            checkpoint = torch.load(self.model_path, map_location=self.map_location)

            state_dict = checkpoint["state_dict"]
            if not self.use_cuda:
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict=state_dict)

            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze layers 26 and 28 in the features module
            model.features[26].requires_grad_(True)
            model.features[28].requires_grad_(True)

            # Unfreeze the last three linear layers in the classifier
            for i in [0, 3, 6]:  # Layers 0, 3, 6 in the classifier
                model.classifier[i].requires_grad_(True)

            # Replace the classifier output layer
            num_features = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_features, self.num_classes)

            if self.use_cuda:
                model.features = torch.nn.DataParallel(model.features).cuda()
                model = torch.nn.DataParallel(model).cuda()
            return model

        elif "alexnet" in self.model_name:
            print("Using the AlexNet architecture.")
            assert os.path.exists(self.model_path), (
                "Please download the AlexNet model yourself from the following link and save it locally: "
                "https://drive.google.com/drive/u/0/folders/1GnxcR6HUyPfRWAmaXwuiMdAMKlL1shTn"
            )

            model = torchvision.models.alexnet(pretrained=False)
            checkpoint = torch.load(self.model_path, map_location=self.map_location)

            state_dict = checkpoint["state_dict"]
            if not self.use_cuda:
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict=state_dict)

            num_features = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_features, self.num_classes)

            if self.use_cuda:
                model.features = torch.nn.DataParallel(model.features).cuda()
                model = torch.nn.DataParallel(model).cuda()
            return model

        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        return data_transforms        

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
