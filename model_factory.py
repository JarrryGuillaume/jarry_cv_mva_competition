"""Python file to instantite the model and the transform that goes with it."""
from torchvision import transforms
from data import data_transforms
from model import Net
import torch
import torchvision
import os
from torch.utils import model_zoo
import timm

class ModelFactory:
    def __init__(self, model_name, model_path: str, tuning_layers, fine_tune=True,  num_classes=500):
        self.model_name = model_name
        self.model_path = model_path
        self.num_classes = num_classes
        self.fine_tune = fine_tune
        self.tuning_layers = tuning_layers
        
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
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict=state_dict)

            if self.fine_tune:
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
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict=state_dict)

            if self.fine_tune: 
                for name, param in model.named_parameters():
                    param.requires_grad = False

                # Unfreeze layers 26 and 28 in the features module
                for layer in self.tuning_layers: 
                    print(f"layers modified : {layer}")
                    model.features[layer].requires_grad_(True)

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
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict=state_dict)

            if self.fine_tune: 
                num_features = model.classifier[6].in_features
                model.classifier[6] = torch.nn.Linear(num_features, self.num_classes)

            if self.use_cuda:
                model.features = torch.nn.DataParallel(model.features).cuda()
                model = torch.nn.DataParallel(model).cuda()
            return model
    
        elif "vit" in self.model_name.lower():
            print("Using the Vision Transformer architecture.")
            # Load the pre-trained ViT-B/16 model pre-trained on ImageNet-21k
            model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
            
            if self.fine_tune:
                # Replace the classification head
                num_features = model.head.in_features
                model.head = torch.nn.Linear(num_features, self.num_classes)
            
            if self.use_cuda:
                model = torch.nn.DataParallel(model).cuda()
            return model
        
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if "vit" in self.model_name.lower():
            data_transforms = {
                "train": transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]),
                "val": transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]),
            }
        else:
            data_transforms = {
                "train": transforms.Compose([
                    transforms.Resize((256, 256)),                # Ensure consistent input size
                    transforms.RandomCrop((224, 224)),           # Random crop while keeping most of the object
                    transforms.RandomHorizontalFlip(),           # Simulate horizontal flipping
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomRotation(degrees=30),       # Small random rotations for variability
                    transforms.ToTensor(),                       # Convert to PyTorch tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]),  # Normalize
                ]),
                "val": transforms.Compose([
                    transforms.Resize((224, 224)),               # Resize directly for validation
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]),
                ]),
            }
        return data_transforms

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
