"""Python file to instantite the model and the transform that goes with it."""
from torchvision import transforms
from data import data_transforms
from model import Net
import torch
import torchvision
import torch.nn as nn
import os
from torch.utils import model_zoo
import timm
import torchvision.models as models

class ModelFactory:
    def __init__(self, model_name, model_type=None, model_path=None, tuning_layers=None, fine_tune=False, hidden_size=None, dropout=0.2,  num_classes=500):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        self.num_classes = num_classes
        self.fine_tune = fine_tune
        self.tuning_layers = tuning_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        
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
        
        elif "efficientNet" in self.model_name:
            model = models.efficientnet_b0(pretrained=False, num_classes=self.num_classes) 

            if self.model_path is not None: 
                state_dict = torch.load(self.model_path)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                print("Succesfully loaded weights")

            if self.use_cuda:
                model = torch.nn.DataParallel(model).cuda()
            print("efficientNet used")
            return model
            
        elif "mobileNet" in self.model_name: 
            model = models.mobilenet_v2(pretrained=False, num_classes=self.num_classes)

            if self.model_path is not None: 
                state_dict = torch.load(self.model_path)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                print("Succesfully loaded weights")

            if self.use_cuda:
                model = torch.nn.DataParallel(model).cuda()
            print("mobileNet used")
            return model
        
        elif "squeezeNet" in self.model_name: 
            model = models.squeezenet1_0(pretrained=False, num_classes=self.num_classes)
            if self.use_cuda:
                model = torch.nn.DataParallel(model).cuda()

            if self.model_path is not None: 
                state_dict = torch.load(self.model_path)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                print("Succesfully loaded weights")

            print("SqsueezeNet used")
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

                for layer in self.tuning_layers: 
                    print(f"layers modified : {layer}")
                    model.features[layer].requires_grad_(True)

                for i in [0, 3, 6]: 
                    model.classifier[i].requires_grad_(True)

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
                model = torch.nn.DataParallel(model).cuda()
            return model
    
        elif "vit" in self.model_name.lower():
            print("Using the Vision Transformer architecture.")
            # Load the pre-trained ViT-B/16 model pre-trained on ImageNet-21k
            model = timm.create_model(self.model_type, pretrained=True)
            
            num_features = model.head.in_features
            model.head = nn.Sequential(
                    nn.Linear(num_features, self.num_classes),
                    nn.Dropout(p=self.dropout),
                )

            if self.hidden_size is not None: 
                model.head = nn.Sequential(
                    nn.Linear(num_features, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(self.hidden_size, self.num_classes)
                )

            if self.fine_tune:
                for block in model.blocks[-self.tuning_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
            
            if self.model_path is not None: 
                state_dict = torch.load(self.model_path)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                print("Succesfully loaded weights")
            
            if self.use_cuda:
                model = torch.nn.DataParallel(model).cuda()
            return model
        
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if "vit" in self.model_name.lower():
            data_transforms = {
                "train": transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop with scaling
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                    transforms.RandomAffine(degrees=0, shear=10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]),
                    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
                ]),
                "val": transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]),
                ]),
            }
        elif "efficientNet" in self.model_name or "squeezeNet" in self.model_name or "mobileNet" in self.model_name: 
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                ]),
                'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                ]),
            }
        else:
            data_transforms = {
                "train": transforms.Compose([
                    transforms.Resize((256, 256)),                
                    transforms.RandomCrop((224, 224)),           
                    transforms.RandomHorizontalFlip(),           
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomRotation(degrees=30),       
                    transforms.ToTensor(),                      
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]), 
                ]),
                "val": transforms.Compose([
                    transforms.Resize((224, 224)),               
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
