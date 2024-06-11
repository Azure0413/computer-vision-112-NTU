import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,10)
        pass

    def forward(self, x):

        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out
        pass
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.resnet.maxpool = Identity()
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),  # Add a fully connected layer
            nn.ReLU(),                                   # Add ReLU activation
            nn.Dropout(0.5),                             # Add dropout layer for regularization
            nn.Linear(512, 10)                            # Output layer with 10 units for classification
        )
        # # Add Batch Normalization layers after convolutional layers
        # self.resnet.layer1 = nn.Sequential(self.resnet.layer1, nn.BatchNorm2d(64))
        # self.resnet.layer2 = nn.Sequential(self.resnet.layer2, nn.BatchNorm2d(128))
        # self.resnet.layer3 = nn.Sequential(self.resnet.layer3, nn.BatchNorm2d(256))
        # self.resnet.layer4 = nn.Sequential(self.resnet.layer4, nn.BatchNorm2d(512))
        # # Define data augmentation transformations
        # self.data_transforms = transforms.Compose([
        #     transforms.RandomResizedCrop(224),         # Random resized crop
        #     transforms.RandomHorizontalFlip(),         # Random horizontal flip
        #     transforms.ToTensor(),                      # Convert PIL Image to tensor
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
        # ])

        #######################################################################
        # TODO (optinal):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################

    def forward(self, x):
        return self.resnet(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
