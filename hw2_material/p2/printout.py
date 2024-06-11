import torch
from torchsummary import summary
from model import MyNet, ResNet18

def print_model_summary(model, input_size):
    print(summary(model, input_size=input_size))

if __name__ == '__main__':
    # Instantiate both models
    mynet_model = MyNet()
    resnet_model = ResNet18()

    # Set input size for each model
    mynet_input_size = (3, 32, 32)  # Example input size for MyNet
    resnet_input_size = (3, 32, 32)  # Example input size for ResNet18

    # Print summary for each model
    print("MyNet Summary:")
    print_model_summary(mynet_model, mynet_input_size)

    print("\nResNet18 Summary:")
    print_model_summary(resnet_model, resnet_input_size)