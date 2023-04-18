import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import argparse
import tarfile

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv2 = torch.nn.Conv2d(10, 20, 5)
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output

def extract_tar_gz(file_path, output_dir):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=output_dir)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tar_gz_file_path', type=str, required=True, help='Path to the tar.gz file')
parser.add_argument('--output_directory', type=str, required=True, help='Output directory to extract the tar.gz file')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file')
args = parser.parse_args()

# Extract the tar.gz file
tar_gz_file_path = args.tar_gz_file_path
output_directory = args.output_directory
extract_tar_gz(tar_gz_file_path, output_directory)

# Load the model
model_path = f"{output_directory}/model.pth"
model = CustomModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Transformations for the MNIST dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Function to run inference on an image
def run_inference(image, model):
    image_tensor = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    input = Variable(image_tensor)

    # Perform inference
    output = model(input)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Example usage
image_path = args.image_path
image = Image.open(image_path)
predicted_class = run_inference(image, model)
print(f"Predicted class: {predicted_class}")
