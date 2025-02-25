import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import transforms, datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


############################################################################################ Set device for training (use GPU if available) ############################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class DAARN(nn.Module):
    def __init__(self, num_classes):
        super(DAARN, self).__init__()
        self.dynamic_branch = self._make_resnet_branch()
        self.steady_branch = self._make_resnet_branch(freeze=True)
        self.num_classes = num_classes
        self.fc = None  # Will initialize after determining feature size

    def _make_resnet_branch(self, freeze=False):
        layers = []
        in_channels = 3
        for out_channels, stride in [(16, 1), (32, 2), (64, 2)]:
            layers.append(ResBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        branch = nn.Sequential(*layers)
        if freeze:
            for param in branch.parameters():
                param.requires_grad = False
        return branch

    def _initialize_fc(self, input_size):
        self.fc = nn.Linear(input_size, self.num_classes)

    def forward(self, x):
        dynamic_out = self.dynamic_branch(x)
        steady_out = self.steady_branch(x)
        # Adaptive aggregation
        aggregated_out = 0.5 * dynamic_out + 0.5 * steady_out
        aggregated_out = F.avg_pool2d(aggregated_out, 4)  # Global average pooling
        aggregated_out = aggregated_out.view(aggregated_out.size(0), -1)  # Flatten
        
        if self.fc is None:
            self._initialize_fc(aggregated_out.size(1))  # Initialize FC layer dynamically

        return self.fc(aggregated_out)


############################################################################################# Define teacher and student models ############################################################################################

# Teacher Model (Larger model for distillation)
class TeacherModel(nn.Module):
    def __init__(self, num_classes):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(260, 64)  # Adjust input size for your dataset
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Student Model (DAARN as you defined earlier, adjusted for numerical data)
class DAARN(nn.Module):
    def __init__(self, num_classes):
        super(DAARN, self).__init__()
        self.fc1 = nn.Linear(260, 64)  # Adjust input size for your dataset
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def distillation_loss(student_output, teacher_output, labels, temperature=3.0, alpha=0.5):
    """
    student_output: The output from the student model.
    teacher_output: The output from the teacher model.
    labels: The true labels (hard targets).
    temperature: Controls the softening of the teacher's output.
    alpha: The weight for the combination of distillation and hard loss.
    """
    # Soft loss (distillation loss)
    soft_loss = F.kl_div(F.log_softmax(student_output / temperature, dim=1),
                         F.softmax(teacher_output / temperature, dim=1),
                         reduction='batchmean') * (temperature * temperature)

    # Hard loss (cross entropy)
    hard_loss = F.cross_entropy(student_output, labels)

    return alpha * soft_loss + (1. - alpha) * hard_loss


############################################################################################# Load training and testing data ############################################################################################
train_path = "./data/train_FD004.txt"
test_path = "./data/test_FD004.txt"

# For the second dataset
# train_path = "./data/train_FD004.txt"
# test_path = "./data/test_FD004.txt"

train_data = pd.read_csv(train_path, sep=" ", header=None)  # Adjust 'sep' and 'header' as needed
test_data = pd.read_csv(test_path, sep=" ", header=None)

# Preview the data
print(train_data.head())


############################################################################################# data pre-processing ############################################################################################

X = np.random.rand(1000, 260)  # Example data: 1000 samples, 128 features
y = np.random.randint(0, 10, 1000)  # Example labels: 10 classes

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Initialize models
num_classes = 10  # Adjust based on your dataset
teacher_model = TeacherModel(num_classes).to(device)
student_model = DAARN(num_classes).to(device)

############################################################################################# Pre-train the teacher  ############################################################################################

# teacher_model.load_state_dict(torch.load('teacher_model.pth'))  # If pre-trained
def pretrain_teacher_model(teacher_model, train_loader, optimizer, criterion, epochs=10):
    teacher_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Get teacher's output
            outputs = teacher_model(inputs)

            # Compute the loss (cross-entropy loss)
            loss = criterion(outputs, labels)

            # Backpropagate and update teacher model
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100
        print(f"Teacher Model Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Save the pretrained teacher model
    torch.save(teacher_model.state_dict(), 'teacher_model.pth')

# Define the optimizer and loss function for the teacher model pretraining
optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Pretrain the teacher model
pretrain_teacher_model(teacher_model, train_loader, optimizer_teacher, criterion, epochs=10)

# Load the pretrained teacher model if needed
teacher_model.load_state_dict(torch.load('teacher_model.pth'))  # If pre-trained

# Optimizer for the student model
optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)

############################################################################################# Training loop with knowledge distillation ############################################################################################
def train_with_distillation(student_model, teacher_model, train_loader, optimizer, epochs=10):
    student_model.train()
    teacher_model.eval()  # Set teacher model to evaluation mode

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Get teacher's output
            teacher_output = teacher_model(inputs)

            # Get student's output
            student_output = student_model(inputs)

            # Compute the distillation loss
            loss = distillation_loss(student_output, teacher_output, labels)

            # Backpropagate and update student model
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(student_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Train the student model using knowledge distillation
train_with_distillation(student_model, teacher_model, train_loader, optimizer_student, epochs=10)


############################################################################################# Evaluation function
def evaluate_model(student_model, test_loader):
    student_model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = student_model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy, all_preds, all_labels

# Evaluate the student model
accuracy, preds, labels = evaluate_model(student_model, test_loader)


############################################################################################ Incremental training function
def incremental_train(student_model, teacher_model, train_loader, optimizer, epochs=1):
    student_model.train()
    teacher_model.eval()  # Teacher model remains frozen during incremental training

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Process each batch in the train_loader
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Get teacher's output
            teacher_output = teacher_model(inputs)

            # Get student's output
            student_output = student_model(inputs)

            # Compute the distillation loss
            loss = distillation_loss(student_output, teacher_output, labels)

            # Backpropagate and update student model
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(student_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")


################################################################################################## Device setup #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10
model = DAARN(num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Initialize teacher model 
teacher_model = TeacherModel(num_classes).to(device) # Initialize the TeacherModel
teacher_model.load_state_dict(torch.load('teacher_model.pth')) # Load the saved weights

# Define the main incremental training loop
for phase in range(3):  # Simulating 3 incremental tasks (phases)
    print(f"Training Phase {phase + 1}")
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train the model incrementally for this phase
    incremental_train(model, teacher_model, train_loader, optimizer, epochs=1)
    
    # Evaluate the model after training this phase
    accuracy, _, _ = evaluate_model(model, test_loader)
    print(f"Phase {phase + 1} Accuracy: {accuracy:.2f}%")
    
    # Update the teacher model to the current model after this phase
    teacher_model = model


