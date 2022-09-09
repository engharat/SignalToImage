import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

def get_train_val_sampler(mydset, percent=0.9, limit=None):
    ## define our indices 
    num_train = len(mydset)
    indices = list(range(num_train))
    split = int(num_train * percent)

    # Random, non-contiguous split
    train_idx = np.random.choice(indices, size=split, replace=False)
    val_idx = list(set(indices) - set(train_idx))
    
    if limit:
        train_idx = train_idx[:limit]
    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    ## define our samplers -- we use a SubsetRandomSampler because it will return
    ## a random subset of the split defined by the given indices without replace
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    return train_sampler, val_sampler

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path',type=str)

args = parser.parse_args()

transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(args.path, transform = transformations)
train_sampler, test_sampler = get_train_val_sampler(dataset,percent=0.8)
train_loader = torch.utils.data.DataLoader(dataset, batch_size= 24,sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size = 48, sampler=test_sampler)

model = models.densenet161(pretrained=True)
classifier_input = model.classifier.in_features
num_labels = 2 #PUT IN THE NUMBER OF LABELS IN YOUR DATA
model.classifier = nn.Sequential(
    nn.Linear(in_features=classifier_input, out_features=num_labels),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4)
epochs = 100
for epoch in tqdm(range(epochs)):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item()*inputs.size(0)
        
        # Print the progress of our training
        counter += 1

    # Evaluating the model
    model.eval()
    test_counter = 0
    running_corrects = 0
    total_test_samples = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = criterion(output, labels)
            # Add loss to the validation set's running loss
            val_loss += valloss.item()*inputs.size(0)
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function

            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_test_samples += inputs.shape[0]
            # Print the progress of our evaluation
            test_counter += 1

    # Get the average loss for the entire epoch
    train_loss = train_loss/counter
    valid_loss = val_loss/test_counter
    # Print out the information
    epoch_acc = running_corrects / total_test_samples
    print('Accuracy: ', epoch_acc)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
