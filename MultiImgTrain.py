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
from dataset import MultiImgDataset2,get_train_val_sampler

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path',type=str)
parser.add_argument('--test_path',type=str)
parser.add_argument('--N_images',type=int,default=6)
parser.add_argument('--bs_train',type=int,default=4)
parser.add_argument('--bs_test',type=int,default=8)
parser.add_argument('--model',type=str,default="densenet")
parser.add_argument('--gpu','--list', nargs='+')

args = parser.parse_args()

transformations = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
##CODEFOR KW51
#train_dataset = MultiImgDataset(args.train_path, transform = transformations,N=args.N_images)
#test_dataset = MultiImgDataset(args.test_path, transform = transformations,N=args.N_images)

full_dataset = MultiImgDataset2(args.train_path, transform = transformations,N=args.N_images)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
#train_sampler, test_sampler = get_train_val_sampler(full_dataset,0.5)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.bs_train,shuffle=True,num_workers=args.bs_train)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = args.bs_test, shuffle=False,num_workers=args.bs_test)
num_labels = full_dataset.num_labels #PUT IN THE NUMBER OF LABELS IN YOUR DATA

if args.model == 'densenet':
    model = models.densenet161(pretrained=True)
    classifier_input = model.classifier.in_features * args.N_images
    model.classifier = nn.Identity()
if args.model == 'resnet18':
    model = models.resnet18(pretrained=True)
    classifier_input = model.fc.in_features * args.N_images
    model.fc = nn.Identity()
if args.model == 'resnet152':
    model = models.resnet152(pretrained=True)
    classifier_input = model.fc.in_features * args.N_images
    model.fc = nn.Identity()
if args.model =='convnext':
    model = models.convnext_large(pretrained=True)
    classifier_input = model.classifier[2].in_features * args.N_images
    model.classifier = nn.Identity()
if args.model =='vit':
    model = models.vit_l_32(pretrained=True)
    classifier_input = model.heads[0].in_features * args.N_images
    model.heads = nn.Identity()
if args.model == 'txt':
    model = nn.Sequential (
         nn.Conv1d(1, 32, kernel_size=4,stride=2),
         nn.ReLU(inplace=True),
         nn.Conv1d(32, 64, kernel_size=4,stride=2),
         nn.ReLU(inplace=True),
         nn.Conv1d(64, 128, kernel_size=4,stride=2),
         nn.ReLU(inplace=True),
         nn.Conv1d(128, 256, kernel_size=4,stride=2),
         nn.ReLU(inplace=True))
    classifier_input = 12288 * args.N_images #model.out_features * args.N_images 

classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=classifier_input, out_features=1024),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1024, out_features=1024),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(in_features=1024, out_features=num_labels),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.gpu:
  print("Let's use", str(args.gpu), "GPUs!")
  model = nn.DataParallel(model,device_ids=[int(val) for val in args.gpu])

model.to(device)
classifier.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.parameters())+list(classifier.parameters()),lr=1e-4)
epochs = 100
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs = [input.to(device, dtype=torch.float) for input in inputs]
        labels = labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output_list = []
        for input in inputs:
            output = model.forward(input)
            output_list.append(output)
        output_list = torch.cat(output_list,dim=1)
        output = classifier.forward(output_list)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item() #*inputs[0].size(0)
        
        # Print the progress of our training
        counter += 1
        if counter % 100 == 0:
            print("Loss:" + str(loss.item()))
    # Evaluating the model
    model.eval()
    test_counter = 0
    running_corrects = 0
    total_test_samples = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = [input.to(device, dtype=torch.float) for input in inputs]
            labels = labels.to(device)

            output_list = []
            for input in inputs:
                output = model.forward(input)
                output_list.append(output)
            output_list = torch.cat(output_list,dim=1)
            output = classifier.forward(output_list)
            # Loss
            valloss = criterion(output, labels)


            # Calculate Loss
            valloss = criterion(output, labels)
            # Add loss to the validation set's running loss
            val_loss += valloss.item() #*inputs[0].size(0)
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function

            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_test_samples += inputs[0].shape[0]

    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader)
    valid_loss = val_loss/len(test_loader)
    # Print out the information
    epoch_acc = running_corrects / total_test_samples
    print('Accuracy: ', epoch_acc)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
