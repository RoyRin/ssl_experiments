import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, SVHN 

from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import mixmatch_models
import torch.nn.functional as F
import click
import itertools
import yaml
import os
from loguru import logger

# Set device
#device = torch.device("mps")
device = torch.device("cuda")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

np.random.seed(478)

def write_yaml(filename, d):
    """ dump contents of d into filename, in yaml format"""
    with open(filename, 'w') as file:
        yaml.dump(d, file, default_flow_style=False)


def open_yaml(filename):
    with open(filename, 'r') as stream:
        return yaml.safe_load(stream)

def mixup(x, y, alpha):
    """Perform mixup on the input batch."""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

def sharpen(logits, T):
    """Sharpen the predicted probabilities."""
    probs = F.softmax(logits / T, dim=1)
    return probs



import torch

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}





# Training function with data augmentation
def mixmatch_train2(model, epoch, optimizer, labeled_loader, unlabeled_loader, test_loader, alpha=0.75, K=2, report_every= 40, T = 0.5, lambda_u= 100, ema= None):
    criterion = nn.CrossEntropyLoss()

    model.train()
    for (labeled_data, labels), (unlabeled_data, _) in zip(labeled_loader, unlabeled_loader):#, total=min(len(labeled_loader), len(unlabeled_loader)):
        labeled_data, labels = labeled_data.to(device), labels.to(device)
        unlabeled_data = unlabeled_data.to(device)

        labeled_data_aug = torch.cat([data_augmentation(labeled_data) for _ in range(K)], dim=0)
        labels_aug = labels.repeat(K)

        # Data augmentation for unlabeled data
        unlabeled_data_aug = torch.cat([data_augmentation(unlabeled_data) for _ in range(K)], dim=0)
        #print("unlabeled_data.shape")
        #print(unlabeled_data.shape) torch.Size([32, 3, 32, 32])
        #print(unlabeled_data_aug.shape) torch.Size([64, 3, 32, 32])
        
        # Data augmentation for labeled data
        optimizer.zero_grad()
        # Forward pass for labeled data
        outputs = model(labeled_data_aug)
        loss_labeled = criterion(outputs, labels_aug)
        
        # WORKING Version of Mix Match
        if False: # from the PATE one

        
             # the one that i know works, but doesnt' seem to be good
            labeled_data_aug = torch.cat([data_augmentation(labeled_data) for _ in range(K)], dim=0)
            labels_aug = labels.repeat(K)

            # Data augmentation for unlabeled data
            unlabeled_data_aug = torch.cat([data_augmentation(unlabeled_data) for _ in range(K)], dim=0)

            # Forward pass for unlabeled data
            with torch.no_grad():
                pseudo_labels = torch.softmax(model(unlabeled_data_aug), dim=1)
            outputs_unlabeled = model(unlabeled_data_aug)
            loss_unlabeled = criterion(outputs_unlabeled, pseudo_labels.argmax(dim=1))
        else:
            # Compute the unlabeled loss
            with torch.no_grad():
                # Generate K augmented versions of the unlabeled data
                logits_unlabeled = model(unlabeled_data_aug) # CHECK  this- are we sharpening across the same data, augmented, or across other data points

                probs_unlabeled = sharpen(logits_unlabeled, T)  
                #print(f"probs_unlabeled - {probs_unlabeled.shape}")
                # Mixup the unlabeled data with itself
                x_unlabeled_mixed, p_unlabeled_mixed = mixup(unlabeled_data_aug, probs_unlabeled, alpha) # alpha =beta 


            logits_unlabeled_mixed = model(x_unlabeled_mixed)
            loss_unlabeled = torch.mean((F.softmax(logits_unlabeled_mixed, dim=1) - p_unlabeled_mixed) ** 2)

        # Total loss
        # loss = loss_labeled + alpha * loss_unlabeled
        
        loss = loss_labeled + (lambda_u * loss_unlabeled) # changed this from alpha -> lambda_U

        # Backward and optimize
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update()

    test_acc = None
    if epoch % report_every == 1:
        if ema is not None:
            ema.apply_shadow()
            # Evaluate your model
            ema.restore()

        print(f"loss_labeled { loss_labeled} - loss_unlabeled {loss_unlabeled} ")
        logger.info(f"loss_labeled { loss_labeled} - loss_unlabeled {loss_unlabeled} ")
        test_acc = round(test(model, test_loader),2)
        print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}. test_acc = {test_acc}%")
        logger.info(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}. test_acc = {test_acc}%")
    #raise Exception("donezos")
    return test_acc


from torchvision import transforms

# Data augmentation function
def data_augmentation(x):
    augment = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])
    return augment(x)


# Training function
def baseline_train(model, epoch, optimizer, labeled_loader, unlabeled_loader, test_loader, alpha=0.75, report_every = 40):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for (labeled_data, labels), (unlabeled_data, _) in zip(labeled_loader, unlabeled_loader):
        labeled_data, labels = labeled_data.to(device), labels.to(device)
        unlabeled_data = unlabeled_data.to(device)

        optimizer.zero_grad()

        # Forward pass for labeled data
        outputs = model(labeled_data)
        loss = criterion(outputs, labels)
        # Backward and optimize
        loss.backward()
        optimizer.step()
    test_acc = None
    if epoch % report_every == 1:
        test_acc = round(test(model, test_loader),2)
        print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}. test_acc = {test_acc}%")
        logger.info(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}. test_acc = {test_acc}%")
    return test_acc

# Test accuracy function
def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        #print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


def noisy_labels(dataset, label_accuracy, num_classes= 10):
    noisy_dataset = []
    for data, label in dataset:
        # With probability 1 - label_accuracy, choose a wrong label
        if torch.rand(1).item() > label_accuracy:
            # List of possible labels excluding the correct one
            wrong_labels = list(range(num_classes))
            wrong_labels.remove(label)
            # Randomly choose from wrong labels
            label = np.random.choice(wrong_labels)
        noisy_dataset.append((data, label))
    return noisy_dataset

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def mixmatch_experiment(alpha, T, lambda_u, weight_decay, ema_decay, lr, num_epochs =4000, report_every = 25, ds_name = "SVHN", num_labels = 250, label_accuracy= None):    
    
    save_path_name = f"res__ds_{ds_name}__lambdaU_{lambda_u}__alpha_{alpha}__T_{T}__lr_{lr}__wd_{weight_decay}__labels_{num_labels}__label_accuracy_{label_accuracy}.yaml"
    
    print(f"will save to {save_path_name}")
    logger.info(f"will save to {save_path_name}")
    if os.path.exists(save_path_name):
        print("Already computed")
        logger.info("already computed")
        return 

    def lambda_u_func(step_count, lambda_u= 75):
        # 
        lambda_u_ = lambda_u * min(1, step_count/ 16_000)
        return lambda_u_

    print(f"lambda_u {lambda_u}, ema_decay {ema_decay}, weight_decay {weight_decay}, lr {lr}, T {T}, alpha {alpha}")
    logger.info(f"lambda_u {lambda_u}, ema_decay {ema_decay}, weight_decay {weight_decay}, lr {lr}, T {T}, alpha {alpha}")
    
    
    if ds_name == "MNIST":
        dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        testset = MNIST(root='./data', train=False, download=True, transform=transform)
    elif ds_name == "SVHN":
        dataset = SVHN(root='./data', split='train', download=True, transform=transform)
        testset = SVHN(root='./data', split='test', download=True, transform=transform)
    else:
        raise Exception("Don't know the dataset!")

    input_channels = 1 if ds_name == "MNIST" else 3 
   
    # change labels in dataset
    # 
    if label_accuracy is not None:
        print(f"randomizing the datasets!")
        num_labels_to_randomize = int(0.7 * len(dataset))

        # Generate random indices for selecting the labels to randomize
        indices_to_randomize = np.random.choice(len(dataset), num_labels_to_randomize, replace=False)
        
        # Randomize labels
        # Note: SVHN labels are integers from 0 to 9
        for idx in indices_to_randomize:
            # Choose a new label at random that is different from the current one
            current_label = int(dataset.labels[idx])
            possible_labels = list(range(10))  # SVHN has classes from 0 to 9
            possible_labels.remove(current_label)  # Exclude the original label
            new_label = np.random.choice(possible_labels)
            
            # Assign the new random label to the dataset
            dataset.labels[idx] = new_label
            #####



    labeled_dataset, unlabeled_dataset = random_split(dataset, [num_labels, len(dataset) - num_labels])
    print(type(labeled_dataset))
    if False: # label_accuracy is not None:
        print(f"Noising the dataset")
        noisy_labeled_dataset = noisy_labels(dataset, label_accuracy)
        labeled_dataset = CustomDataset(noisy_labeled_dataset)
    print(type(labeled_dataset))
    # for labeled_dataset, corrupt the labels with prob (1- label_accuracy)
    
    # Data loaders
    batch_size = 64
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    test_batch_size = 256
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    # check which  dataset it is
    #print(f"dataset = {ds_name}")
    #print(f"input_channels = {input_channels}")
    # check the labeleddataset
    print(f"labeled_dataset =  - shape = {labeled_dataset[0][0].shape}")
    logger.info(f"labeled_dataset = - shape = {labeled_dataset[0][0].shape}")
    



    ####
    model = mixmatch_models.WideResNet(10, depth=28, widen_factor=2, dropRate=0.0, input_channels= input_channels).to(device)

    ema = EMA(model, decay=ema_decay)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

    print(f"mixmatch_train2")
    logger.info(f"mixmatch_train2")
    test_accs = []
    import datetime
    start = datetime.datetime.now()
    report_every = 25
    step_count = 0 

    for epoch in range(num_epochs):
        print(f"epoch {epoch}")
        #lambda_u_ = lambda_u_func(step_count=step_count, lambda_u=lambda_u) 
        lambda_u_ = lambda_u
        test_acc= mixmatch_train2(model, epoch, optimizer, report_every=report_every, labeled_loader= labeled_loader, unlabeled_loader= unlabeled_loader, test_loader= test_loader, alpha=alpha, T=T, lambda_u= lambda_u_, ema= ema)
        if test_acc is not None:
            test_accs.append(test_acc)
        if epoch % 250 == 1:
            now = datetime.datetime.now()
            print(f"time since start : {now - start}")
            print(f"step_count - {step_count}")
            print(test_accs)
            print(f"max - {max(test_accs)}")
            logger.info(f"time since start : {now - start}")
            logger.info(f"step_count - {step_count}")
            logger.info(test_accs)
            logger.info(f"max - {max(test_accs)}")
        step_count += (batch_size * len(labeled_loader))

    print(f"mix match test_accs = {test_accs} - max = {max(test_accs)}")  

    print("Continuing now with supervised training!")

    logger.info(f"mix match test_accs = {test_accs} - max = {max(test_accs)}")  
    logger.info("Continuing now with supervised training!")

    # OOPS observation - if I train "regular" after doing mix_match-  I get better accuracy. it goes up to a
    # baseline train:
    
    #test_accs = []
    
    for epoch in range(400):
        test_acc =baseline_train(model, epoch, optimizer, report_every=report_every, labeled_loader= labeled_loader, unlabeled_loader= unlabeled_loader, test_loader= test_loader, alpha=alpha)
        if test_acc is not None:
            test_accs.append(test_acc)
    print(f"SSL + Supervised test_accs = {test_accs} - max = {max(test_accs)}")  
    print(f"lambda_u {lambda_u}, weight_decay {weight_decay}, lr {lr}, T {T}, alpha {alpha} ema_decay {ema_decay}")
    print(f"BEEP - max acc: {max(test_accs)}")
    print(f"saving to {save_path_name}")

    logger.info(f"SSL + Supervised test_accs = {test_accs} - max = {max(test_accs)}")  
    logger.info(f"lambda_u {lambda_u}, weight_decay {weight_decay}, lr {lr}, T {T}, alpha {alpha} ema_decay {ema_decay}")
    logger.info(f"BEEP - max acc: {max(test_accs)}")
    logger.info(f"saving to {save_path_name}")
    write_yaml(save_path_name, test_accs)


    # baseline train:
    if False:
        model = mixmatch_models.WideResNet(10, depth=28, widen_factor=2, dropRate=0.0, input_channels= input_channels).to(device)
        
        optimizer = Adam(model.parameters(), lr=lr)
        test_accs = []
        for epoch in range(num_epochs):
            test_acc =baseline_train(model, epoch, optimizer, report_every=report_every, labeled_loader= labeled_loader, unlabeled_loader= unlabeled_loader, test_loader= test_loader, alpha=alpha)
            if test_acc is not None:
                test_accs.append(test_acc)
        print(f"baseline test_accs = {test_accs} - max = {max(test_accs)}")  



@click.command()
@click.option('--index',
              '-i',
              required=True,)
def slrm_index_run(index):

    index= int(index)
    print(f"index - {index}")
    #λU = 250
    alpha = 0.25
    T = 0.5 # where is this used?
    #λU = 100    

    alpha = 0.25 # 0.75
    T = 0.5
    # In all experiments, we linearly ramp up λU to its maximum value over the first 16,000 steps of training as is common practice
    lambda_u = 100# 50 # 75 # 50 # 250 # 100
    ema_decay = 0.999
    weight_decay = 0.0004 # 0.004
    
    #weight_decay =  0.0004

    #weight_decay = 0.04
    lr = 0.002 # 0.0001 # 0.002
    # mix match paper does --beta=0.75 --w_match=75
    # weight decay = 0.04
    # ema =0.999
    # w_match', 75, 'Weight for distribution matching loss.')
    # filters', 32, 'Filter size of convolutions.')
    # repeat', 4, 'Number of residual layers per stage.')

    lambda_us = [20, 50,75,90, 100, 150, 200, 250]  # 8
    
    lambda_us = [ 50, 75, 100, 150, 250]

    #alphas = [0.1, 0.25, 0.5, .75, 0.9] # 0.5, #5 
    alphas = [ 0.5, 0.75]
    #Ts = [ 0.25, 0.5, .75]
    Ts = [0.5]

    ema_decays = [0.999]
    weight_decays = [0.0004 , 0.004]
    #weight_decays = [0.0004]
    #lrs = [0.002 , 0.0001]

    lrs = [0.002]
    # check the data augmentation is the same?
    
    ### According to Github
    #lambda_us = [175]
    # alphas = [0.75]
    #Ts = [0.5]
    #ema_decays = [0.999]
    #weight_decays= [0.0004]
    #lrs = [0.0002]

    # 3 3 3 1 2 1 = 54 

    # 9 * 5 * 3 * 2 *2 = (45 * 12) = 540 

    # 8 * 3 * 2 = 48
    tups = list(itertools.product(lambda_us, alphas, Ts, ema_decays, weight_decays, lrs))
    tup = tups[index]
    lambda_u, alpha, T, ema_decay, weight_decay, lr = tup
    
    #
    num_labels = 700
    
    num_labels = 500 # 150
    
    label_accuracy = None # 0.7 # simulate the privacy setting.

    num_epochs =4000
    ds_name = "SVHN"
    # TODO - SAVE results !

    # from the previous tests
    if False:
        T = 0.5
        wd= 0.0004
        lr = 0.002
        alpha = 0.5
        lambda_u = 75
        num_labels = 250

    mixmatch_experiment(alpha=alpha, T=T, lambda_u=lambda_u, weight_decay=weight_decay, ema_decay=ema_decay, lr=lr, num_epochs =num_epochs, report_every = 25, ds_name = ds_name, num_labels = num_labels,label_accuracy=label_accuracy)


if __name__ == "__main__":
    slrm_index_run()

if False:
    def random_rotation(x, max_degrees=10):
        batch_size, _, _, _ = x.size()
        angles = np.random.uniform(-max_degrees, max_degrees, size=batch_size)
        thetas = []
        for angle in angles:
            theta = torch.tensor([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
            ], dtype=torch.float)
            thetas.append(theta)
        theta = torch.stack(thetas)
        grid = nn.functional.affine_grid(theta, x.size())
        rotated_x = nn.functional.grid_sample(x, grid)
        return rotated_x


    def data_augmentation(x):
        print(f"data augment")
        return random_rotation(x)




##################
# Training function
def mixmatch_train(model, epoch, optimizer, labeled_loader, unlabeled_loader, test_loader, alpha=0.75, report_every = 40):
    model.train()
    for (labeled_data, labels), (unlabeled_data, _) in zip(labeled_loader, unlabeled_loader):
        labeled_data, labels = labeled_data.to(device), labels.to(device)
        unlabeled_data = unlabeled_data.to(device)

        optimizer.zero_grad()

        # Forward pass for labeled data
        outputs = model(labeled_data)
        loss_labeled = criterion(outputs, labels)

        # Forward pass for unlabeled data
        with torch.no_grad():
            pseudo_labels = torch.softmax(model(unlabeled_data), dim=1)
        outputs_unlabeled = model(unlabeled_data)
        loss_unlabeled = criterion(outputs_unlabeled, pseudo_labels.argmax(dim=1))

        ####

        # Total loss
        loss = loss_labeled + alpha * loss_unlabeled

        # Backward and optimize
        loss.backward()
        optimizer.step()
    test_acc = None
    if epoch % report_every == 1:
        test_acc = round(test(model, test_loader),2)
        print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}. test_acc = {test_acc}%")
    return test_acc

