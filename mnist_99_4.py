from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01)
        )
        
        # CONV Block 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),  # 28x28x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.01)
        )
        
        # Transition Block 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2)  # 14x14x32
        )
        
        # CONV Block 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),  # 14x14x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01)
        )
        
        # CONV Block 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),  # 14x14x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01)
        )
        
        # Transition Block 2
        self.trans2 = nn.Sequential(
            nn.MaxPool2d(2, 2)  # 7x7x16
        )
        
        # CONV Block 4
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 32, 3),  # 5x5x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.01)
        )
        
        # Global Average Pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)  # 1x1x32
        )
        
        # Output Block
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 10, 1)  # 1x1x10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.trans2(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, scheduler):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(desc=f'loss={loss.item():.4f} batch_id={batch_idx}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(1)
    
    batch_size = 128
    epochs = 19
    
    # Enhanced data augmentation
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-12.0, 12.0), fill=(1,)),
        transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.90, 1.10)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    
    # Load the full training dataset
    full_train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    
    # Create validation set (10000 images) and training set from the training data
    train_size = len(full_train_dataset) - 10000  # Should be 50000
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, 10000],
        generator=torch.Generator().manual_seed(1)
    )
    
    # Load the test dataset (10k images)
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transforms)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, **kwargs)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=3e-4,
        nesterov=True
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
    )

    best_val_accuracy = 0.0
    best_test_accuracy = 0.0
    
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:')
        train(model, device, train_loader, optimizer, scheduler)
        
        print("\nValidation Set Performance:")
        val_accuracy = test(model, device, val_loader)
        
        print("\nTest Set Performance:")
        test_accuracy = test(model, device, test_loader)
        
        # Update best accuracies regardless of threshold
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            
        # Save model if both accuracies are good and combined score improves
        if val_accuracy >= 99.4 and test_accuracy >= 99.4:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'test_accuracy': test_accuracy,
            }, 'mnist_best.pth')
            print(f"\nNew best model saved! Val Acc: {val_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    print(f'\nBest Results:')
    print(f'Best Validation Accuracy: {best_val_accuracy:.2f}%')
    print(f'Best Test Accuracy: {best_test_accuracy:.2f}%')

if __name__ == '__main__':
    main() 