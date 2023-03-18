import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
import argparse

# Some images fail to load...
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True




def test(model, test_loader, criterion):
    model.eval()

    los = 0
    cor = 0

    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        los += loss.item() * inputs.size(0)
        cor += torch.sum(preds == labels.data).item()

    total_loss = los / len(test_loader.dataset)
    total_acc = cor / len(test_loader.dataset)
    print(f"Accuracy: {100 * total_acc}%, Testing Loss: {total_loss}")


def train(model, train_loader, criterion, optimizer):
    model.train()
    trained_images = 0
    num_images = len(train_loader.dataset)
    los = 0
    cor = 0
    for (inputs, labels) in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        los += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        cor += torch.sum(preds == labels.data).item()
        trained_images += len(inputs)
        loss.backward()
        optimizer.step()
        print(f"{trained_images}/{num_images} images trained...")

    total_loss = los / len(train_loader.dataset)
    total_acc = cor / len(train_loader.dataset)
    print(f"Accuracy: {100 * total_acc}%, Testing Loss: {total_loss}")


def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 5))

    return model


def create_data_loader(data, transform_functions, batch_size, shuffle=True):
    # https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2
    data = datasets.ImageFolder(data, transform=transform_functions)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for testing (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=.01, metavar="LR", help="learning rate (default: .01)"
    )
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    args = parser.parse_args()

    model = net()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), normalize])

    train_loader = create_data_loader(args.train, train_transforms, args.batch_size)
    test_loader = create_data_loader(args.test, test_transforms, args.batch_size, shuffle=False)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, loss_criterion, optimizer)
        test(model, test_loader, loss_criterion)

    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':
    main()