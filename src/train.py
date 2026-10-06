from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.optim as optim
import torch.nn as nn
from dataset import create_dataloader, get_classes



def build_model(num_classes):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freeze the pretrained backbone — only train the new final layer
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_epoch(model, loss_function, optimizer, loader, device):
    running_loss = 0.0
    model.train()
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)




def evaluate(model, loader, device, loss_function):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_function(outputs, labels)

            correct += (outputs.argmax(dim=1) == labels).sum().item()
    
            running_loss += loss.item() * images.size(0)

    accuracy = correct / len(loader.dataset)
    avg_loss = running_loss / len(loader.dataset)

    return avg_loss, accuracy

 


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device:{device}")

    classes = get_classes()
    model = build_model(num_classes=len(classes)).to(device)
    
    train_dataloader = create_dataloader(train=True)
    val_dataloader = create_dataloader(train=False)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = 0.001)

    num_epochs = 10
    best_val_acc = 0.0

    for x in range(num_epochs):
        avg_training_loss = train_epoch(model=model, loss_function=loss_function,
                                        optimizer=optimizer, loader=train_dataloader,
                                        device=device)

        avg_val_loss, val_acc = evaluate(model=model, loss_function=loss_function,
                                loader=val_dataloader, device=device)

        print(f"Epoch {x+1}/{num_epochs}")
          
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")
            print("Model saved")
            torch.save(model.state_dict(), "models/butterfly_classifier.pt")
        else:
            print(f"Best validationa accuracy: {best_val_acc:.4f}")
        
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")







if __name__ == "__main__":
    main()
