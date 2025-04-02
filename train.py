import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import GenderClassifier
from src.data_loader import get_data_loaders
from src.utils import ensure_dir, save_model, plot_training_history, evaluate_model, plot_confusion_matrix

def train(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Create model
    model = GenderClassifier(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)
    
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=args.lr / 100
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for inputs, labels in pbar:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            save_model(model, os.path.join(args.model_dir, 'best_model.pth'))
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    save_model(model, os.path.join(args.model_dir, 'final_model.pth'))
    
    # Plot training history
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(args.output_dir, 'training_history.png')
    )
    
    # Evaluate final model
    print("\nEvaluating final model on validation set...")
    results = evaluate_model(model, val_loader, device)
    print(f"Final validation accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gender classifier")
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to dataset directory')
    parser.add_argument('--img_size', type=int, default=96,
                        help='Image size for model input')
    
    # Model parameters
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Patch size for image embedding')
    parser.add_argument('--embed_dim', type=int, default=192,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=6,
                        help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    # Output parameters
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Create output directories
    ensure_dir(args.model_dir)
    ensure_dir(args.output_dir)
    
    # Train model
    train(args)
