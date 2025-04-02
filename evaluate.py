import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import GenderClassifier
from src.data_loader import get_data_loaders
from src.utils import load_model, evaluate_model, plot_confusion_matrix

def evaluate(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loader (only need test loader)
    _, test_loader = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Create model
    model = GenderClassifier(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)
    
    # Load model weights
    model = load_model(model, args.model_path, device)
    print(f"Model loaded from {args.model_path}")
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print(f"\nTest accuracy: {results['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot confusion matrix
    if args.output_dir:
        plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=os.path.join(args.output_dir, 'test_confusion_matrix.png')
        )
        
        # Plot some examples
        if args.plot_examples:
            plot_examples(model, test_loader, results, device, args.output_dir)
    
    return results

def plot_examples(model, test_loader, results, device, output_dir, num_examples=16):
    """Plot some example predictions"""
    model.eval()
    
    # Get a batch of data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Limit to num_examples
    images = images[:num_examples]
    labels = labels[:num_examples]
    
    # Move to device
    images = images.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Move back to CPU
    images = images.cpu()
    preds = preds.cpu()
    
    # Plot examples
    fig = plt.figure(figsize=(15, 10))
    for i in range(min(num_examples, len(images))):
        ax = fig.add_subplot(4, 4, i + 1)
        
        # Convert tensor to numpy and transpose to HxWxC
        img = images[i].numpy().transpose(1, 2, 0)
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Display image
        ax.imshow(img)
        
        # Set title (green if correct, red if wrong)
        title_color = 'green' if preds[i] == labels[i] else 'red'
        gender = 'Male' if preds[i] == 0 else 'Female'
        true_gender = 'Male' if labels[i] == 0 else 'Female'
        
        title = f"Pred: {gender}\nTrue: {true_gender}"
        ax.set_title(title, color=title_color)
        
        # Remove axes
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_predictions.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate gender classifier")
    
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
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--plot_examples', action='store_true',
                        help='Plot example predictions')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Evaluate model
    evaluate(args)
