import os
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from src.model import GenderClassifier
from src.face_detector import FaceDetector
from src.utils import load_model, predict_gender

def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Create face detector
    face_detector = FaceDetector()
    
    # Process single image
    if args.image_path:
        process_image(args.image_path, model, face_detector, device, args)
    
    # Process images in a directory
    elif args.image_dir:
        process_directory(args.image_dir, model, face_detector, device, args)

def process_image(image_path, model, face_detector, device, args):
    """Process a single image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Create a copy for drawing
    display_image = image.copy()
    
    # Detect faces and draw rectangles
    faces = face_detector.face_cascade.detectMultiScale(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # If no faces detected
    if len(faces) == 0:
        print("No faces detected in the image")
        
        # Try to predict on the whole image
        gender, confidence, _ = predict_gender(
            model, image, face_detector, device, args.img_size
        )
        
        text = f"{gender}: {confidence:.2f}"
        cv2.putText(display_image, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face with margin
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)
            
            face_img = image[y1:y2, x1:x2]
            
            # Predict gender
            gender, confidence, _ = predict_gender(
    model, face_img, face_detector, device, args.img_size
)
            
            # Draw rectangle around face
            color = (0, 255, 0) if gender == 'Male' else (255, 0, 0)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            
            # Add text label
            text = f"{gender}: {confidence:.2f}"
            cv2.putText(display_image, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Display or save the result
    if args.display:
        cv2.imshow('Gender Prediction', display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        output_path = os.path.join(
            args.output_dir, 
            f"pred_{os.path.basename(image_path)}"
        )
        cv2.imwrite(output_path, display_image)
        print(f"Result saved to {output_path}")

def process_directory(image_dir, model, face_detector, device, args):
    """Process all images in a directory"""
    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    # Process each image
    print(f"Processing {len(image_files)} images...")
    for image_path in image_files:
        print(f"Processing {image_path}...")
        process_image(image_path, model, face_detector, device, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict gender from facial images")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str,
                       help='Path to input image')
    group.add_argument('--image_dir', type=str,
                       help='Path to directory containing images')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=96,
                        help='Image size for model input')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Patch size for image embedding')
    parser.add_argument('--embed_dim', type=int, default=192,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=6,
                        help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save output images')
    parser.add_argument('--display', action='store_true',
                        help='Display results')
    
    args = parser.parse_args()
    
    # Run main function
    main(args)
