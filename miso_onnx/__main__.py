"""
CLI for miso-onnx image classification.
"""

import click
from pathlib import Path
from typing import Optional, List
import sys

from data.network_info import NetworkInfo
from inference.image_classification import ImageClassificationPipeline


def parse_labels(labels_input: str) -> List[str]:
    """
    Parse labels from either a comma-separated string or a file path.
    
    Args:
        labels_input: Either comma-separated labels or path to text file
        
    Returns:
        List of label strings
    """
    labels_path = Path(labels_input)
    
    # Check if it's a file path
    if labels_path.exists() and labels_path.is_file():
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        return labels
    
    # Otherwise treat as comma-separated string
    labels = [label.strip() for label in labels_input.split(',') if label.strip()]
    return labels


def collect_image_paths(folder: Path) -> List[Path]:
    """
    Collect all image files from a folder.
    
    Args:
        folder: Path to folder containing images
        
    Returns:
        List of image file paths
    """
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(folder.glob(f'*{ext}'))
        image_paths.extend(folder.glob(f'*{ext.upper()}'))
    
    return sorted(image_paths)


@click.command()
@click.option(
    '--network-info',
    type=click.Path(exists=True, path_type=Path),
    help='Path to network info XML file (contains all model configuration)'
)
@click.option(
    '--model',
    type=click.Path(exists=True, path_type=Path),
    help='Path to ONNX model file (required if not using --network-info)'
)
@click.option(
    '--labels',
    type=str,
    help='Comma-separated labels or path to labels file (required if not using --network-info)'
)
@click.option(
    '--images',
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help='Path to folder containing images to classify'
)
@click.option(
    '--batch-size',
    default=32,
    type=int,
    help='Batch size for inference (default: 32)'
)
@click.option(
    '--num-workers',
    default=4,
    type=int,
    help='Number of parallel workers for image loading (default: 4)'
)
@click.option(
    '--device',
    default='cpu',
    type=click.Choice(['cpu', 'cuda'], case_sensitive=False),
    help='Device to run inference on (default: cpu)'
)
@click.option(
    '--output-json',
    type=click.Path(path_type=Path),
    help='Path to save results as JSON'
)
@click.option(
    '--output-csv',
    type=click.Path(path_type=Path),
    help='Path to save predictions as CSV'
)
@click.option(
    '--show-progress/--no-progress',
    default=True,
    help='Show progress bars during processing (default: show)'
)
def classify(
    network_info: Optional[Path],
    model: Optional[Path],
    labels: Optional[str],
    images: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    output_json: Optional[Path],
    output_csv: Optional[Path],
    show_progress: bool
):
    """
    Classify images using an ONNX model.
    
    You can provide either:
    
    1. A network info XML file (--network-info) that contains all configuration, OR
    
    2. Individual parameters (--model and --labels)
    
    Examples:
    
        # Using network info XML:
        python -m miso-onnx classify --network-info model/network_info.xml --images data/images/
    
        # Using individual parameters:
        python -m miso-onnx classify --model model.onnx --labels "cat,dog,bird" --images data/images/
    
        # With output files:
        python -m miso-onnx classify --network-info model/network_info.xml --images data/images/ \\
            --output-json results.json --output-csv predictions.csv
    
        # Using CUDA:
        python -m miso-onnx classify --network-info model/network_info.xml --images data/images/ \\
            --device cuda --batch-size 64
    """
    
    # Validation: ensure either network-info OR (model + labels) are provided
    if network_info is not None:
        if model is not None or labels is not None:
            click.echo("Error: Cannot specify both --network-info and individual parameters (--model, --labels)", err=True)
            sys.exit(1)
        mode = "network_info"
    elif model is not None and labels is not None:
        mode = "individual"
    else:
        click.echo("Error: Must provide either --network-info OR both --model and --labels", err=True)
        click.echo("\nUse --help for usage information", err=True)
        sys.exit(1)
    
    # Collect image paths
    try:
        image_paths = collect_image_paths(images)
    except Exception as e:
        click.echo(f"Error: Failed to collect images from {images}: {e}", err=True)
        sys.exit(1)
    
    if not image_paths:
        click.echo(f"Error: No images found in {images}", err=True)
        click.echo("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif", err=True)
        sys.exit(1)
    
    click.echo(f"Found {len(image_paths)} images to classify")
    
    # Create pipeline based on mode
    try:
        if mode == "network_info":
            click.echo(f"\nLoading network configuration from: {network_info}")
            net_info = NetworkInfo(network_info)
            
            if show_progress:
                net_info.print_summary()
            
            # Check if model file exists
            if not net_info.model_path.exists():
                click.echo(f"\nError: Model file not found at {net_info.model_path}", err=True)
                click.echo("Please ensure the model file is in the same directory as the XML file.", err=True)
                sys.exit(1)
            
            click.echo(f"\nCreating inference pipeline...")
            pipeline = ImageClassificationPipeline.from_network_info(
                net_info,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device.lower()
            )
        
        else:  # individual mode
            click.echo(f"\nLoading model: {model}")
            
            # Parse labels
            try:
                class_labels = parse_labels(labels)
            except Exception as e:
                click.echo(f"Error: Failed to parse labels: {e}", err=True)
                sys.exit(1)
            
            if not class_labels:
                click.echo("Error: No labels provided", err=True)
                sys.exit(1)
            
            click.echo(f"Loaded {len(class_labels)} class labels")
            
            click.echo(f"\nCreating inference pipeline...")
            pipeline = ImageClassificationPipeline(
                model_path=model,
                class_labels=class_labels,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device.lower(),
                model_name=model.stem
            )
    
    except Exception as e:
        click.echo(f"Error: Failed to create pipeline: {e}", err=True)
        sys.exit(1)
    
    # Run inference
    click.echo(f"\nRunning inference on {len(image_paths)} images...")
    click.echo(f"Batch size: {batch_size}, Workers: {num_workers}, Device: {device.upper()}")
    click.echo("")
    
    try:
        results = pipeline.predict(
            image_paths,
            return_probabilities=True,
            show_progress=show_progress
        )
    except Exception as e:
        click.echo(f"\nError: Inference failed: {e}", err=True)
        sys.exit(1)
    
    # Print summary
    click.echo("\n")
    results.print_summary()
    
    # Show example predictions
    if results.predictions:
        click.echo("\n" + "=" * 70)
        click.echo("Top 5 most confident predictions:")
        click.echo("=" * 70)
        for i, pred in enumerate(results.get_top_confident(5), 1):
            click.echo(f"  {i}. {pred}")
        
        click.echo("\n" + "=" * 70)
        click.echo("Top 5 least confident predictions:")
        click.echo("=" * 70)
        for i, pred in enumerate(results.get_low_confident(5), 1):
            click.echo(f"  {i}. {pred}")
    
    # Show errors if any
    if results.errors:
        click.echo("\n")
        results.print_errors(max_errors=5)
    
    # Save outputs
    output_files = []
    
    if output_json:
        try:
            results.save_json(output_json)
            output_files.append(f"JSON: {output_json}")
        except Exception as e:
            click.echo(f"\nWarning: Failed to save JSON: {e}", err=True)
    
    if output_csv:
        try:
            results.save_csv(output_csv)
            output_files.append(f"CSV: {output_csv}")
        except Exception as e:
            click.echo(f"\nWarning: Failed to save CSV: {e}", err=True)
    
    if output_files:
        click.echo("\n" + "=" * 70)
        click.echo("Output files created:")
        for output_file in output_files:
            click.echo(f"  {output_file}")
        click.echo("=" * 70)
    
    click.echo("\nClassification complete!")


@click.group()
def cli():
    """miso-onnx: ONNX-based image classification toolkit"""
    pass


cli.add_command(classify)


if __name__ == '__main__':
    cli()
