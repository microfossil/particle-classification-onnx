"""
Result dataclasses for image classification inference.
"""

import csv
import json
from pathlib import Path
from typing import List, Tuple, Union, Optional, Any
import time
from pydantic import BaseModel, Field, field_serializer, ConfigDict
from pydantic.types import FilePath


class ImageClassificationResult(BaseModel):
    """
    Result for a single image classification prediction.
    
    Attributes:
        path: Path to the input image
        label: Predicted class label
        confidence: Confidence score (probability) for the prediction
        class_index: Index of the predicted class in the label list
    """
    path: Union[str, Path]
    label: str
    confidence: float
    class_index: int
    
    @field_serializer('path')
    def serialize_path(self, path: Union[str, Path], _info) -> str:
        """Serialize Path objects to strings for JSON."""
        return str(path)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{Path(self.path).name}: {self.label} ({self.confidence:.3f})"


class ImageClassificationResults(BaseModel):
    """
    Collection of classification results for a batch of images.
    
    Attributes:
        model_name: Name of the model used
        model_path: Path to the model file
        predictions: List of successful predictions
        errors: List of (path, error_message) tuples for failed images
        total_images: Total number of images processed
        batch_size: Batch size used for inference
        device: Device used for inference
        inference_time: Total time taken for inference (seconds)
        img_size: Input image size as (height, width)
        num_channels: Number of input channels
    """
    model_name: str
    model_path: Union[str, Path]
    predictions: List[ImageClassificationResult]
    errors: List[Tuple[Union[str, Path], str]] = Field(default_factory=list)
    total_images: int = 0
    batch_size: int = 32
    device: str = "cpu"
    inference_time: Optional[float] = None
    img_size: Optional[Tuple[int, int]] = None
    num_channels: Optional[int] = None
    
    @field_serializer('model_path')
    def serialize_model_path(self, model_path: Union[str, Path], _info) -> str:
        """Serialize Path objects to strings for JSON."""
        return str(model_path)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def successful_count(self) -> int:
        """Number of successfully processed images."""
        return len(self.predictions)
    
    @property
    def failed_count(self) -> int:
        """Number of failed images."""
        return len(self.errors)
    
    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        if self.total_images == 0:
            return 0.0
        return self.successful_count / self.total_images
    
    def get_predictions_by_label(self, label: str) -> List[ImageClassificationResult]:
        """Get all predictions for a specific label."""
        return [pred for pred in self.predictions if pred.label == label]
    
    def get_top_confident(self, n: int = 10) -> List[ImageClassificationResult]:
        """Get the top N most confident predictions."""
        return sorted(self.predictions, key=lambda x: x.confidence, reverse=True)[:n]
    
    def get_low_confident(self, n: int = 10) -> List[ImageClassificationResult]:
        """Get the N least confident predictions."""
        return sorted(self.predictions, key=lambda x: x.confidence)[:n]
    
    def print_predictions(self):
        """Print predictions in comma-separated format: path, label, confidence."""
        print("Image Path, Class, Score")
        for pred in self.predictions:
            print(f"{pred.path}, {pred.label}, {pred.confidence:.6f}")
            
    def finalize(self):
        self.predictions.sort(key=lambda x: x.path)
    
    def save_json(self, output_path: Union[str, Path], indent: int = 2):
        """
        Save the entire results object as JSON.
        
        Args:
            output_path: Path where the JSON file will be saved
            indent: Number of spaces for JSON indentation (default: 2)
        """
        output_path = Path(output_path)        
        
        # Use Pydantic's model_dump to convert to dict, then save as JSON        
        data = self.model_dump_json(indent=4)
        output_path.write_text(data)        
        print(f"Results saved to JSON: {output_path}")
    
    def save_csv(self, output_path: Union[str, Path]):
        """
        Save predictions as CSV file.
        
        The CSV will contain columns: path, label, confidence, class_index
        
        Args:
            output_path: Path where the CSV file will be saved
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['path', 'label', 'confidence', 'class_index'])
            
            # Write predictions
            for pred in self.predictions:
                writer.writerow([
                    str(pred.path),
                    pred.label,
                    f"{pred.confidence:.6f}",
                    pred.class_index
                ])
        
        print(f"Predictions saved to CSV: {output_path}")
    
    def print_summary(self):
        """Print a summary of the results."""
        print("=" * 70)
        print(f"Classification Results - {self.model_name}")
        print("=" * 70)
        print(f"Model: {Path(self.model_path).name}")
        print(f"Device: {self.device}")
        if self.img_size:
            print(f"Input size: {self.img_size[0]}x{self.img_size[1]}x{self.num_channels}")
        print()
        print(f"Total images: {self.total_images}")
        print(f"Successful: {self.successful_count} ({self.success_rate:.1%})")
        print(f"Failed: {self.failed_count}")
        
        if self.inference_time is not None:
            print(f"\nInference time: {self.inference_time:.2f}s")
            if self.successful_count > 0:
                avg_time = self.inference_time / self.successful_count
                print(f"Average per image: {avg_time*1000:.1f}ms")
        
        if self.predictions:
            print(f"\nConfidence statistics:")
            confidences = [p.confidence for p in self.predictions]
            print(f"  Mean: {sum(confidences)/len(confidences):.3f}")
            print(f"  Min: {min(confidences):.3f}")
            print(f"  Max: {max(confidences):.3f}")
        
        print("=" * 70)
    
    def print_errors(self, max_errors: int = 10):
        """Print error messages for failed images."""
        if not self.errors:
            print("No errors occurred.")
            return
        
        print(f"\nErrors ({len(self.errors)} total):")
        for i, (path, error) in enumerate(self.errors[:max_errors]):
            print(f"  {i+1}. {Path(path).name}: {error}")
        
        if len(self.errors) > max_errors:
            print(f"  ... and {len(self.errors) - max_errors} more errors")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"ImageClassificationResults("
            f"model={self.model_name}, "
            f"successful={self.successful_count}/{self.total_images}, "
            f"failed={self.failed_count})"
        )
