"""
Image classification pipeline using ONNX Runtime.
Supports parallel image loading and batched inference.
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple
import time
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).parent.parent))
from data.image_loader import load_and_batch_images_streaming
from inference.results import ImageClassificationResult, ImageClassificationResults


class ImageClassificationPipeline:
    """
    High-performance image classification pipeline with parallel preprocessing
    and batched ONNX inference.
    
    Args:
        model_path: Path to the ONNX model file
        img_size: Target image size as (height, width)
        num_channels: Number of input channels (1 for grayscale, 3 for RGB)
        class_labels: List of class label names
        batch_size: Batch size for inference (default: 32)
        num_workers: Number of parallel workers for image loading (default: 4)
        device: Device to run inference on ('cpu' or 'cuda', default: 'cpu')
        model_name: Optional name of the model
    """

    def __init__(
            self,
            model_path: Union[str, Path],
            # img_size: Tuple[int, int],
            # num_channels: int,
            class_labels: List[str],
            batch_size: int = 32,
            num_workers: int = 4,
            device: str = "cpu",
            model_name: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        # self.img_size = img_size  # (H, W)
        # self.num_channels = num_channels
        self.class_labels = class_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.model_name = model_name or self.model_path.stem

        # Validate inputs
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # if num_channels not in [1, 3]:
        #     raise ValueError(f"num_channels must be 1 or 3, got {num_channels}")

        if len(class_labels) == 0:
            raise ValueError("class_labels cannot be empty")

        # Load ONNX model
        self._load_model()

    def _load_model(self):
        """Load the ONNX model and validate input/output shapes."""
        # Set up session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Choose execution providers based on device
        if self.device.lower() == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Create inference session
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )

        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Validate input shape
        input_shape = self.session.get_inputs()[0].shape

        # Determine input format (NCHW or NHWC)
        if len(input_shape) == 4:
            if input_shape[1] in [1, 3]:
                self.input_format = "NCHW"
                self.num_channels = input_shape[1]
                self.img_size = (input_shape[2], input_shape[3])
            elif input_shape[3] in [1, 3]:
                self.input_format = "NHWC"
                self.num_channels = input_shape[3]
                self.img_size = (input_shape[1], input_shape[2])
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")

        # expected_channels = input_shape[1] if "NCHW" else input_shape[3]

        print(f"Model input shape: {input_shape}")
        print(f"Model input format: {self.input_format}")

        # Handle dynamic batch dimension
        # if isinstance(expected_channels, str):
        #     # If channels dimension is dynamic, skip validation
        #     pass
        # elif expected_channels != self.num_channels:
        #     raise ValueError(
        #         f"Model expects {expected_channels} channels, "
        #         f"but num_channels={self.num_channels}"
        #     )

        print(f"Loaded model: {self.model_path}")
        print(f"Input format: {self.input_format}")
        print(f"Input shape: {input_shape}")
        print(f"Device: {self.session.get_providers()[0]}")

    @staticmethod
    def from_network_info(
            network_info,
            batch_size: int = 32,
            num_workers: int = 4,
            device: str = "cpu"
    ) -> "ImageClassificationPipeline":
        """
        Create an ImageClassificationPipeline from a NetworkInfo object.
        
        Args:
            network_info: NetworkInfo instance containing model configuration
            batch_size: Batch size for inference (default: 32)
            num_workers: Number of parallel workers for image loading (default: 4)
            device: Device to run inference on ('cpu' or 'cuda', default: 'cpu')
            
        Returns:
            ImageClassificationPipeline instance configured with the network's settings
        """
        if not network_info.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {network_info.model_path}\n"
                f"Expected at: {network_info.model_path.absolute()}"
            )

        return ImageClassificationPipeline(
            model_path=network_info.model_path,
            # img_size=network_info.img_size,
            # num_channels=network_info.num_channels,
            class_labels=network_info.labels,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            model_name=network_info.name
        )

    def _run_inference(self, batch: np.ndarray) -> np.ndarray:
        """
        Run inference on a single batch.
        
        Args:
            batch: Batch of images in model's expected format
            
        Returns:
            Model outputs (logits or probabilities)
        """
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: batch}
        )
        return outputs[0]

    def _postprocess_outputs(
            self,
            outputs: np.ndarray,
            return_probabilities: bool = True,
            return_indices: bool = False
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Post-process model outputs to get predictions.
        
        Args:
            outputs: Raw model outputs (batch_size, num_classes)
            return_probabilities: Whether to return confidence scores
            return_indices: Whether to return class indices
            
        Returns:
            Tuple of (predicted_labels, confidence_scores, class_indices)
        """
        # Apply softmax if outputs look like logits (values outside [0, 1])
        if outputs.min() < 0 or outputs.max() > 1.5:
            # Compute softmax
            exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
        else:
            probabilities = outputs

        # Get predicted class indices
        predicted_indices = np.argmax(probabilities, axis=1)
        confidence_scores = np.max(probabilities, axis=1)

        # Map to class labels
        predicted_labels = [self.class_labels[idx] for idx in predicted_indices]

        confidence_list = confidence_scores.tolist() if return_probabilities else [1.0] * len(predicted_labels)

        if return_indices:
            return predicted_labels, confidence_list, predicted_indices.tolist()
        else:
            return predicted_labels, confidence_list, []

    def predict(
            self,
            image_paths: List[Union[str, Path]],
            return_probabilities: bool = True,
            show_progress: bool = True,
            raise_on_error: bool = False
    ) -> ImageClassificationResults:
        """
        Run inference on a list of images.
        
        Args:
            image_paths: List of paths to images
            return_probabilities: Whether to include confidence scores in results
            show_progress: Whether to show progress bars
            raise_on_error: Whether to raise exception on image loading errors
            
        Returns:
            ImageClassificationResults object containing predictions and metadata
        """
        if len(image_paths) == 0:
            return ImageClassificationResults(
                model_name=self.model_name,
                model_path=self.model_path,
                predictions=[],
                errors=[],
                total_images=0,
                batch_size=self.batch_size,
                device=self.session.get_providers()[0],
                inference_time=0.0,
                img_size=self.img_size,
                num_channels=self.num_channels
            )

        # Initialize result accumulators
        predictions = []
        errors = []
        total_processed = 0

        # Track inference time
        start_time = time.time()

        # Create progress bar for inference if requested
        inference_pbar = None
        if show_progress:
            inference_pbar = tqdm(total=len(image_paths), desc="Running inference", position=1)

        # Stream batches and run inference
        for batch_array, batch_paths, current_errors in load_and_batch_images_streaming(
                image_paths=image_paths,
                img_size=self.img_size,
                num_channels=self.num_channels,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                input_format=self.input_format,
                show_progress=show_progress
        ):
            # Update errors list
            errors = current_errors

            # Skip if this is the final empty batch (just error reporting)
            if batch_array is None:
                continue

            # Run inference on this batch
            outputs = self.session.run([self.output_name], {self.input_name: batch_array})[0]
            labels, confidences, indices = self._postprocess_outputs(
                outputs, return_probabilities, return_indices=True
            )

            # Accumulate predictions for this batch
            for path, label, confidence, idx in zip(batch_paths, labels, confidences, indices):
                predictions.append(ImageClassificationResult(
                    path=str(path),
                    label=label,
                    confidence=confidence,
                    class_index=idx
                ))

            # Update inference progress
            total_processed += len(batch_paths)
            if inference_pbar:
                inference_pbar.update(len(batch_paths))

        if inference_pbar:
            inference_pbar.close()

        inference_time = time.time() - start_time

        # Check for errors
        if raise_on_error and len(errors) > 0:
            raise RuntimeError(f"Failed to load {len(errors)} images. First error: {errors[0]}")

        if total_processed == 0:
            print("Warning: No images were successfully loaded")

        results = ImageClassificationResults(
            model_name=self.model_name,
            model_path=self.model_path,
            predictions=predictions,
            errors=errors,
            total_images=len(image_paths),
            batch_size=self.batch_size,
            device=self.session.get_providers()[0],
            inference_time=inference_time,
            img_size=self.img_size,
            num_channels=self.num_channels
        )
        results.finalize()
        return results

    def predict_single(
            self,
            image_path: Union[str, Path],
            return_probabilities: bool = True
    ) -> ImageClassificationResult:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to the image
            return_probabilities: Whether to include confidence score
            
        Returns:
            ImageClassificationResult object
        """
        results = self.predict(
            [image_path],
            return_probabilities=return_probabilities,
            show_progress=False,
            raise_on_error=True
        )
        return results.predictions[0]


def main():
    """Example usage of the pipeline with NetworkInfo."""
    import sys
    from pathlib import Path

    # Add parent directory to path to import NetworkInfo
    from data.network_info import NetworkInfo

    # Example 1: Load network info from XML and create pipeline
    xml_path = r"C:\Users\ross.marchant\data\Files_to_Ross\Files_to_Ross\CNNs\ResNet50_20250521-152437\model_onnx\network_info.xml"

    if not Path(xml_path).exists():
        print(f"Example XML not found at: {xml_path}")
        print("\nUsage:")
        print("  python image_classification.py <path_to_network_info.xml> <image_folder>")
        return

    # Parse network info
    print("Loading network configuration...")
    network_info = NetworkInfo(xml_path)
    network_info.print_summary()

    # Check if model exists
    if not network_info.model_path.exists():
        print(f"\nError: Model file not found at {network_info.model_path}")
        print("Please ensure the model file is in the same directory as the XML file.")
        return

    print("\nCreating inference pipeline using from_network_info...")
    # Method 1: Using the static method (recommended)
    pipeline = ImageClassificationPipeline.from_network_info(
        network_info,
        batch_size=32,
        num_workers=4,
        device="cpu"
    )

    # Method 2: Using NetworkInfo.create_pipeline (also works, delegates to above)
    # pipeline = network_info.create_pipeline(batch_size=32, num_workers=4, device="cpu")

    # Example 2: Run inference on images
    # Get image paths from command line or use example
    image_folder = Path(r"C:\Users\ross.marchant\data\Files_to_Ross\Files_to_Ross\Individual images examples")
    image_paths = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    # if len(sys.argv) > 2:
    #     image_folder = Path(sys.argv[2])
    #
    # else:
    #     print("\nNo image folder provided. Use: python image_classification.py <xml_path> <image_folder>")
    #     return

    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    print(f"\nRunning inference on {len(image_paths)} images...")
    results = pipeline.predict(image_paths, show_progress=True)

    print("\n")
    results.print_summary()

    # Example: Print predictions in comma-separated format
    print("\n" + "=" * 70)
    print("Predictions (comma-separated format):")
    print("=" * 70)
    results.print_predictions()

    # Example: Save results as JSON
    json_output = image_folder / "classification_results.json"
    results.save_json(json_output)

    # Example: Save predictions as CSV
    csv_output = image_folder / "predictions.csv"
    results.save_csv(csv_output)

    print("\n" + "=" * 70)
    print("Output files created:")
    print(f"  JSON: {json_output}")
    print(f"  CSV: {csv_output}")
    print("=" * 70)

    # Show some example predictions
    if results.predictions:
        print("\nTop 5 most confident predictions:")
        for i, pred in enumerate(results.get_top_confident(5), 1):
            print(f"  {i}. {pred}")

        print("\nTop 5 least confident predictions:")
        for i, pred in enumerate(results.get_low_confident(5), 1):
            print(f"  {i}. {pred}")

    # Show errors if any
    if results.errors:
        results.print_errors(max_errors=5)

    # Example 3: Single image prediction
    if image_paths:
        print(f"\nSingle image prediction example:")
        single_result = pipeline.predict_single(image_paths[0])
        print(f"  {single_result}")


if __name__ == "__main__":
    main()
