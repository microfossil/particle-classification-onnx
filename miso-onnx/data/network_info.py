"""
Parser for network info XML files.
Extracts model configuration and metadata for inference.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class LabelInfo:
    """Information about a single class label."""
    code: str
    count: int
    precision: float
    recall: float
    f1score: float
    support: int


class NetworkInfo:
    """
    Parses and stores network configuration from XML file.
    
    This class encapsulates all the metadata needed to run inference,
    including model path, input/output specs, labels, and preprocessing info.
    """
    
    def __init__(self, xml_path: str | Path):
        """
        Load and parse network info from XML file.
        
        Args:
            xml_path: Path to the network info XML file
        """
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Network info file not found: {self.xml_path}")
        
        self._parse_xml()
    
    def _parse_xml(self):
        """Parse the XML file and extract all relevant information."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        # Basic metadata
        self.name = self._get_text(root, 'name', 'Unknown')
        self.description = self._get_text(root, 'description', '')
        self.type = self._get_text(root, 'type', 'unknown')
        self.date = self._get_text(root, 'date', '')
        
        # Model file path (relative to XML file location)
        protobuf_filename = self._get_text(root, 'protobuf', 'model.onnx')
        self.model_path = self.xml_path.parent / protobuf_filename
        
        # Input specifications
        input_elem = root.find('.//inputs/input')
        if input_elem is None:
            raise ValueError("No input specification found in XML")
        
        self.input_name = self._get_text(input_elem, 'name', 'image')
        self.input_operation = self._get_text(input_elem, 'operation', '')
        self.input_height = int(self._get_text(input_elem, 'height', '224'))
        self.input_width = int(self._get_text(input_elem, 'width', '224'))
        self.input_channels = int(self._get_text(input_elem, 'channels', '3'))
        
        # Output specifications
        output_elem = root.find('.//outputs/output')
        if output_elem is None:
            raise ValueError("No output specification found in XML")
        
        self.output_name = self._get_text(output_elem, 'name', 'pred')
        self.output_operation = self._get_text(output_elem, 'operation', '')
        self.output_height = int(self._get_text(output_elem, 'height', '0'))
        
        # Parse labels (in order)
        self.labels: List[str] = []
        self.label_info: Dict[str, LabelInfo] = {}
        
        labels_elem = root.find('labels')
        if labels_elem is not None:
            for label_elem in labels_elem.findall('label'):
                code = self._get_text(label_elem, 'code', '')
                if code:
                    self.labels.append(code)
                    
                    # Store detailed label info
                    self.label_info[code] = LabelInfo(
                        code=code,
                        count=int(self._get_text(label_elem, 'count', '0')),
                        precision=float(self._get_text(label_elem, 'precision', '0.0')),
                        recall=float(self._get_text(label_elem, 'recall', '0.0')),
                        f1score=float(self._get_text(label_elem, 'f1score', '0.0')),
                        support=int(self._get_text(label_elem, 'support', '0'))
                    )
        
        # Preprocessing info
        prepro_elem = root.find('prepro')
        if prepro_elem is not None:
            self.prepro_name = self._get_text(prepro_elem, 'name', 'rescale')
            params_elem = prepro_elem.find('params')
            if params_elem is not None:
                self.prepro_params = [
                    float(self._get_text(param, '.', '1.0'))
                    for param in params_elem.findall('param')
                ]
            else:
                self.prepro_params = []
        else:
            self.prepro_name = 'rescale'
            self.prepro_params = [255.0, 0.0, 1.0]
        
        # Training/evaluation metrics
        self.accuracy = float(self._get_text(root, 'accuracy', '0.0'))
        self.precision = float(self._get_text(root, 'precision', '0.0'))
        self.recall = float(self._get_text(root, 'recall', '0.0'))
        self.f1score = float(self._get_text(root, 'f1score', '0.0'))
        
        # Source data info
        self.source_data = self._get_text(root, 'source_data', '')
        self.source_size = int(self._get_text(root, 'source_size', '0'))
        
        # Training load info
        load_elem = root.find('load')
        if load_elem is not None:
            self.training_epochs = int(self._get_text(load_elem, 'training_epochs', '0'))
            self.training_time = float(self._get_text(load_elem, 'training_time', '0.0'))
            self.training_split = float(self._get_text(load_elem, 'training_split', '0.0'))
            self.training_time_per_image = float(self._get_text(load_elem, 'training_time_per_image', '0.0'))
            self.inference_time_per_image = float(self._get_text(load_elem, 'inference_time_per_image', '0.0'))
        else:
            self.training_epochs = 0
            self.training_time = 0.0
            self.training_split = 0.0
            self.training_time_per_image = 0.0
            self.inference_time_per_image = 0.0
    
    @staticmethod
    def _get_text(element: ET.Element, tag: str, default: str = '') -> str:
        """Safely get text content from an XML element."""
        child = element.find(tag)
        if child is not None and child.text is not None:
            return child.text.strip()
        return default
    
    @property
    def img_size(self) -> Tuple[int, int]:
        """Get image size as (height, width) tuple."""
        return (self.input_height, self.input_width)
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.labels)
    
    @property
    def num_channels(self) -> int:
        """Get number of input channels."""
        return self.input_channels
    
    def create_pipeline(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = "cpu"
    ):
        """
        Create an ImageClassificationPipeline using this network configuration.
        
        Args:
            batch_size: Batch size for inference
            num_workers: Number of parallel workers for image loading
            device: Device to run inference on ('cpu' or 'cuda')
            
        Returns:
            ImageClassificationPipeline instance configured with this network's settings
        """
        from ..inference.image_classification import ImageClassificationPipeline
        
        return ImageClassificationPipeline.from_network_info(self, batch_size, num_workers, device)
    
    def __repr__(self) -> str:
        """String representation of the network info."""
        return (
            f"NetworkInfo(name='{self.name}', type='{self.type}', "
            f"classes={self.num_classes}, img_size={self.img_size}, "
            f"accuracy={self.accuracy:.3f})"
        )
    
    def print_summary(self):
        """Print a detailed summary of the network configuration."""
        print("=" * 70)
        print(f"Network: {self.name}")
        print("=" * 70)
        print(f"Type: {self.type}")
        print(f"Date: {self.date}")
        print(f"Model: {self.model_path.name}")
        print()
        print(f"Input: {self.input_height}x{self.input_width}x{self.input_channels}")
        print(f"Classes: {self.num_classes}")
        print()
        print(f"Accuracy: {self.accuracy:.3f}")
        print(f"Precision: {self.precision:.3f}")
        print(f"Recall: {self.recall:.3f}")
        print(f"F1 Score: {self.f1score:.3f}")
        print()
        if self.source_size > 0:
            print(f"Training data: {self.source_size} images")
            print(f"Training epochs: {self.training_epochs}")
            print(f"Training time: {self.training_time:.1f}s")
        print("=" * 70)
