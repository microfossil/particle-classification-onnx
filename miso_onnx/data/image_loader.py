import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image


def load_image(
    filename: str | Path,
    img_size: Optional[tuple[int, int]] = None,  # (H, W)
    img_type: str = "rgb",
    out_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Pillow loader.
    - rgb -> HWC uint8
    - greyscale -> HW1 uint8
    Then optionally resize+pad, then convert to float [0,1].
    Returns: HWC float32 in [0,1].
    """
    filename = Path(filename)

    with Image.open(filename) as im:
        if img_type == "rgb":
            im = im.convert("RGB")
            arr = np.asarray(im)  # uint8 HWC
        elif img_type in ("k", "greyscale"):
            im = im.convert("L")
            arr = np.asarray(im)[..., None]  # uint8 HW1
        else:
            raise ValueError("img_type must be 'rgb' or 'k'/'greyscale'")

    if img_size is not None:
        arr = resize_and_pad_image(arr, img_size)

    # float [0,1] only (as requested)
    if arr.dtype != np.float32:
        arr = arr.astype(out_dtype) * (1.0 / 255.0)
    else:
        # If someone passed float already, assume it's 0-255 and scale to 0-1.
        # If that's not true in your codebase, remove this branch.
        arr = arr.astype(out_dtype, copy=False)
        if arr.max() > 1.5:
            arr *= (1.0 / 255.0)

    return arr


def resize_and_pad_image(im: np.ndarray, img_size: tuple[int, int]) -> np.ndarray:
    """
    Matches your original behavior:
    - pad to desired aspect ratio using per-channel median(edge pixels)
    - resize to img_size
    Input: HWC (uint8 preferred)
    Output: HWC uint8 (keeps uint8 so resize is fast), caller converts to float [0,1].
    """
    if im.ndim == 2:
        im = im[..., None]
    if im.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {im.shape}")

    target_h, target_w = img_size
    h, w, c = im.shape

    desired_ratio = target_w / target_h
    current_ratio = w / h

    need_pad = int(round(h * desired_ratio)) != w

    if need_pad:
        # Vectorized median of edge pixels per channel
        top = im[0, :, :]
        bottom = im[-1, :, :]
        left = im[:, 0, :]
        right = im[:, -1, :]
        edges = np.concatenate([top, bottom, left, right], axis=0)
        pad_val = np.median(edges, axis=0)  # (C,)

        # Target padded canvas size
        if desired_ratio > current_ratio:
            new_h = h
            new_w = int(round(h * desired_ratio))
        else:
            new_w = w
            new_h = int(round(w / desired_ratio))

        pad_left = max(0, (new_w - w) // 2)
        pad_right = max(0, new_w - w - pad_left)
        pad_top = max(0, (new_h - h) // 2)
        pad_bottom = max(0, new_h - h - pad_top)

        canvas = np.empty((h + pad_top + pad_bottom, w + pad_left + pad_right, c), dtype=im.dtype)
        canvas[...] = pad_val.reshape(1, 1, c).astype(im.dtype, copy=False)
        canvas[pad_top:pad_top + h, pad_left:pad_left + w, :] = im
        im = canvas

        h, w, _ = im.shape

    # Resize using Pillow for speed
    if (h, w) != (target_h, target_w):
        if im.dtype != np.uint8:
            im = np.clip(im, 0, 255).astype(np.uint8)

        if c == 1:
            pil = Image.fromarray(im[..., 0], mode="L")
        else:
            pil = Image.fromarray(im, mode="RGB")

        pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
        out = np.asarray(pil)
        if c == 1:
            out = out[..., None]
        return out

    return im


def load_single_image_for_inference(
    image_path: Union[str, Path],
    img_size: Tuple[int, int],
    num_channels: int
) -> Tuple[Optional[np.ndarray], Union[str, Path], Optional[str]]:
    """
    Load and preprocess a single image for inference.
    
    Args:
        image_path: Path to the image file
        img_size: Target image size as (height, width)
        num_channels: Number of channels (1 for grayscale, 3 for RGB)
    
    Returns:
        Tuple of (preprocessed_image, image_path, error_message)
        If error occurs, preprocessed_image will be None
    """
    try:
        img_type = "rgb" if num_channels == 3 else "greyscale"
        img = load_image(
            filename=image_path,
            img_size=img_size,
            img_type=img_type,
            out_dtype=np.float32
        )
        return img, image_path, None
    except Exception as e:
        return None, image_path, str(e)


def create_batch_array(images: List[np.ndarray], input_format: str) -> np.ndarray:
    """
    Convert a list of images into a single batch array.
    
    Args:
        images: List of preprocessed images (each HWC format)
        input_format: Expected format - either "NCHW" or "NHWC"
        
    Returns:
        Batched array with shape (batch_size, ...) in specified format
    """
    # Stack into batch (N, H, W, C)
    batch = np.stack(images, axis=0)
    
    # Convert to model's expected format
    if input_format == "NCHW":
        # Convert from NHWC to NCHW
        batch = np.transpose(batch, (0, 3, 1, 2))
    
    return batch


def load_and_batch_images_streaming(
    image_paths: List[Union[str, Path]],
    img_size: Tuple[int, int],
    num_channels: int,
    batch_size: int,
    num_workers: int,
    input_format: str,
    show_progress: bool = True
) -> Generator[Tuple[Optional[np.ndarray], List[Union[str, Path]], List[Tuple[Union[str, Path], str]]], None, None]:
    """
    Generator that loads images in parallel and yields batches on-the-fly.
    This is memory-efficient as it doesn't load all images at once.
    
    Args:
        image_paths: List of paths to images
        img_size: Target image size as (height, width)
        num_channels: Number of channels (1 for grayscale, 3 for RGB)
        batch_size: Number of images per batch
        num_workers: Number of parallel workers for loading
        input_format: Expected format - either "NCHW" or "NHWC"
        show_progress: Whether to show progress bar
        
    Yields:
        Tuple of (batch_array, batch_paths, errors) where:
        - batch_array: numpy array of shape (batch_size, ...) in model's expected format
        - batch_paths: list of paths corresponding to images in this batch
        - errors: list of (path, error_message) tuples accumulated so far
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
    
    errors = []
    batch_buffer = []
    batch_paths_buffer = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(load_single_image_for_inference, path, img_size, num_channels): path 
            for path in image_paths
        }
        
        # Process completed tasks
        iterator = as_completed(future_to_path)
        if show_progress and has_tqdm:
            iterator = tqdm(iterator, total=len(image_paths), desc="Loading images")
        
        for future in iterator:
            img, path, error = future.result()
            
            if error is None:
                batch_buffer.append(img)
                batch_paths_buffer.append(path)
                
                # Yield batch when buffer is full
                if len(batch_buffer) == batch_size:
                    batch_array = create_batch_array(batch_buffer, input_format)
                    yield batch_array, batch_paths_buffer.copy(), errors.copy()
                    batch_buffer.clear()
                    batch_paths_buffer.clear()
            else:
                errors.append((path, error))
        
        # Yield remaining partial batch if any
        if len(batch_buffer) > 0:
            batch_array = create_batch_array(batch_buffer, input_format)
            yield batch_array, batch_paths_buffer.copy(), errors.copy()
            batch_buffer.clear()
            batch_paths_buffer.clear()
