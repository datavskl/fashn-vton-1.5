# image_preprocessing.py
"""
Image preprocessing utilities for FASHN VTON API
Implements best practices for optimal performance and quality
"""

from PIL import Image
import io
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Preprocesses images according to FASHN API best practices:
    1. Resize to max resolution while maintaining aspect ratio
    2. Convert to JPEG with quality 95
    3. Use quality-preserving downsampling (LANCZOS)
    """
    
    # Resolution limits for different endpoint types
    MAX_RESOLUTION_1K = 2000  # 1K endpoints (Try-On v1.5/v1.6)
    MAX_RESOLUTION_4K = 6000  # 4K endpoints (Product to Model)
    
    # Optimal JPEG quality
    JPEG_QUALITY = 95
    
    # Resampling filter for quality preservation
    RESAMPLE_FILTER = Image.Resampling.LANCZOS  # Highest quality downsampling
    
    def __init__(self, max_resolution: int = MAX_RESOLUTION_1K):
        """
        Initialize preprocessor.
        
        Args:
            max_resolution: Maximum pixels on longest edge (2000 for 1K, 6000 for 4K)
        """
        self.max_resolution = max_resolution
    
    def preprocess(
        self, 
        image: Image.Image,
        convert_to_jpeg: bool = True,
        force_rgb: bool = True
    ) -> Tuple[Image.Image, dict]:
        """
        Preprocess image according to FASHN best practices.
        
        Args:
            image: PIL Image object
            convert_to_jpeg: Whether to convert to JPEG format
            force_rgb: Whether to force RGB mode
            
        Returns:
            Tuple of (processed_image, metadata_dict)
        """
        metadata = {
            'original_size': image.size,
            'original_format': image.format,
            'original_mode': image.mode,
        }
        
        processed_image = image.copy()
        
        # Step 1: Convert to RGB if needed
        if force_rgb and processed_image.mode != 'RGB':
            if processed_image.mode == 'RGBA':
                # Handle transparency
                background = Image.new('RGB', processed_image.size, (255, 255, 255))
                background.paste(processed_image, mask=processed_image.split()[3])
                processed_image = background
            else:
                processed_image = processed_image.convert('RGB')
            metadata['converted_to_rgb'] = True
        
        # Step 2: Resize if necessary (maintaining aspect ratio)
        resized, resize_info = self._resize_image(processed_image)
        processed_image = resized
        metadata.update(resize_info)
        
        # Step 3: Convert to JPEG (if requested)
        if convert_to_jpeg:
            processed_image = self._convert_to_jpeg(processed_image)
            metadata['converted_to_jpeg'] = True
            metadata['jpeg_quality'] = self.JPEG_QUALITY
        
        metadata['final_size'] = processed_image.size
        metadata['final_format'] = 'JPEG' if convert_to_jpeg else processed_image.format
        
        return processed_image, metadata
    
    def _resize_image(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """
        Resize image to fit max resolution while maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (resized_image, resize_metadata)
        """
        width, height = image.size
        longest_edge = max(width, height)
        
        metadata = {
            'resized': False,
            'resize_ratio': 1.0,
        }
        
        if longest_edge > self.max_resolution:
            # Calculate new dimensions
            ratio = self.max_resolution / longest_edge
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize with LANCZOS filter (highest quality)
            resized = image.resize(
                (new_width, new_height),
                self.RESAMPLE_FILTER
            )
            
            metadata.update({
                'resized': True,
                'resize_ratio': ratio,
                'new_dimensions': (new_width, new_height),
                'resample_method': 'LANCZOS',
            })
            
            return resized, metadata
        
        return image, metadata
    
    def _convert_to_jpeg(self, image: Image.Image) -> Image.Image:
        """
        Convert image to JPEG format with optimal quality.
        
        Args:
            image: PIL Image object
            
        Returns:
            JPEG-encoded PIL Image
        """
        # Save to bytes buffer as JPEG
        buffer = io.BytesIO()
        image.save(
            buffer,
            format='JPEG',
            quality=self.JPEG_QUALITY,
            optimize=True,
            progressive=True  # Progressive JPEG for better streaming
        )
        buffer.seek(0)
        
        # Load back as PIL Image
        jpeg_image = Image.open(buffer)
        return jpeg_image
    
    def preprocess_to_bytes(
        self,
        image: Image.Image,
        convert_to_jpeg: bool = True
    ) -> Tuple[bytes, dict]:
        """
        Preprocess image and return as bytes.
        
        Args:
            image: PIL Image object
            convert_to_jpeg: Whether to convert to JPEG
            
        Returns:
            Tuple of (image_bytes, metadata)
        """
        processed_image, metadata = self.preprocess(image, convert_to_jpeg)
        
        # Convert to bytes
        buffer = io.BytesIO()
        if convert_to_jpeg:
            processed_image.save(
                buffer,
                format='JPEG',
                quality=self.JPEG_QUALITY,
                optimize=True,
                progressive=True
            )
        else:
            processed_image.save(buffer, format=processed_image.format or 'PNG')
        
        image_bytes = buffer.getvalue()
        metadata['file_size_bytes'] = len(image_bytes)
        metadata['file_size_mb'] = len(image_bytes) / (1024 * 1024)
        
        return image_bytes, metadata
    
    @staticmethod
    def calculate_optimal_size(width: int, height: int, max_resolution: int) -> Tuple[int, int]:
        """
        Calculate optimal dimensions for given constraints.
        
        Args:
            width: Original width
            height: Original height
            max_resolution: Maximum pixels on longest edge
            
        Returns:
            Tuple of (optimal_width, optimal_height)
        """
        longest_edge = max(width, height)
        
        if longest_edge <= max_resolution:
            return width, height
        
        ratio = max_resolution / longest_edge
        return int(width * ratio), int(height * ratio)
    
    def get_preprocessing_stats(self, image: Image.Image) -> dict:
        """
        Get preprocessing statistics without actually processing.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with preprocessing statistics
        """
        width, height = image.size
        longest_edge = max(width, height)
        
        will_resize = longest_edge > self.max_resolution
        
        if will_resize:
            new_width, new_height = self.calculate_optimal_size(
                width, height, self.max_resolution
            )
            resize_ratio = self.max_resolution / longest_edge
        else:
            new_width, new_height = width, height
            resize_ratio = 1.0
        
        return {
            'current_size': (width, height),
            'current_megapixels': (width * height) / 1_000_000,
            'will_resize': will_resize,
            'target_size': (new_width, new_height),
            'target_megapixels': (new_width * new_height) / 1_000_000,
            'resize_ratio': resize_ratio,
            'max_resolution': self.max_resolution,
        }


# Convenience functions
def preprocess_for_tryon(image: Image.Image) -> Tuple[Image.Image, dict]:
    """
    Preprocess image for Try-On v1.5/v1.6 (1K endpoint).
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (processed_image, metadata)
    """
    preprocessor = ImagePreprocessor(max_resolution=2000)
    return preprocessor.preprocess(image)


def preprocess_for_4k(image: Image.Image) -> Tuple[Image.Image, dict]:
    """
    Preprocess image for 4K endpoints.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (processed_image, metadata)
    """
    preprocessor = ImagePreprocessor(max_resolution=6000)
    return preprocessor.preprocess(image)


def optimize_image_size(image: Image.Image, max_size_mb: float = 5.0) -> Image.Image:
    """
    Optimize image to fit within a target file size.
    
    Args:
        image: PIL Image object
        max_size_mb: Maximum file size in megabytes
        
    Returns:
        Optimized PIL Image
    """
    quality = 95
    
    while quality >= 60:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        
        if size_mb <= max_size_mb:
            buffer.seek(0)
            return Image.open(buffer)
        
        quality -= 5
    
    # If still too large, resize
    preprocessor = ImagePreprocessor(max_resolution=1500)
    processed, _ = preprocessor.preprocess(image)
    return processed


# Example usage
if __name__ == "__main__":
    # Example: Preprocess an image for Try-On
    test_image = Image.open("test.jpg")
    
    preprocessor = ImagePreprocessor(max_resolution=2000)
    
    # Get stats before processing
    stats = preprocessor.get_preprocessing_stats(test_image)
    print("Preprocessing stats:")
    print(f"  Current: {stats['current_size']} ({stats['current_megapixels']:.2f} MP)")
    print(f"  Will resize: {stats['will_resize']}")
    print(f"  Target: {stats['target_size']} ({stats['target_megapixels']:.2f} MP)")
    
    # Preprocess
    processed_image, metadata = preprocessor.preprocess(test_image)
    print(f"\nProcessing complete:")
    print(f"  Resized: {metadata['resized']}")
    print(f"  Final size: {metadata['final_size']}")
    
    # Get as bytes
    image_bytes, byte_metadata = preprocessor.preprocess_to_bytes(test_image)
    print(f"  File size: {byte_metadata['file_size_mb']:.2f} MB")
