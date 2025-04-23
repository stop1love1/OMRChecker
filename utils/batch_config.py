"""
Batch processing configuration options for OMRChecker.
"""
import os
import platform

SYSTEM_INFO = {
    "cpu_count": os.cpu_count() or 4,
    "platform": platform.system(),
    "is_64bit": platform.architecture()[0] == '64bit',
    "memory_gb": 40.0, 
}

if SYSTEM_INFO["cpu_count"] >= 8:
    HIGH_PERFORMANCE = True
    DEFAULT_WORKERS = min(SYSTEM_INFO["cpu_count"], 24)
else:
    HIGH_PERFORMANCE = False
    DEFAULT_WORKERS = max(4, SYSTEM_INFO["cpu_count"])


RESOURCE_LIMITS = {
    "max_process_workers": 4,
    "max_thread_workers": 12,
    "max_pdf_workers": min(DEFAULT_WORKERS + 2, 16),
    "max_page_workers": min(DEFAULT_WORKERS, 8),
    "max_omr_workers": min(DEFAULT_WORKERS + 4, 32),
    "max_memory_mb": 4096,
}

# Batch processing chunk sizes for optimizing memory use
BATCH_SIZES = {
    "max_files_per_batch": 150,
    "max_pdfs_per_batch": 25,
    "pdf_page_chunk": 50,
    "image_chunk": 250,
    "result_chunk": 1000,
    "omr_file_chunk": 250,
}

FILE_PATTERNS = {
    "image_extensions": [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"],
    "pdf_extensions": [".pdf"],
    "valid_extensions": [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pdf"],
    "prefixes_to_ignore": ["."],
    "ignore_patterns": ["*_out.*", "*_marked.*", "*_copy.*", "*_template.*"],
}

PERFORMANCE_TUNING = {
    "save_marked_images": True,
    "save_image_level": 1,
    "maintain_original_size": True,
    "show_image_level": 0,
    "dpi": None,
    "quality": None,
    "processing_width": 1240,
}

ULTRA_FAST = {
    "save_marked_images": False,
    "save_image_level": 0,
    "maintain_original_size": True,
    "show_image_level": 0,
    "dpi": None,
    "quality": None,
    "processing_width": 1240,
}

HIGH_QUALITY = {
    "save_marked_images": True,
    "save_image_level": 2,
    "maintain_original_size": True,
    "show_image_level": 0,
    "dpi": None,
    "quality": None,
    "processing_width": 1240,
}

MEMORY_MANAGEMENT = {
    "max_cache_size": 300,       
    "force_gc_after_files": 20,  
    "max_batch_memory_gb": 30.0, 
    "prefetch_enabled": True,    
    "prefetch_pages": 20,        
    "use_ram_disk": True,        
    "ram_disk_size_gb": 15.0,    
}

def get_batch_profile(batch_size, high_speed=False, high_quality=True):
    """
    Get the appropriate batch profile based on the number of files
    
    Args:
        batch_size: Number of files in the batch
        high_speed: Whether to prioritize speed over quality
        high_quality: Whether to prioritize quality over speed
        
    Returns:
        Dictionary of performance parameters
    """
    if high_quality and not high_speed:
        return HIGH_QUALITY
    
    if high_speed and not high_quality:
        return ULTRA_FAST
    
    if batch_size <= 5:
        return {
            "save_marked_images": True,
            "save_image_level": 1,
            "maintain_original_size": True,
            "show_image_level": 0,
            "dpi": None,
            "quality": None,
            "processing_width": 1240,
        }
    
    return PERFORMANCE_TUNING

IMAGE_SAVE_OPTIONS = {
    "format": "JPEG",              
    "jpeg_quality": 95,            
    "png_compression": 1,          
} 