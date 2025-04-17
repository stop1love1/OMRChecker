"""
Configuration for batch processing of PDFs and images
"""
import os
import platform

SYSTEM_INFO = {
    "cpu_count": os.cpu_count() or 4,
    "platform": platform.system(),
    "is_64bit": platform.architecture()[0] == '64bit',
    "memory_gb": 40.0,  # Server has 40GB RAM
}

if SYSTEM_INFO["cpu_count"] >= 8:
    HIGH_PERFORMANCE = True
    DEFAULT_WORKERS = min(SYSTEM_INFO["cpu_count"], 24)  # Increased worker limit for high-RAM server
else:
    HIGH_PERFORMANCE = False
    DEFAULT_WORKERS = max(4, SYSTEM_INFO["cpu_count"])  

# Larger batch sizes for high-memory server
BATCH_SIZES = {
    "pdf_page_chunk": 100,  # Much larger chunks for 40GB RAM
    "omr_file_chunk": 200,  # Process many files in parallel
    "result_chunk": 500,    # Very large result chunks
}

# Maximize workers for high-RAM server
RESOURCE_LIMITS = {
    "max_pdf_workers": min(DEFAULT_WORKERS + 4, 32),  # More PDF workers
    "max_page_workers": min(DEFAULT_WORKERS + 2, 24),  # More page workers
    "max_omr_workers": min(DEFAULT_WORKERS + 4, 32),  # More OMR workers
}

PERFORMANCE_TUNING = {
    "small_batch": {
        "dpi": 72,
        "quality": 60,
        "processing_width": 700,
        "save_image_level": 0,
        "show_image_level": 0,
    },
    "medium_batch": {
        "dpi": 72,
        "quality": 55,
        "processing_width": 650,
        "save_image_level": 0,
        "show_image_level": 0,
    },
    "large_batch": {
        "dpi": 72,
        "quality": 50,
        "processing_width": 600,
        "save_image_level": 0,
        "show_image_level": 0,
    },
}

ULTRA_FAST = {
    "dpi": 72,
    "quality": 45,
    "processing_width": 500,
    "save_image_level": 0,
    "show_image_level": 0,
}

HIGH_QUALITY = {
    "dpi": 150,
    "quality": 95,
    "processing_width": 1000,
    "save_image_level": 1,
    "show_image_level": 0,
}

def get_batch_profile(batch_size, high_speed=True, high_quality=False):
    """
    Get the appropriate batch profile based on the number of files
    
    Args:
        batch_size: Number of files in the batch
        high_speed: Whether to prioritize speed over quality
        high_quality: Whether to prioritize quality over speed
        
    Returns:
        Dictionary of performance parameters
    """
    if high_speed and not high_quality:
        return ULTRA_FAST
    
    if high_quality:
        return HIGH_QUALITY
        
    if batch_size <= 5:
        if high_speed:
            return {
                "dpi": 72,
                "quality": 60,
                "processing_width": 700,
                "save_image_level": 0,
                "show_image_level": 0,
            }
        else:
            return {
                "dpi": 120,
                "quality": 80,
                "processing_width": 800,
                "save_image_level": 0,
                "show_image_level": 0,
            }
    elif batch_size <= 20:
        return PERFORMANCE_TUNING["small_batch"]
    elif batch_size <= 50:
        return PERFORMANCE_TUNING["medium_batch"]
    else:
        return PERFORMANCE_TUNING["large_batch"]

COMPRESSION_OPTIONS = {
    "jpg": {
        "optimize": True,
        "progressive": False
    },
    "png": {
        "optimize": False,
        "compression_level": 1
    }
}

# Optimized for 40GB RAM server
MEMORY_MANAGEMENT = {
    "max_cache_size": 500,  # Much larger cache for 40GB RAM
    "force_gc_after_files": 50,  # Less frequent garbage collection
    "max_batch_memory_gb": 30.0,  # Allow using up to 30GB for processing
    "prefetch_enabled": True,  # Enable prefetching
    "prefetch_pages": 50,  # Prefetch more pages
    "use_ram_disk": True,  # Use RAM disk for temporary files
    "ram_disk_size_gb": 10.0,  # 10GB RAM disk
} 