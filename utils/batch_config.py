"""
Configuration for batch processing of PDFs and images with maximum quality
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

BATCH_SIZES = {
    "pdf_page_chunk": 50, 
    "omr_file_chunk": 100, 
    "result_chunk": 500,    
}

RESOURCE_LIMITS = {
    "max_pdf_workers": min(DEFAULT_WORKERS + 4, 32),  
    "max_page_workers": min(DEFAULT_WORKERS + 2, 24),  
    "max_omr_workers": min(DEFAULT_WORKERS + 4, 32),  
}

PERFORMANCE_TUNING = {
    "small_batch": {
        "dpi": 300,               
        "quality": 100,           
        "processing_width": 1200, 
        "save_image_level": 1,    
        "show_image_level": 0,
    },
    "medium_batch": {
        "dpi": 250,               
        "quality": 95,            
        "processing_width": 1100, 
        "save_image_level": 1,
        "show_image_level": 0,
    },
    "large_batch": {
        "dpi": 200,               
        "quality": 95,            
        "processing_width": 1000, 
        "save_image_level": 1,
        "show_image_level": 0,
    },
}

ULTRA_FAST = {
    "dpi": 150,                  
    "quality": 80,               
    "processing_width": 800,     
    "save_image_level": 0,
    "show_image_level": 0,
}

ULTRA_QUALITY = {
    "dpi": 400,                     
    "quality": 100,                  
    "processing_width": 1500,       
    "save_image_level": 1,          
    "show_image_level": 0,
}

HIGH_QUALITY = {
    "dpi": 300,                     
    "quality": 100,                  
    "processing_width": 1200,       
    "save_image_level": 1,
    "show_image_level": 0,
}

def get_batch_profile(batch_size, high_speed=False, high_quality=True):
    """
    Get the appropriate batch profile based on the number of files
    
    Args:
        batch_size: Number of files in the batch
        high_speed: Whether to prioritize speed over quality
        high_quality: Whether to prioritize quality over speed (default True)
        
    Returns:
        Dictionary of performance parameters
    """
    if high_quality and not high_speed:
        return ULTRA_QUALITY
    
    if high_speed and not high_quality:
        return ULTRA_FAST
    
    if batch_size <= 5:
        return {
            "dpi": 400,               
            "quality": 100,           
            "processing_width": 1500, 
            "save_image_level": 1,    
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
        "progressive": True,     
        "quality": 100           
    },
    "png": {
        "optimize": True,        
        "compression_level": 9   
    }
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