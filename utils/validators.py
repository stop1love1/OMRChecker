"""
Validation utility functions for OMRChecker API
"""
import re
import os
import math
import numpy as np

def validate_directory_name(directory_name):
    """Validates that directory name doesn't contain path separators or invalid characters"""
    if '/' in directory_name or '\\' in directory_name:
        return False, "Directory name cannot contain path separators ('/' or '\\')"
    
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', directory_name):
        return False, "Directory name can only contain alphanumeric characters, underscores, hyphens, and periods"
    
    return True, None
    
def clean_nan_values(obj):
    """Replace NaN, Infinity with empty strings or string representations"""
    if isinstance(obj, dict):
        # Ensure studentId and code are always strings
        if 'studentId' in obj and obj['studentId'] is not None:
            obj['studentId'] = str(obj['studentId'])
        if 'code' in obj and obj['code'] is not None:
            obj['code'] = str(obj['code'])
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return "" 
        if math.isinf(obj):
            return str(obj)
    return obj

def force_string_conversion(data, string_fields):
    """
    Ensure all specified fields are converted to strings with leading zeros preserved.
    This handles cases where pandas might have already converted strings to numbers.
    """
    for field in string_fields:
        if field in data.columns:
            # Convert to string and ensure leading zeros are preserved
            data[field] = data[field].fillna('')
            data[field] = data[field].apply(lambda x: str(x) if x != '' else '')
    return data 