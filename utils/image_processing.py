"""
Image processing utilities for OMRChecker API
"""
import os
import concurrent.futures
import time
import threading
import gc
from pathlib import Path
from src.logger import logger
import cv2
import re
import shutil
import pandas as pd
import json
import glob
import uuid
from PIL import Image
from tqdm import tqdm
import multiprocessing
from src.entry import process_dir
from src.defaults import CONFIG_DEFAULTS

PDF_CACHE = {}
MAX_CACHE_SIZE = 50

# Tối ưu tham số cho xử lý PDF giữ nguyên chất lượng cao
PDF_EXTRACTION_SETTINGS = {
    "dpi": 300,             # Giữ DPI cao để duy trì chất lượng
    "quality": 100,         # Chất lượng tối đa
    "thread_count": 8       # Tăng số luồng xử lý PDF lên 8
}

def process_pdf_page(page_info):
    """Process a single PDF page with PyMuPDF"""
    page, matrix, output_path = page_info
    try:
        # Create pixmap with original size and color settings
        pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace="rgb")
        # Save with original quality
        pix.save(output_path)
        # Release memory explicitly
        pix = None
        return output_path
    except Exception as e:
        logger.error(f"Error processing PDF page: {str(e)}")
        return None

def process_pdf2image_page(page_info):
    """Process a single page with pdf2image"""
    image, output_path = page_info
    # Save the image
    image.save(output_path, 'JPEG')
    return output_path

def verify_pdf_page_sequence(image_paths, pdf_path):
    """
    Verify that all pages from the PDF were extracted in sequence
    
    Args:
        image_paths: List of extracted image paths
        pdf_path: Original PDF path
        
    Returns:
        bool: True if sequence is complete, False otherwise
    """
    if not image_paths:
        logger.error(f"No images extracted from PDF: {pdf_path}")
        return False
        
    # Extract page numbers from filenames
    page_numbers = []
    for path in image_paths:
        try:
            # Extract page number from path
            filename = os.path.basename(path)
            match = re.search(r'page_(\d+)', filename)
            if match:
                page_numbers.append(int(match.group(1)))
        except Exception as e:
            logger.warning(f"Could not extract page number from {path}: {str(e)}")
    
    # Sort the page numbers
    page_numbers.sort()
    
    # Verify sequence is complete without gaps
    expected_pages = list(range(1, len(page_numbers) + 1))
    if page_numbers != expected_pages:
        missing_pages = set(expected_pages) - set(page_numbers)
        if missing_pages:
            logger.error(f"Missing pages in PDF {pdf_path}: {missing_pages}")
            return False
        
        # This shouldn't happen, but just in case:
        extra_pages = set(page_numbers) - set(expected_pages)
        if extra_pages:
            logger.warning(f"Extra pages in PDF {pdf_path}: {extra_pages}")
    
    return True

class PDFProcessingPool:
    """Manages batch processing of multiple PDF files with limited resources"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = PDFProcessingPool()
        return cls._instance
    
    def __init__(self):
        # Get optimal resource limits from batch_config if available
        try:
            from utils.batch_config import RESOURCE_LIMITS
            self.pdf_workers = RESOURCE_LIMITS.get("max_pdf_workers", min(8, os.cpu_count() or 1))
            self.page_workers = RESOURCE_LIMITS.get("max_page_workers", 6)
        except ImportError:
            # Default values if config not available
            self.pdf_workers = min(8, os.cpu_count() or 1)
            self.page_workers = 6
            
        # Current active jobs
        self.active_jobs = 0
        self.lock = threading.Lock()
        self.job_finished = threading.Condition(self.lock)
        self.job_results = {}
    
    def process_pdf_batch(self, pdf_paths, input_dir):
        """Process multiple PDF files with resource management"""
        results = {}
        
        # Prioritize files with lowest number of pages first for better throughput
        prioritized_paths = []
        
        # Check cache first for page counts
        for pdf_path in pdf_paths:
            path_str = str(pdf_path)
            if path_str in PDF_CACHE and os.path.exists(PDF_CACHE[path_str]["first_page"]):
                # If cached and first page still exists, get page count from cache
                page_count = PDF_CACHE[path_str]["page_count"]
                prioritized_paths.append((pdf_path, page_count))
            else:
                # Just add without priority info
                prioritized_paths.append((pdf_path, 999))  # Default high number
        
        # Sort by page count (process smaller PDFs first)
        prioritized_paths.sort(key=lambda x: x[1])
        sorted_pdf_paths = [p[0] for p in prioritized_paths]
        
        # Use process pool for CPU-bound tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.pdf_workers) as executor:
            # Preserve original quality
            logger.info(f"Processing {len(sorted_pdf_paths)} PDFs with original settings")
            
            # Submit all PDF processing jobs
            future_to_pdf = {}
            for pdf_path in sorted_pdf_paths:
                future = executor.submit(
                    self._process_single_pdf_monitored, 
                    pdf_path, 
                    input_dir
                )
                future_to_pdf[future] = pdf_path
            
            # Process results as they complete (not in submission order)
            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    image_paths = future.result()
                    if image_paths:
                        results[pdf_path] = image_paths
                        
                        # Cache the result
                        if len(PDF_CACHE) >= MAX_CACHE_SIZE:
                            # Remove oldest item if cache is full
                            oldest = next(iter(PDF_CACHE))
                            del PDF_CACHE[oldest]
                        
                        PDF_CACHE[str(pdf_path)] = {
                            "page_count": len(image_paths),
                            "first_page": image_paths[0] if image_paths else None
                        }
                    else:
                        results[pdf_path] = []
                    
                    # Run garbage collection after each PDF is processed
                    gc.collect()
                except Exception as e:
                    logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
                    results[pdf_path] = []
        
        return results
    
    def _process_single_pdf_monitored(self, pdf_path, input_dir):
        """Process a single PDF with resource monitoring"""
        path_str = str(pdf_path)
        cache_key = f"{path_str}_original"
        
        if cache_key in PDF_CACHE:
            cached_data = PDF_CACHE[cache_key]
            all_exist = True
            if "image_paths" in cached_data:
                for img_path in cached_data["image_paths"]:
                    if not os.path.exists(img_path):
                        all_exist = False
                        break
                
                if all_exist:
                    logger.info(f"Using cached version of PDF {os.path.basename(path_str)}")
                    return cached_data["image_paths"]
        
        with self.lock:
            while self.active_jobs >= self.pdf_workers:
                self.job_finished.wait()
            self.active_jobs += 1
        
        try:
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(process_single_pdf, pdf_path, input_dir)
                    result = future.result(timeout=60)
                    
                    if result:
                        PDF_CACHE[cache_key] = {
                            "image_paths": result,
                            "page_count": len(result),
                            "timestamp": time.time()
                        }
                    
                    return result
            except concurrent.futures.TimeoutError:
                logger.error(f"Timeout processing PDF {pdf_path}")
                return []
        finally:
            with self.lock:
                self.active_jobs -= 1
                self.job_finished.notify()

def process_single_pdf(pdf_path, input_dir, max_workers=8):
    """
    Process a single PDF file and convert to images sequentially
    
    Args:
        pdf_path: Path to the PDF file
        input_dir: Directory to save the converted images
        max_workers: Maximum number of concurrent workers (ignored for sequential processing)
        
    Returns:
        List of paths to converted images
    """
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    if file_size_mb > 50:
        logger.info(f"Large PDF detected ({file_size_mb:.1f}MB)")
    
    image_paths = []
    pdf_processed = False
    pdf_start_time = time.time()
    expected_page_count = 0  # Store the expected page count
    
    # Try FastPDF approach first - much faster
    try:
        import fitz
        
        # Check if file exists and is readable
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file does not exist: {pdf_path}")
            return []
            
        # Configure for faster processing
        multithreaded_flag = True
        use_accelerator = True
        use_fast_mode = True
        
        # Use memory-optimized document opening with repair mode for corrupt PDFs
        try:
            pdf_document = fitz.open(pdf_path, filetype="pdf")
        except Exception as e:
            logger.warning(f"Error opening PDF with default settings, trying repair mode: {str(e)}")
            try:
                pdf_document = fitz.open(pdf_path, filetype="pdf", repair=True)
            except Exception as repair_error:
                logger.error(f"Failed to open PDF even with repair mode: {str(repair_error)}")
                return []
        
        if pdf_document.page_count == 0:
            logger.warning(f"PDF file has no pages: {os.path.basename(pdf_path)}")
            return image_paths
            
        # Fast matrix for quicker processing
        matrix = fitz.Matrix(2, 2)  # 2x scaling for better quality but faster than 3x
        
        # Check if PDF is encrypted
        if pdf_document.is_encrypted:
            try:
                # Try to decrypt with empty password
                pdf_document.authenticate("")
            except:
                logger.warning(f"PDF is encrypted and could not be decrypted: {os.path.basename(pdf_path)}")
                return []
        
        # Get total page count for progress tracking
        total_pages = pdf_document.page_count
        expected_page_count = total_pages  # Store for later use
        logger.info(f"Processing PDF {os.path.basename(pdf_path)} with {total_pages} pages using optimized method")
        
        # Process in larger batches with concurrent processing using ThreadPoolExecutor
        # This is MUCH faster than sequential processing
        batch_size = 50  # Tăng từ 30 lên 50 trang mỗi lô
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:  # Tăng từ 8 lên 12 luồng
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                batch_futures = []
                
                # Submit batch of pages for processing
                for i in range(batch_start, batch_end):
                    future = executor.submit(
                        _process_pdf_page, 
                        pdf_document, 
                        i, 
                        matrix, 
                        os.path.basename(pdf_path),
                        input_dir
                    )
                    batch_futures.append((i, future))
                
                # Process results as they complete
                for i, future in batch_futures:
                    try:
                        img_path = future.result()
                        if img_path:
                            image_paths.append(img_path)
                    except Exception as e:
                        logger.error(f"Error processing page {i+1} in PDF {pdf_path}: {str(e)}")
                
                # Report progress after each batch
                logger.info(f"PDF {os.path.basename(pdf_path)}: Processed {batch_end}/{total_pages} pages ({batch_end/total_pages*100:.1f}%)")
                
                # Force GC after each batch to avoid memory issues
                gc.collect()
                    
        # Verify sequence before closing
        if len(image_paths) != total_pages:
            logger.warning(f"Expected {total_pages} pages but processed {len(image_paths)} pages from {os.path.basename(pdf_path)}")
        
        # Sort paths to ensure correct order
        image_paths.sort()
        
        # Close the document now that we're done with it
        pdf_document.close()
        pdf_processed = True
        
    except ImportError as e:
        logger.warning(f"PyMuPDF not available: {str(e)}. Will try pdf2image...")
    except Exception as e:
        logger.warning(f"Error using PyMuPDF: {str(e)}. Will try pdf2image...")
        # Try to close the document if it exists
        try:
            if 'pdf_document' in locals() and pdf_document is not None:
                pdf_document.close()
        except:
            pass
    
    # Fall back to pdf2image if PyMuPDF failed
    if not pdf_processed:
        try:
            from pdf2image import convert_from_path
            
            try:
                # Use pdf2image with optimized settings
                logger.info(f"Using pdf2image for processing of {os.path.basename(pdf_path)}")
                
                # These settings make pdf2image much faster
                pdf_images = convert_from_path(
                    pdf_path,
                    thread_count=8,  # Use more threads
                    use_pdftocairo=True,  # Faster than poppler
                    fmt="jpeg",
                    grayscale=False,
                    transparent=False,
                    use_cropbox=False,
                    strict=False,
                    paths_only=True,  # Return paths only to save memory
                )
                
                # Update expected page count from pdf2image
                expected_page_count = len(pdf_images)
                
                # Process images with concurrent processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    # Submit all tasks
                    futures = []
                    for i, img_path in enumerate(pdf_images):
                        img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1:03d}.jpg"
                        dest_path = os.path.join(input_dir, img_filename)
                        futures.append((i, executor.submit(shutil.copy2, img_path, dest_path)))
                    
                    # Collect results
                    for i, future in futures:
                        try:
                            future.result()
                            image_paths.append(os.path.join(input_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1:03d}.jpg"))
                        except Exception as e:
                            logger.error(f"Error saving page {i+1} from {pdf_path}: {str(e)}")
                
                pdf_processed = True
                
                # Verify that all pages were processed
                if len(image_paths) != expected_page_count:
                    logger.warning(f"Expected {expected_page_count} pages but processed {len(image_paths)} pages from {os.path.basename(pdf_path)} with pdf2image")
                
            except Exception as e:
                logger.error(f"Error using pdf2image: {str(e)}")
                raise
                
        except ImportError:
            logger.error("pdf2image not available")
            raise ImportError("Neither PyMuPDF nor pdf2image are installed. Please install at least one of these libraries to process PDF files: 'pip install PyMuPDF pdf2image'")
    
    if not pdf_processed:
        raise Exception("Failed to process PDF with any available method")
    
    pdf_processing_time = time.time() - pdf_start_time
    page_count = len(image_paths)
    
    # Final verification uses the stored expected page count
    if pdf_processed and expected_page_count > 0 and page_count != expected_page_count:
        logger.warning(f"Expected {expected_page_count} pages but processed {page_count} pages from {os.path.basename(pdf_path)}")
    
    pages_per_second = page_count / pdf_processing_time if pdf_processing_time > 0 else 0
    logger.info(f"PDF processed in {pdf_processing_time:.2f} seconds: {os.path.basename(pdf_path)}, {page_count} pages ({pages_per_second:.2f} pages/sec)")
    
    return image_paths

def _process_pdf_page(pdf_document, page_number, matrix, pdf_name, output_dir):
    """Process single PDF page with optimized settings (helper function for parallel processing)"""
    try:
        # Get page
        page = pdf_document[page_number]
        
        # Create unique filename with zero-padded page number for proper sorting
        img_filename = f"{os.path.splitext(pdf_name)[0]}_page_{page_number+1:03d}.jpg"
        img_path = os.path.join(output_dir, img_filename)
        
        # Use optimized settings
        pix = page.get_pixmap(
            matrix=matrix, 
            alpha=False, 
            colorspace="rgb",
            annots=False  # Skip annotations for speed
        )
        
        # Save without jpg_quality parameter
        pix.save(img_path, output="jpeg")
        
        # Explicitly release memory
        pix = None
        page = None
        
        return img_path
    except Exception as e:
        logger.error(f"Error processing page {page_number+1}: {str(e)}")
        return None

def fast_pdf_check(pdf_path):
    """Quickly check if a PDF is valid and get page count without full processing"""
    try:
        import fitz
        with fitz.open(pdf_path) as doc:
            return {
                "valid": True,
                "page_count": doc.page_count,
                "is_encrypted": doc.is_encrypted
            }
    except ImportError:
        # PyMuPDF not available, try pdf2image
        try:
            from pdf2image.pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(pdf_path)
            return {
                "valid": True,
                "page_count": info["Pages"],
                "is_encrypted": False  # Not easily determined with pdfinfo
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

def process_pdf(pdf_path, input_dir, max_workers=12, **kwargs):
    """
    Process PDF file and convert to images
    
    Args:
        pdf_path: Path to the PDF file
        input_dir: Directory to save the converted images
        max_workers: Maximum number of concurrent workers for processing
        **kwargs: Additional arguments for backward compatibility (ignored)
        
    Returns:
        List of paths to converted images
    """
    start_time = time.time()
    
    # Fast check PDF before processing
    info = fast_pdf_check(pdf_path)
    if not info.get("valid", False):
        logger.error(f"Invalid PDF file: {pdf_path}")
        return []
    
    # Create a verification file for this PDF
    verification_path = os.path.join(input_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_verification.txt")
    with open(verification_path, 'w') as verification_file:
        verification_file.write(f"PDF Processing: {os.path.basename(pdf_path)}\n")
        verification_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        verification_file.write(f"Expected Pages: {info.get('page_count', 'Unknown')}\n")
        verification_file.write("-" * 50 + "\n\n")
    
    # Process PDF directly without using the pool
    try:
        image_paths = process_single_pdf(pdf_path, input_dir)
        
        # Update verification file
        with open(verification_path, 'a') as verification_file:
            verification_file.write(f"Processing Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            verification_file.write(f"Pages Extracted: {len(image_paths)}\n")
            verification_file.write("\nExtracted Images:\n")
            for path in image_paths:
                verification_file.write(f"  {os.path.basename(path)}\n")
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        
        # Update verification file with error
        try:
            with open(verification_path, 'a') as verification_file:
                verification_file.write(f"Processing Failed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                verification_file.write(f"Error: {str(e)}\n")
        except:
            pass
            
        # Return empty list on error
        return []
    
    end_time = time.time()
    processing_time = end_time - start_time
    page_count = len(image_paths)
    
    # Calculate processing statistics
    pages_per_second = page_count / processing_time if processing_time > 0 and page_count > 0 else 0
    
    # Update verification file with statistics
    with open(verification_path, 'a') as verification_file:
        verification_file.write("\nProcessing Statistics:\n")
        verification_file.write(f"  Total Processing Time: {processing_time:.2f} seconds\n")
        verification_file.write(f"  Pages Per Second: {pages_per_second:.2f}\n")
    
    logger.info(f"PDF processing completed in {processing_time:.2f} seconds: {os.path.basename(pdf_path)}, {page_count} pages ({pages_per_second:.2f} pages/sec)")
    
    return image_paths

def process_pdf_batch(pdf_paths, input_dir, **kwargs):
    """
    Process multiple PDF files in a resource-managed batch with high performance
    
    Args:
        pdf_paths: List of paths to PDF files
        input_dir: Directory where to save the converted images
        **kwargs: Additional arguments for backward compatibility (ignored)
        
    Returns:
        Dictionary mapping PDF paths to their converted image paths
    """
    start_time = time.time()
    
    # Create a summary file to track all PDFs and their pages
    summary_path = os.path.join(input_dir, "pdf_processing_summary.txt")
    with open(summary_path, 'w') as summary_file:
        summary_file.write(f"PDF Processing Summary\n")
        summary_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_file.write(f"Total PDFs: {len(pdf_paths)}\n")
        summary_file.write("-" * 50 + "\n\n")
    
    # Pre-check PDFs to prioritize processing order
    valid_pdfs = []
    for pdf_path in pdf_paths:
        # First check if already in cache
        path_str = str(pdf_path)
        if path_str in PDF_CACHE and os.path.exists(PDF_CACHE[path_str]["first_page"]):
            # If cached and first page still exists, consider it valid
            valid_pdfs.append((pdf_path, PDF_CACHE[path_str].get("page_count", 999)))
            continue
        
        # Not in cache, do a fast check
        info = fast_pdf_check(pdf_path)
        if info.get("valid", False):
            valid_pdfs.append((pdf_path, info.get("page_count", 999)))
            
            # Write to summary file
            with open(summary_path, 'a') as summary_file:
                summary_file.write(f"PDF: {os.path.basename(pdf_path)}\n")
                summary_file.write(f"  Expected Pages: {info.get('page_count', 'Unknown')}\n")
                summary_file.write(f"  Status: Queued for processing\n\n")
    
    # Sort files by page count (process smaller PDFs first for better UX)
    valid_pdfs.sort(key=lambda x: x[1])
    sorted_pdf_paths = [p[0] for p in valid_pdfs]
    
    # Process PDFs with concurrent.futures for better performance
    # This approach processes multiple PDFs in parallel for significant speedup
    results = {}
    max_parallel_pdfs = min(os.cpu_count() + 2, 8)  # Tăng số lượng PDF xử lý song song lên 8
    
    logger.info(f"Processing {len(sorted_pdf_paths)} PDFs in parallel using {max_parallel_pdfs} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_pdfs) as executor:
        # Submit PDF processing tasks
        future_to_pdf = {}
        for pdf_path in sorted_pdf_paths:
            future = executor.submit(
                process_single_pdf,
                pdf_path,
                input_dir
            )
            future_to_pdf[future] = pdf_path
        
        # Process results as they complete
        completed = 0
        total = len(future_to_pdf)
        
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            completed += 1
            
            try:
                image_paths = future.result()
                results[pdf_path] = image_paths
                
                # Update summary file
                with open(summary_path, 'a') as summary_file:
                    summary_file.write(f"PDF: {os.path.basename(pdf_path)}\n")
                    summary_file.write(f"  Pages Processed: {len(image_paths)}\n")
                    summary_file.write(f"  Status: {'Success' if image_paths else 'Failed'}\n")
                    summary_file.write(f"  Images: {[os.path.basename(p) for p in image_paths[:5]]}{'...' if len(image_paths) > 5 else ''}\n\n")
                
                # Add to cache for future use
                if len(PDF_CACHE) >= MAX_CACHE_SIZE:
                    oldest = next(iter(PDF_CACHE))
                    del PDF_CACHE[oldest]
                
                PDF_CACHE[str(pdf_path)] = {
                    "page_count": len(image_paths),
                    "first_page": image_paths[0] if image_paths else None,
                    "timestamp": time.time()
                }
                
                # Report progress
                logger.info(f"Progress: {completed}/{total} PDFs processed ({completed/total*100:.1f}%)")
            
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
                results[pdf_path] = []
                
                # Update summary for failed PDF
                with open(summary_path, 'a') as summary_file:
                    summary_file.write(f"PDF: {os.path.basename(pdf_path)}\n")
                    summary_file.write(f"  Status: Failed with error\n")
                    summary_file.write(f"  Error: {str(e)}\n\n")
            
            # Force memory cleanup after each PDF
            gc.collect()
    
    end_time = time.time()
    processing_time = end_time - start_time
    pdf_count = len(sorted_pdf_paths)
    
    # Calculate total pages processed
    total_pages = sum(len(pages) for pages in results.values())
    
    # Finalize summary
    with open(summary_path, 'a') as summary_file:
        summary_file.write("-" * 50 + "\n")
        summary_file.write(f"Processing Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_file.write(f"Total PDFs Processed: {pdf_count}\n")
        summary_file.write(f"Total Pages Extracted: {total_pages}\n")
        summary_file.write(f"Total Processing Time: {processing_time:.2f} seconds\n")
        
        if pdf_count > 0 and processing_time > 0:
            summary_file.write(f"Average Time Per PDF: {processing_time/pdf_count:.2f} seconds\n")
            if total_pages > 0:
                summary_file.write(f"Performance: {total_pages/processing_time:.2f} pages/second\n")
    
    logger.info(f"Batch PDF processing completed in {processing_time:.2f} seconds for {pdf_count} PDF files ({total_pages} pages)")
    logger.info(f"Performance: {total_pages/processing_time:.2f} pages/second")
    
    return results

def validate_image(image_path):
    """
    Validate an image file to ensure it can be processed correctly
    
    Args:
        image_path: Path to the image file
        
    Returns:
        (is_valid, error_message) tuple
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return False, f"File not found: {image_path}"
        
        # Check file size
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            return False, f"File is empty: {image_path}"
        
        # If it's a PDF file, we should not try to validate it as an image
        if image_path.lower().endswith('.pdf'):
            return True, ""
        
        # Try to read the image
        img = cv2.imread(image_path)
        
        # Check if image was loaded successfully
        if img is None:
            return False, f"Failed to load image: {image_path}"
        
        # Check image dimensions
        height, width = img.shape[:2]
        if width <= 0 or height <= 0:
            return False, f"Invalid image dimensions ({width}x{height}): {image_path}"
        
        # Image is valid
        return True, ""
        
    except Exception as e:
        return False, f"Error validating image {image_path}: {str(e)}"

def safe_resize(img, width, height=None):
    """
    Safely resize an image with error handling
    
    Args:
        img: OpenCV image
        width: Target width
        height: Target height (calculated from aspect ratio if None)
        
    Returns:
        Resized image or None if failed
    """
    try:
        # Check if source image is valid
        if img is None:
            logger.error("Cannot resize None image")
            return None
            
        # Get original dimensions
        h, w = img.shape[:2]
        
        # Check if original dimensions are valid
        if w <= 0 or h <= 0:
            logger.error(f"Invalid source image dimensions: {w}x{h}")
            return None
            
        # Calculate height if not provided
        if height is None:
            # Maintain aspect ratio
            height = int(h * width / w)
            
        # Ensure target dimensions are valid
        if width <= 0 or height <= 0:
            logger.error(f"Invalid target dimensions: {width}x{height}")
            return None
            
        # Perform resize
        return cv2.resize(img, (int(width), int(height)))
        
    except Exception as e:
        logger.error(f"Error during image resize: {str(e)}")
        return None

def safe_imwrite(image_path, image):
    """
    Safely write an image to a file with error handling
    
    Args:
        image_path: Path to save the image
        image: OpenCV image to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if image is None or empty
        if image is None:
            logger.error(f"Cannot save None image to {image_path}")
            return False
            
        # Check if image has valid dimensions
        if image.size == 0 or len(image.shape) < 2:
            logger.error(f"Cannot save empty image to {image_path}")
            return False
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
        
        # Write image
        result = cv2.imwrite(image_path, image)
        
        if not result:
            logger.error(f"OpenCV imwrite returned False for {image_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error saving image to {image_path}: {str(e)}")
        return False

def organize_files_into_subfolders(file_paths, base_input_dir, files_per_folder=50):
    """
    Organize a large number of files into subfolders with a specified number of files per folder
    
    Args:
        file_paths (list): List of file paths to organize
        base_input_dir (str): Base directory where subfolders will be created
        files_per_folder (int): Number of files to place in each subfolder
        
    Returns:
        dict: Dictionary mapping subfolder paths to the files they contain
    """
    try:
        # Create a summary log for batch processing
        summary_file = os.path.join(base_input_dir, "batch_processing_summary.json")
        
        # Initialize summary data
        summary_data = {
            "total_files": len(file_paths),
            "files_per_folder": files_per_folder,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "subfolders": []
        }
        
        subfolder_map = {}
        total_folders = (len(file_paths) + files_per_folder - 1) // files_per_folder
        
        logger.info(f"Organizing {len(file_paths)} files into {total_folders} subfolders with {files_per_folder} files each")
        
        for folder_idx in range(total_folders):
            # Create subfolder
            subfolder_name = f"batch_{folder_idx + 1:03d}"
            subfolder_path = os.path.join(base_input_dir, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            
            # Get files for this subfolder
            start_idx = folder_idx * files_per_folder
            end_idx = min(start_idx + files_per_folder, len(file_paths))
            folder_files = file_paths[start_idx:end_idx]
            
            subfolder_map[subfolder_path] = []
            folder_file_list = []
            
            # Move files to subfolder
            for file_path in folder_files:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(subfolder_path, file_name)
                
                # Copy the file to the subfolder
                shutil.copy2(file_path, dest_path)
                subfolder_map[subfolder_path].append(dest_path)
                folder_file_list.append(file_name)
            
            # Update summary data
            summary_data["subfolders"].append({
                "name": subfolder_name,
                "path": subfolder_path,
                "files_count": len(folder_files),
                "files": folder_file_list
            })
        
        # Write summary file
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Created {len(subfolder_map)} subfolders and saved summary to {summary_file}")
        
        return subfolder_map
        
    except Exception as e:
        logger.error(f"Error organizing files into subfolders: {str(e)}")
        raise e

def process_subfolder_batches(subfolder_map, base_output_dir):
    """
    Process each subfolder batch and merge results
    
    Args:
        subfolder_map (dict): Dictionary mapping subfolder paths to their contained files
        base_output_dir (str): Base output directory where results will be saved
        
    Returns:
        dict: Processing summary information
    """
    try:
        # Create merged results directory
        merged_dir = os.path.join(base_output_dir, "merged_results")
        os.makedirs(merged_dir, exist_ok=True)
        
        start_time = time.time()
        
        # Initialize batch processing summary
        processing_summary = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_subfolders": len(subfolder_map),
            "total_files": sum(len(files) for files in subfolder_map.values()),
            "processed_subfolders": 0,
            "total_csv_files": 0,
            "total_pdf_pages": 0,
            "merged_dir": merged_dir,
            "subfolder_results": []
        }
        
        csv_files = []
        pdf_pages_count = 0
        
        # Process each subfolder in sequence
        for subfolder_idx, (subfolder_path, files) in enumerate(subfolder_map.items(), 1):
            subfolder_name = os.path.basename(subfolder_path)
            logger.info(f"Processing subfolder {subfolder_idx}/{len(subfolder_map)}: {subfolder_name}")
            
            # Get the output directory for this subfolder
            subfolder_output = os.path.join(base_output_dir, subfolder_name)
            os.makedirs(subfolder_output, exist_ok=True)
            
            # Run processing on this subfolder
            try:
                # Configure processing
                root_dir = Path(os.path.dirname(subfolder_path))
                curr_dir = Path(subfolder_path)
                
                # Set up API arguments
                api_args = {
                    'input_paths': [subfolder_path],
                    'output_dir': os.path.dirname(subfolder_output),
                    'autoAlign': True,
                    'setLayout': False,
                    'debug': True,
                }
                
                # Process using process_dir instead of evaluate
                from src.defaults import CONFIG_DEFAULTS
                results = process_dir(
                    root_dir,
                    curr_dir,
                    api_args,
                    tuning_config=CONFIG_DEFAULTS
                )
                
                # Update PDF page count for this batch
                pdf_files = [f for f in files if f.lower().endswith('.pdf')]
                pdf_pages = len(glob.glob(os.path.join(subfolder_path, "*_page_*.jpg")))
                pdf_pages_count += pdf_pages
                
                # Find all CSV files in the output directory
                batch_csv_files = glob.glob(os.path.join(subfolder_output, "**/*.csv"), recursive=True)
                csv_files.extend(batch_csv_files)
                
                processing_summary["subfolder_results"].append({
                    "name": subfolder_name,
                    "input_path": subfolder_path,
                    "output_path": subfolder_output,
                    "files_count": len(files),
                    "pdf_files": len(pdf_files),
                    "pdf_pages": pdf_pages,
                    "csv_files": len(batch_csv_files),
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Error processing subfolder {subfolder_name}: {str(e)}")
                
                processing_summary["subfolder_results"].append({
                    "name": subfolder_name,
                    "input_path": subfolder_path,
                    "output_path": subfolder_output,
                    "files_count": len(files),
                    "status": "error",
                    "error": str(e)
                })
            
            processing_summary["processed_subfolders"] += 1
        
        # Merge all CSV files
        if csv_files:
            merged_csv_path = os.path.join(merged_dir, "merged_results.csv")
            
            try:
                # Read all CSV files
                dfs = []
                
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file, dtype={'studentId': str, 'code': str})
                        
                        # Add source filename as a column
                        df['source_batch'] = os.path.basename(os.path.dirname(csv_file))
                        df['source_file'] = os.path.basename(csv_file)
                        
                        dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error reading CSV file {csv_file}: {str(e)}")
                
                if dfs:
                    # Concatenate all dataframes
                    merged_df = pd.concat(dfs, ignore_index=True)
                    
                    # Convert studentId and code to string 
                    string_cols = ['studentId', 'code']
                    for col in string_cols:
                        if col in merged_df.columns:
                            merged_df[col] = merged_df[col].astype(str)
                    
                    # Save merged CSV
                    merged_df.to_csv(merged_csv_path, index=False)
                    logger.info(f"Merged {len(csv_files)} CSV files into {merged_csv_path}")
                else:
                    logger.warning("No valid CSV files to merge")
            except Exception as e:
                logger.error(f"Error merging CSV files: {str(e)}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Update processing summary
        processing_summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        processing_summary["total_processing_time"] = round(total_time, 2)
        processing_summary["total_csv_files"] = len(csv_files)
        processing_summary["total_pdf_pages"] = pdf_pages_count
        
        # Write processing summary
        summary_path = os.path.join(merged_dir, "processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(processing_summary, f, indent=2)
        
        logger.info(f"Batch processing complete. Summary saved to {summary_path}")
        
        return processing_summary
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise e

def speed_optimized_batch_processor(file_paths, base_input_dir, base_output_dir, files_per_folder=50, max_workers=None):
    """
    Xử lý nhanh các file theo batch với xử lý song song, giữ nguyên chất lượng
    
    Args:
        file_paths: Danh sách đường dẫn file
        base_input_dir: Thư mục đầu vào cơ sở
        base_output_dir: Thư mục đầu ra cơ sở
        files_per_folder: Số file mỗi thư mục con
        max_workers: Số luồng xử lý tối đa
        
    Returns:
        Dict thông tin kết quả xử lý
    """
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 4)  # Giới hạn 4 luồng mặc định
    
    start_time = time.time()
    logger.info(f"Bắt đầu xử lý nhanh với {max_workers} luồng, {files_per_folder} file/thư mục, giữ nguyên chất lượng")
    
    # 1. Tổ chức file vào các thư mục con
    subfolders = organize_files_into_subfolders(
        file_paths, 
        base_input_dir, 
        files_per_folder=files_per_folder
    )
    
    # 2. Xác định tệp template và marker để sao chép vào các thư mục con
    template_path = None
    marker_path = None
    
    # Tìm template.json trong thư mục gốc
    for subfolder, files in subfolders.items():
        for file in files:
            if os.path.basename(file) == 'template.json':
                template_path = file
            elif os.path.basename(file) in ['marker.png', 'marker.jpg', 'marker.jpeg']:
                marker_path = file
    
    # Sao chép template và marker vào các thư mục con
    if template_path:
        for subfolder in subfolders.keys():
            dest_template = os.path.join(subfolder, 'template.json')
            if not os.path.exists(dest_template):
                shutil.copy2(template_path, dest_template)
                
    if marker_path:
        for subfolder in subfolders.keys():
            dest_marker = os.path.join(subfolder, os.path.basename(marker_path))
            if not os.path.exists(dest_marker):
                shutil.copy2(marker_path, dest_marker)
    
    # 3. Xử lý song song các thư mục con
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Chuẩn bị các tác vụ
        future_to_folder = {}
        
        for subfolder_path, files in subfolders.items():
            subfolder_name = os.path.basename(subfolder_path)
            subfolder_output = os.path.join(base_output_dir, subfolder_name)
            os.makedirs(subfolder_output, exist_ok=True)
            
            future = executor.submit(
                process_folder_high_quality, 
                subfolder_path,
                subfolder_output, 
                PDF_EXTRACTION_SETTINGS
            )
            future_to_folder[future] = subfolder_path
        
        # Xử lý kết quả khi hoàn thành
        for future in concurrent.futures.as_completed(future_to_folder):
            subfolder_path = future_to_folder[future]
            subfolder_name = os.path.basename(subfolder_path)
            
            try:
                folder_result = future.result()
                results[subfolder_name] = folder_result
                logger.info(f"Hoàn thành thư mục {subfolder_name}")
            except Exception as e:
                logger.error(f"Lỗi xử lý thư mục {subfolder_name}: {str(e)}")
                results[subfolder_name] = {"status": "error", "error": str(e)}
    
    # 4. Gộp kết quả từ các thư mục con
    merged_dir = os.path.join(base_output_dir, "merged_results")
    os.makedirs(merged_dir, exist_ok=True)
    
    # Tìm tất cả file CSV
    csv_files = []
    for subfolder_name, result in results.items():
        subfolder_output = os.path.join(base_output_dir, subfolder_name)
        subfolder_csv = glob.glob(os.path.join(subfolder_output, "**/*.csv"), recursive=True)
        csv_files.extend(subfolder_csv)
    
    # Gộp CSV
    if csv_files:
        try:
            merged_csv_path = os.path.join(merged_dir, "merged_results.csv")
            dfs = []
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['source_batch'] = os.path.basename(os.path.dirname(csv_file))
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Lỗi đọc file CSV {csv_file}: {str(e)}")
            
            if dfs:
                merged_df = pd.concat(dfs, ignore_index=True)
                merged_df.to_csv(merged_csv_path, index=False)
                
                # Xuất Excel để dễ xem
                excel_path = os.path.join(merged_dir, "merged_results.xlsx")
                merged_df.to_excel(excel_path, index=False)
                
                logger.info(f"Đã gộp {len(dfs)} file CSV thành {merged_csv_path}")
        except Exception as e:
            logger.error(f"Lỗi gộp CSV: {str(e)}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Kết quả cuối cùng
    return {
        "status": "success",
        "total_files": len(file_paths),
        "total_subfolders": len(subfolders),
        "total_csv_files": len(csv_files),
        "processing_time": processing_time,
        "merged_dir": merged_dir
    }

def process_folder_high_quality(folder_path, output_path, settings=None):
    """
    Xử lý một thư mục con với chất lượng cao nhất
    """
    try:
        # Tạo thư mục đầu ra
        os.makedirs(output_path, exist_ok=True)
        
        # Xử lý các PDF trước (nếu có)
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        
        # Xử lý PDF với chất lượng cao
        if pdf_files:
            for pdf_file in pdf_files:
                process_pdf(
                    pdf_file,
                    folder_path,
                    dpi=settings.get("dpi", 300),
                    quality=settings.get("quality", 100)
                )
        
        # Tuỳ chỉnh giữ nguyên chất lượng
        tuning_config = CONFIG_DEFAULTS.copy()
        # Giữ kích thước gốc
        tuning_config.dimensions.processing_width = 1200  # Kích thước lớn hơn để đảm bảo chất lượng
        # Lưu một số hình ảnh trung gian cho debugging
        tuning_config.outputs.save_image_level = 1
        tuning_config.outputs.show_image_level = 0
        
        # Tạo đường dẫn tương đối
        root_dir = Path(os.path.dirname(folder_path))
        curr_dir = Path(folder_path)
        
        # Chuẩn bị tham số API
        api_args = {
            'input_paths': [folder_path],
            'output_dir': os.path.dirname(output_path),
            'autoAlign': True,  # Bật tự động căn chỉnh để cải thiện độ chính xác
            'setLayout': False,
            'debug': True,  # Bật debug để có nhiều thông tin hơn
        }
        
        # Xử lý thư mục bằng process_dir thay vì evaluate
        results = process_dir(
            root_dir,
            curr_dir,
            api_args,
            tuning_config=tuning_config
        )
        
        return {
            "status": "success",
            "file_count": len(glob.glob(os.path.join(folder_path, "*.*"))),
            "pdf_count": len(pdf_files),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Lỗi xử lý thư mục {folder_path}: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        } 