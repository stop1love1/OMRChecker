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

PDF_CACHE = {}
MAX_CACHE_SIZE = 50

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

def process_single_pdf(pdf_path, input_dir, max_workers=4):
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
    
    # Try PyMuPDF first (faster than pdf2image)
    try:
        import fitz
        
        # Check if file exists and is readable
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file does not exist: {pdf_path}")
            return []
            
        # Use memory-optimized document opening with repair mode for corrupt PDFs
        try:
            pdf_document = fitz.open(pdf_path, filetype="pdf")
        except Exception as e:
            # If opening fails with generic settings, try opening with repair mode
            logger.warning(f"Error opening PDF with default settings, trying repair mode: {str(e)}")
            try:
                pdf_document = fitz.open(pdf_path, filetype="pdf", repair=True)
            except Exception as repair_error:
                logger.error(f"Failed to open PDF even with repair mode: {str(repair_error)}")
                return []
        
        if pdf_document.page_count == 0:
            logger.warning(f"PDF file has no pages: {os.path.basename(pdf_path)}")
            return image_paths
            
        # Use Identity matrix to preserve exact size of original PDF
        matrix = fitz.Identity
        
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
        logger.info(f"Processing PDF {os.path.basename(pdf_path)} with {total_pages} pages sequentially")
        
        # Process pages SEQUENTIALLY to ensure no pages are missed
        for i in range(total_pages):
            try:
                # Get current page
                page = pdf_document[i]
                
                # Create unique filename with zero-padded page number for proper sorting
                img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1:03d}.jpg"
                img_path = os.path.join(input_dir, img_filename)
                
                # Process page directly without concurrent processing
                pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace="rgb")
                pix.save(img_path)
                pix = None  # Release memory explicitly
                
                # Save path and report progress regularly
                image_paths.append(img_path)
                if (i+1) % 10 == 0 or (i+1) == total_pages:
                    logger.info(f"PDF {os.path.basename(pdf_path)}: Processed {i+1}/{total_pages} pages")
                
                # Force garbage collection for large PDFs to avoid memory issues
                if (i+1) % 50 == 0:
                    gc.collect()
                    
            except Exception as page_error:
                logger.error(f"Error processing page {i+1} in PDF {pdf_path}: {str(page_error)}")
                # Continue processing other pages even if one fails
        
        # Verify that all pages were processed before closing the document
        if verify_pdf_page_sequence(image_paths, pdf_path):
            logger.info(f"All {len(image_paths)}/{total_pages} pages successfully extracted from {os.path.basename(pdf_path)}")
        else:
            logger.warning(f"Page sequence verification failed for {os.path.basename(pdf_path)}")
        
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
                # Use pdf2image with sequential processing (disable threading)
                logger.info(f"Using pdf2image for sequential processing of {os.path.basename(pdf_path)}")
                
                # Process without threading to ensure sequence
                pdf_images = convert_from_path(
                    pdf_path,
                    thread_count=1,  # Force single-threaded
                    use_pdftocairo=True,
                    fmt="jpeg",
                    grayscale=False,
                    transparent=False,
                    use_cropbox=False,
                    strict=False,
                )
                
                # Update expected page count from pdf2image
                expected_page_count = len(pdf_images)
                
                # Process images SEQUENTIALLY
                image_paths = []
                for i, image in enumerate(pdf_images):
                    # Create zero-padded page number for proper sorting
                    img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1:03d}.jpg"
                    img_path = os.path.join(input_dir, img_filename)
                    
                    # Save directly without concurrency
                    image.save(img_path, 'JPEG')
                    image_paths.append(img_path)
                    
                    # Report progress
                    if (i+1) % 10 == 0 or (i+1) == len(pdf_images):
                        logger.info(f"PDF {os.path.basename(pdf_path)}: Processed {i+1}/{len(pdf_images)} pages with pdf2image")
                    
                    # Force garbage collection for large PDFs
                    if (i+1) % 50 == 0:
                        gc.collect()
                
                pdf_processed = True
                
                # Verify that all pages were processed
                if verify_pdf_page_sequence(image_paths, pdf_path):
                    logger.info(f"All {len(image_paths)} pages successfully extracted from {os.path.basename(pdf_path)} with pdf2image")
                else:
                    logger.warning(f"Page sequence verification failed for {os.path.basename(pdf_path)} with pdf2image")
                
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
    
    logger.info(f"PDF processed in {pdf_processing_time:.2f} seconds: {os.path.basename(pdf_path)}, {page_count} pages ({page_count/pdf_processing_time:.2f} pages/sec)")
    
    return image_paths

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
    Process multiple PDF files sequentially in a batch
    
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
    
    # Create verification file to track all PDF pages
    verification_path = os.path.join(input_dir, "pdf_page_verification.txt")
    with open(verification_path, 'w') as verification_file:
        verification_file.write(f"PDF Page Verification\n")
        verification_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        verification_file.write("-" * 50 + "\n\n")
    
    # Pre-check PDFs to prioritize processing order
    valid_pdfs = []
    for pdf_path in pdf_paths:
        try:
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
            else:
                # Log invalid PDF
                logger.warning(f"Invalid PDF file detected, skipping: {pdf_path}")
                with open(summary_path, 'a') as summary_file:
                    summary_file.write(f"PDF: {os.path.basename(pdf_path)}\n")
                    summary_file.write(f"  Status: Invalid PDF, skipped\n")
                    summary_file.write(f"  Error: {info.get('error', 'Unknown error')}\n\n")
        except Exception as e:
            # Log any errors during pre-check
            logger.error(f"Error pre-checking PDF {pdf_path}: {str(e)}")
            with open(summary_path, 'a') as summary_file:
                summary_file.write(f"PDF: {os.path.basename(pdf_path)}\n")
                summary_file.write(f"  Status: Pre-check failed\n")
                summary_file.write(f"  Error: {str(e)}\n\n")
    
    # Sort files by page count (process smaller PDFs first for better UX)
    valid_pdfs.sort(key=lambda x: x[1])
    sorted_pdf_paths = [p[0] for p in valid_pdfs]
    
    # Process all PDFs sequentially
    results = {}
    
    # Process PDFs one by one for maximum reliability
    for i, pdf_path in enumerate(sorted_pdf_paths):
        try:
            logger.info(f"Processing PDF {i+1}/{len(sorted_pdf_paths)}: {os.path.basename(pdf_path)}")
            
            # Process the PDF directly (not using the pool)
            image_paths = process_single_pdf(pdf_path, input_dir)
            results[pdf_path] = image_paths
            
            # Write page numbers to verification file
            with open(verification_path, 'a') as verification_file:
                verification_file.write(f"PDF: {os.path.basename(pdf_path)}\n")
                for img_path in image_paths:
                    try:
                        # Extract page number from path
                        filename = os.path.basename(img_path)
                        match = re.search(r'page_(\d+)', filename)
                        if match:
                            page_num = match.group(1)
                            verification_file.write(f"  Page {page_num}: {filename}\n")
                    except Exception as e:
                        verification_file.write(f"  Error extracting page number: {str(e)}\n")
                verification_file.write("\n")
            
            # Update summary
            with open(summary_path, 'a') as summary_file:
                summary_file.write(f"PDF: {os.path.basename(pdf_path)}\n")
                summary_file.write(f"  Pages Processed: {len(image_paths)}\n")
                summary_file.write(f"  Status: {'Success' if image_paths else 'Failed'}\n")
                if image_paths:
                    summary_file.write(f"  Images: {[os.path.basename(p) for p in image_paths[:5]]}{'...' if len(image_paths) > 5 else ''}\n\n")
                else:
                    summary_file.write(f"  No images extracted\n\n")
                
            # Force garbage collection after each PDF
            gc.collect()
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            results[pdf_path] = []
            
            # Update summary for failed PDF
            with open(summary_path, 'a') as summary_file:
                summary_file.write(f"PDF: {os.path.basename(pdf_path)}\n")
                summary_file.write(f"  Status: Failed with error\n")
                summary_file.write(f"  Error: {str(e)}\n\n")
    
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
        if pdf_count > 0:
            summary_file.write(f"Average Time Per PDF: {processing_time/pdf_count:.2f} seconds\n")
            if total_pages > 0:
                summary_file.write(f"Average Time Per Page: {processing_time/total_pages:.2f} seconds\n")
    
    logger.info(f"Batch PDF processing completed in {processing_time:.2f} seconds for {pdf_count} PDF files ({total_pages} pages)")
    
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
            
            # Run evaluation on this subfolder
            try:
                # Run evaluation
                evaluation_result = evaluate(subfolder_path, template_dir=subfolder_path, save_dir=subfolder_output)
                
                # Update PDF page count for this batch
                pdf_files = [f for f in files if f.lower().endswith('.pdf')]
                pdf_pages = count_pdf_pages(subfolder_path)
                pdf_pages_count += pdf_pages
                
                # Find all CSV files in the output directory
                batch_csv_files = glob.glob(os.path.join(subfolder_output, "*.csv"))
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