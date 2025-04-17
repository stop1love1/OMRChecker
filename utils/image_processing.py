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

PDF_CACHE = {}
MAX_CACHE_SIZE = 50

def process_pdf_page(page_info):
    """Process a single PDF page with PyMuPDF"""
    page, matrix, output_path, quality = page_info
    try:
        pix = page.get_pixmap(matrix=matrix)
        try:
            pix.save(output_path, quality=quality)
        except TypeError:
            pix.save(output_path)
            logger.debug(f"Using older PyMuPDF version without quality parameter support")
        pix = None
        return output_path
    except Exception as e:
        logger.error(f"Error processing PDF page: {str(e)}")
        return None

def process_pdf2image_page(page_info):
    """Process a single page with pdf2image"""
    image, output_path, quality = page_info
    image.save(output_path, 'JPEG', quality=quality, optimize=True)
    return output_path

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
    
    def process_pdf_batch(self, pdf_paths, input_dir, dpi=100, quality=70):
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
            # Adjust DPI based on number of files for better performance
            if len(sorted_pdf_paths) > 20:
                dpi = max(72, dpi - 20)  # Reduce DPI for large batches
                quality = max(55, quality - 10)  # Reduce quality for large batches
                logger.info(f"Large batch detected, reducing DPI to {dpi} and quality to {quality}")
            
            # Submit all PDF processing jobs
            future_to_pdf = {}
            for pdf_path in sorted_pdf_paths:
                future = executor.submit(
                    self._process_single_pdf_monitored, 
                    pdf_path, 
                    input_dir, 
                    dpi, 
                    quality
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
    
    def _process_single_pdf_monitored(self, pdf_path, input_dir, dpi, quality):
        """Process a single PDF with resource monitoring"""
        # Check cache first
        path_str = str(pdf_path)
        cache_key = f"{path_str}_{dpi}_{quality}"
        
        if cache_key in PDF_CACHE:
            cached_data = PDF_CACHE[cache_key]
            # Verify the cached files still exist
            all_exist = True
            if "image_paths" in cached_data:
                for img_path in cached_data["image_paths"]:
                    if not os.path.exists(img_path):
                        all_exist = False
                        break
                
                if all_exist:
                    logger.info(f"Using cached version of PDF {os.path.basename(path_str)}")
                    return cached_data["image_paths"]
        
        # Wait until we have resources available
        with self.lock:
            while self.active_jobs >= self.pdf_workers:
                self.job_finished.wait()
            self.active_jobs += 1
        
        try:
            # Add timeout to prevent hanging on corrupt PDFs
            try:
                # Process the PDF with a timeout of 60 seconds
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(process_single_pdf, pdf_path, input_dir, dpi, quality, self.page_workers)
                    result = future.result(timeout=60)
                    
                    # Cache the result
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
            # Release resources
            with self.lock:
                self.active_jobs -= 1
                self.job_finished.notify()

def process_single_pdf(pdf_path, input_dir, dpi=100, quality=70, max_workers=4):
    """
    Process a single PDF file and convert to images
    
    Args:
        pdf_path: Path to the PDF file
        input_dir: Directory to save the converted images
        dpi: DPI for rendering (lower = faster but less detail)
        quality: JPEG quality (lower = smaller files, faster)
        max_workers: Maximum number of concurrent workers for processing
        
    Returns:
        List of paths to converted images
    """
    # Skip processing if file is too large (>50MB) unless forced
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    if file_size_mb > 50:
        # For large files, reduce quality and DPI even more
        logger.warning(f"Large PDF detected ({file_size_mb:.1f}MB), reducing quality settings")
        dpi = max(72, dpi - 20)
        quality = max(50, quality - 15)
    
    image_paths = []
    pdf_processed = False
    pdf_start_time = time.time()
    
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
            
        # Calculate matrix based on provided DPI (default 72dpi in PDF)
        # Use lower DPI for faster processing
        matrix = fitz.Matrix(dpi/72, dpi/72)
        
        # Check if PDF is encrypted
        if pdf_document.is_encrypted:
            try:
                # Try to decrypt with empty password
                pdf_document.authenticate("")
            except:
                logger.warning(f"PDF is encrypted and could not be decrypted: {os.path.basename(pdf_path)}")
                return []
        
        # Process pages in parallel for speed
        tasks = []
        for i in range(pdf_document.page_count):
            try:
                page = pdf_document[i]
                img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1}.jpg"
                img_path = os.path.join(input_dir, img_filename)
                tasks.append((page, matrix, img_path, quality))
            except Exception as page_error:
                logger.warning(f"Error accessing page {i} in PDF {pdf_path}: {str(page_error)}")
                continue
        
        # Use efficient chunk processing with error handling
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process in chunks for better memory management
            try:
                from utils.batch_config import BATCH_SIZES
                chunk_size = BATCH_SIZES.get("pdf_page_chunk", 20)
            except ImportError:
                chunk_size = min(20, len(tasks))
                
            for i in range(0, len(tasks), chunk_size):
                chunk_tasks = tasks[i:i+chunk_size]
                futures = {executor.submit(process_pdf_page, task): task for task in chunk_tasks}
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        path = future.result()
                        if path:
                            image_paths.append(path)
                    except Exception as e:
                        logger.error(f"Error processing PDF page: {str(e)}")
                
                # Force garbage collection after each chunk
                gc.collect()
        
        pdf_document.close()
        pdf_processed = True
        
    except ImportError as e:
        logger.warning(f"PyMuPDF not available: {str(e)}. Will try pdf2image...")
    except Exception as e:
        logger.warning(f"Error using PyMuPDF: {str(e)}. Will try pdf2image...")
        
    # Fall back to pdf2image if PyMuPDF failed
    if not pdf_processed:
        try:
            from pdf2image import convert_from_path
            
            try:
                # Optimize pdf2image conversion with multithreading and lower DPI
                pdf_images = convert_from_path(
                    pdf_path, 
                    dpi=dpi,  # Lower DPI for faster processing
                    thread_count=max(1, max_workers//2),  # Use parallel processing but avoid too many threads
                    use_pdftocairo=True,  # Faster than pdftoppm
                    output_folder=input_dir,  # Write directly to output folder
                    fmt="jpeg",  # Use JPEG format - much faster than PNG
                    jpegopt={"quality": quality, "optimize": True, "progressive": False},
                    paths_only=True,  # Return paths only to reduce memory usage
                )
                
                # If paths_only works, we have the paths already
                if isinstance(pdf_images, list) and pdf_images and isinstance(pdf_images[0], str):
                    image_paths = pdf_images
                    pdf_processed = True
                else:
                    # Process images in parallel - only if paths_only didn't work
                    tasks = []
                    for i, image in enumerate(pdf_images):
                        img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1}.jpg"
                        img_path = os.path.join(input_dir, img_filename)
                        tasks.append((image, img_path, quality))
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_image = {executor.submit(process_pdf2image_page, task): task for task in tasks}
                        for future in concurrent.futures.as_completed(future_to_image):
                            try:
                                path = future.result()
                                image_paths.append(path)
                            except Exception as e:
                                logger.error(f"Error saving PDF image: {str(e)}")
                    
                    pdf_processed = True
                
            except Exception as poppler_err:
                logger.warning(f"Default poppler path failed: {str(poppler_err)}. Trying alternate paths...")
                
                # Try different poppler paths for different systems
                poppler_paths = [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/homebrew/bin',  # MacOS Homebrew
                    '/usr/lib/x86_64-linux-gnu',
                    'C:/Program Files/poppler/bin',  # Windows paths
                    'C:/poppler/bin'
                ]
                
                for poppler_path in poppler_paths:
                    try:
                        pdf_images = convert_from_path(
                            pdf_path, 
                            dpi=dpi,
                            thread_count=max(1, max_workers//2),
                            use_pdftocairo=True,
                            poppler_path=poppler_path,
                            output_folder=input_dir,
                            fmt="jpeg",
                            jpegopt={"quality": quality, "optimize": True, "progressive": False},
                            paths_only=True
                        )
                        
                        if isinstance(pdf_images, list) and pdf_images and isinstance(pdf_images[0], str):
                            image_paths = pdf_images
                            pdf_processed = True
                            break
                        
                        # Process images in parallel - only if paths_only didn't work
                        tasks = []
                        for i, image in enumerate(pdf_images):
                            img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1}.jpg"
                            img_path = os.path.join(input_dir, img_filename)
                            tasks.append((image, img_path, quality))
                        
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            future_to_image = {executor.submit(process_pdf2image_page, task): task for task in tasks}
                            for future in concurrent.futures.as_completed(future_to_image):
                                try:
                                    path = future.result()
                                    image_paths.append(path)
                                except Exception as e:
                                    logger.error(f"Error saving PDF image: {str(e)}")
                        
                        pdf_processed = True
                        break
                        
                    except Exception as path_err:
                        logger.warning(f"Failed with poppler path {poppler_path}: {str(path_err)}")
                else:
                    raise Exception("All poppler paths failed. Make sure poppler-utils is correctly installed.")
            
        except ImportError as e:
            logger.error(f"pdf2image not available: {str(e)}")
            raise ImportError("Neither PyMuPDF nor pdf2image are installed. Please install at least one of these libraries to process PDF files: 'pip install PyMuPDF pdf2image'")
        except Exception as e:
            logger.error(f"Error processing PDF with pdf2image: {str(e)}")
            raise
    
    if not pdf_processed:
        raise Exception("Failed to process PDF with any available method")
    
    pdf_processing_time = time.time() - pdf_start_time
    page_count = len(image_paths)
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

def process_pdf(pdf_path, input_dir, dpi=100, quality=70, max_workers=12):
    """
    Process PDF file and convert to images
    
    Args:
        pdf_path: Path to the PDF file
        input_dir: Directory to save the converted images
        dpi: DPI for rendering (lower = faster but less detail)
        quality: JPEG quality (lower = smaller files, faster)
        max_workers: Maximum number of concurrent workers for processing
        
    Returns:
        List of paths to converted images
    """
    start_time = time.time()
    
    # Get singleton instance of processing pool
    pool = PDFProcessingPool.get_instance()
    
    # Fast check PDF before processing
    info = fast_pdf_check(pdf_path)
    if not info.get("valid", False):
        logger.error(f"Invalid PDF file: {pdf_path}")
        return []
    
    # For large PDFs (many pages), reduce DPI and quality
    if info.get("page_count", 0) > 50:
        dpi = max(72, dpi - 20)  # Minimum 72 DPI (1:1 scale)
        quality = max(50, quality - 15)  # Minimum quality 50
        logger.info(f"Large PDF detected ({info['page_count']} pages), reducing DPI to {dpi} and quality to {quality}")
    
    # For batch processing, use the monitored version
    image_paths = pool._process_single_pdf_monitored(pdf_path, input_dir, dpi, quality)
    
    end_time = time.time()
    processing_time = end_time - start_time
    page_count = len(image_paths)
    logger.info(f"PDF processing completed in {processing_time:.2f} seconds: {os.path.basename(pdf_path)}, {page_count} pages")
    
    return image_paths

def process_pdf_batch(pdf_paths, input_dir, dpi=100, quality=70):
    """
    Process multiple PDF files in a resource-managed batch
    
    Args:
        pdf_paths: List of paths to PDF files
        input_dir: Directory where to save the converted images
        dpi: DPI for rendering (lower = faster but less detail)
        quality: JPEG quality (lower = smaller files, faster)
        
    Returns:
        Dictionary mapping PDF paths to their converted image paths
    """
    start_time = time.time()
    
    # Pre-check PDFs to prioritize processing order
    valid_pdfs = []
    for pdf_path in pdf_paths:
        # First check if already in cache
        path_str = str(pdf_path)
        if path_str in PDF_CACHE:
            # If cached and first page still exists, consider it valid
            if "first_page" in PDF_CACHE[path_str] and PDF_CACHE[path_str]["first_page"] and os.path.exists(PDF_CACHE[path_str]["first_page"]):
                valid_pdfs.append((pdf_path, PDF_CACHE[path_str].get("page_count", 999)))
                continue
        
        # Not in cache, do a fast check
        info = fast_pdf_check(pdf_path)
        if info.get("valid", False):
            valid_pdfs.append((pdf_path, info.get("page_count", 999)))
    
    # Sort files by page count (process smaller PDFs first for better UX)
    valid_pdfs.sort(key=lambda x: x[1])
    sorted_pdf_paths = [p[0] for p in valid_pdfs]
    
    # Adjust DPI based on batch size
    if len(sorted_pdf_paths) > 20:
        dpi = max(72, dpi - 20)  # Reduce DPI for large batches
        quality = max(55, quality - 10)  # Reduce quality for large batches
    elif len(sorted_pdf_paths) > 5:
        dpi = max(80, dpi - 10)  # Slightly reduce DPI for medium batches
    
    # Get singleton instance of processing pool
    pool = PDFProcessingPool.get_instance()
    
    # Process all PDFs with resource management
    results = pool.process_pdf_batch(sorted_pdf_paths, input_dir, dpi, quality)
    
    end_time = time.time()
    processing_time = end_time - start_time
    pdf_count = len(sorted_pdf_paths)
    # Calculate total pages processed
    total_pages = sum(len(pages) for pages in results.values())
    
    logger.info(f"Batch PDF processing completed in {processing_time:.2f} seconds for {pdf_count} PDF files ({total_pages} pages)")
    if processing_time > 0 and total_pages > 0:
        logger.info(f"Performance: {total_pages/processing_time:.2f} pages/second")
    
    # Force garbage collection after batch processing
    gc.collect()
    
    return results 