"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from src.logger import logger

plt.rcParams["figure.figsize"] = (10.0, 8.0)
CLAHE_HELPER = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))


class ImageUtils:
    """A Static-only Class to hold common image processing utilities & wrappers over OpenCV functions"""

    @staticmethod
    def save_img(path, im):
        """
        Save an image to specified path with error checking
        
        Args:
            path: Path to save image
            im: Image to save
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Check if path is valid
            if not path or not isinstance(path, str):
                logger.error(f"Invalid save path: {path}")
                return False
                
            # Check if image is None
            if im is None:
                logger.error(f"Cannot save None image to {path}")
                return False
                
            # Get the directory path
            directory = os.path.dirname(path)
            
            # Create directory if it doesn't exist
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to create directory {directory}: {str(e)}")
                    return False
                
            # Check if image is empty or has invalid dimensions
            if len(im.shape) < 2:
                logger.error(f"Invalid image format (shape={im.shape}) for {path}")
                return False
                
            h, w = im.shape[:2]
            if h <= 0 or w <= 0:
                logger.error(f"Cannot save image with invalid dimensions (h={h}, w={w}) to {path}")
                return False
                
            # Save the image
            result = cv2.imwrite(path, im)
            if not result:
                logger.error(f"cv2.imwrite failed to save image to {path}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving image to {path}: {str(e)}")
            return False

    @staticmethod
    def resize_util(img, u_width, u_height=None):
        """
        Resize image with proper error handling
        
        Args:
            img: Input image
            u_width: Target width
            u_height: Target height (calculated from aspect ratio if None)
            
        Returns:
            Resized image
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
            if u_height is None:
                # Maintain aspect ratio
                u_height = int(h * u_width / w)
                
            # Ensure target dimensions are valid
            if u_width <= 0 or u_height <= 0:
                return None
                
            # Perform resize with integer dimensions
            return cv2.resize(img, (int(u_width), int(u_height)))
            
        except Exception as e:
            logger.error(f"Error during image resize: {str(e)}")
            return None

    @staticmethod
    def resize_util_h(img, u_height, u_width=None):
        """
        Resize image by height with proper error handling
        
        Args:
            img: Input image
            u_height: Target height
            u_width: Target width (calculated from aspect ratio if None)
            
        Returns:
            Resized image
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
                
            # Calculate width if not provided
            if u_width is None:
                # Maintain aspect ratio
                u_width = int(w * u_height / h)
                
            # Ensure target dimensions are valid
            if u_width <= 0 or u_height <= 0:
                logger.error(f"Invalid target dimensions: {u_width}x{u_height}")
                return None
                
            # Perform resize with integer dimensions
            return cv2.resize(img, (int(u_width), int(u_height)))
            
        except Exception as e:
            logger.error(f"Error during image resize: {str(e)}")
            return None

    @staticmethod
    def grab_contours(cnts):
        # source: imutils package

        # if the length the contours tuple returned by cv2.findContours
        # is '2' then we are using either OpenCV v2.4, v4-beta, or
        # v4-official
        if len(cnts) == 2:
            cnts = cnts[0]

        # if the length of the contours tuple is '3' then we are using
        # either OpenCV v3, v4-pre, or v4-alpha
        elif len(cnts) == 3:
            cnts = cnts[1]

        # otherwise OpenCV has changed their cv2.findContours return
        # signature yet again and I have no idea WTH is going on
        else:
            raise Exception(
                (
                    "Contours tuple must have length 2 or 3, "
                    "otherwise OpenCV changed their cv2.findContours return "
                    "signature yet again. Refer to OpenCV's documentation "
                    "in that case"
                )
            )

        # return the actual contours array
        return cnts

    @staticmethod
    def normalize_util(img, alpha=0, beta=255):
        return cv2.normalize(img, alpha, beta, norm_type=cv2.NORM_MINMAX)

    @staticmethod
    def auto_canny(image, sigma=0.93):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    @staticmethod
    def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = ImageUtils.order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        max_width = max(int(width_a), int(width_b))
        # max_width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

        # compute the height of the new image, which will be the
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        # max_height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-br)))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        transform_matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

        # return the warped image
        return warped

    @staticmethod
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect
