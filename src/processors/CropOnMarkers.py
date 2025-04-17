import os

import cv2
import numpy as np

from src.logger import logger
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


class CropOnMarkers(ImagePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.tuning_config
        marker_ops = self.options
        self.threshold_circles = []
        
        # options with defaults
        self.marker_path = os.path.join(
            self.relative_dir, marker_ops.get("relativePath", "omr_marker.jpg")
        )
        self.min_matching_threshold = marker_ops.get("min_matching_threshold", 0.3)
        self.max_matching_variation = marker_ops.get("max_matching_variation", 0.41)
        
        # Optimize scale range to reduce search space
        self.marker_rescale_range = tuple(
            int(r) for r in marker_ops.get("marker_rescale_range", (35, 90))
        )
        # Reduce steps for faster processing - larger step size
        self.marker_rescale_steps = int(marker_ops.get("marker_rescale_steps", 5))
        self.apply_erode_subtract = marker_ops.get("apply_erode_subtract", True)
        
        # Cache the marker for faster processing
        self.marker = self.load_marker(marker_ops, config)
        
        # Pre-compute rescaled markers for all steps to avoid repeating this calculation
        self._cached_markers = {}
        descent_per_step = (
            self.marker_rescale_range[1] - self.marker_rescale_range[0]
        ) // self.marker_rescale_steps
        _h, _w = self.marker.shape[:2]
        
        for r0 in np.arange(
            self.marker_rescale_range[1],
            self.marker_rescale_range[0],
            -1 * descent_per_step,
        ):
            s = float(r0 * 1 / 100)
            if s == 0.0:
                continue
            self._cached_markers[s] = ImageUtils.resize_util_h(
                self.marker, u_height=int(_h * s)
            )

    def __str__(self):
        return self.marker_path

    def exclude_files(self):
        return [self.marker_path]

    def apply_filter(self, image, file_path):
        config = self.tuning_config
        image_instance_ops = self.image_instance_ops
        
        # Downscale image for faster processing
        scale_factor = 0.5
        small_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        
        # Apply erode subtract on smaller image
        small_eroded_sub = ImageUtils.normalize_util(
            small_image
            if self.apply_erode_subtract
            else (small_image - cv2.erode(small_image, kernel=np.ones((3, 3)), iterations=3))
        )
        
        # Quads on smaller image (faster processing)
        h1, w1 = small_eroded_sub.shape[:2]
        midh, midw = h1 // 3, w1 // 2
        quads = {}
        origins = [[0, 0], [midw, 0], [0, midh], [midw, midh]]
        quads[0] = small_eroded_sub[0:midh, 0:midw]
        quads[1] = small_eroded_sub[0:midh, midw:w1]
        quads[2] = small_eroded_sub[midh:h1, 0:midw]
        quads[3] = small_eroded_sub[midh:h1, midw:w1]

        # Draw Quadlines (only for visualization, could be skipped)
        small_eroded_sub[:, midw : midw + 2] = 255
        small_eroded_sub[midh : midh + 2, :] = 255

        # Find best match on smaller image (faster)
        best_scale, all_max_t = self.getBestMatch(small_eroded_sub)
        if best_scale is None:
            if config.outputs.show_image_level >= 1:
                InteractionUtils.show("Quads", small_eroded_sub, config=config)
            return None

        # Use the cached marker instead of resizing again
        optimal_marker = self._cached_markers.get(best_scale)
        if optimal_marker is None:
            # Fallback if not in cache
            optimal_marker = ImageUtils.resize_util_h(
                self.marker, u_height=int(self.marker.shape[0] * best_scale)
            )
            
        # Scale down optimal marker to match the downscaled image
        small_optimal_marker = cv2.resize(optimal_marker, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        _h, w = small_optimal_marker.shape[:2]
        
        centres = []
        sum_t, max_t = 0, 0
        quarter_match_log = "Matching Marker:  "
        
        # Adjust template matching to be faster (use TM_CCORR_NORMED instead of TM_CCOEFF_NORMED for speed)
        for k in range(0, 4):
            res = cv2.matchTemplate(quads[k], small_optimal_marker, cv2.TM_CCORR_NORMED)
            max_t = res.max()
            quarter_match_log += f"Quarter{str(k + 1)}: {str(round(max_t, 3))}\t"
            if (
                max_t < self.min_matching_threshold
                or abs(all_max_t - max_t) >= self.max_matching_variation
            ):
                logger.error(
                    file_path,
                    "\nError: No circle found in Quad",
                    k + 1,
                    "\n\t min_matching_threshold",
                    self.min_matching_threshold,
                    "\t max_matching_variation",
                    self.max_matching_variation,
                    "\t max_t",
                    max_t,
                    "\t all_max_t",
                    all_max_t,
                )
                if config.outputs.show_image_level >= 1:
                    InteractionUtils.show(
                        f"No markers: {file_path}",
                        small_eroded_sub,
                        0,
                        config=config,
                    )
                    InteractionUtils.show(
                        f"res_Q{str(k + 1)} ({str(max_t)})",
                        res,
                        1,
                        config=config,
                    )
                return None

            pt = np.argwhere(res == max_t)[0]
            pt = [pt[1], pt[0]]
            pt[0] += origins[k][0]
            pt[1] += origins[k][1]
            
            # Scale up the points to match the original image
            scaled_pt = [int(pt[0] / scale_factor), int(pt[1] / scale_factor)]
            
            # Scale up marker dimensions 
            full_w, full_h = int(w / scale_factor), int(_h / scale_factor)
            
            image = cv2.rectangle(
                image, tuple(scaled_pt), (scaled_pt[0] + full_w, scaled_pt[1] + full_h), (150, 150, 150), 2
            )
            
            # We only need the centers for the four point transform
            centres.append([scaled_pt[0] + full_w / 2, scaled_pt[1] + full_h / 2])
            sum_t += max_t

        # Comment out detailed marker matching logs
        # logger.info(quarter_match_log)
        # logger.info(f"Optimal Scale: {best_scale}")
        
        # analysis data
        self.threshold_circles.append(sum_t / 4)

        # Apply four point transform on the original image
        image = ImageUtils.four_point_transform(image, np.array(centres))
        
        # Skip saving intermediate images when not needed to save time
        if config.outputs.show_image_level >= 2:
            # Only upscale small_eroded_sub if needed for visualization
            image_eroded_sub = cv2.resize(small_eroded_sub, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_LINEAR)
            image_instance_ops.append_save_img(2, image_eroded_sub)
            
            if config.outputs.show_image_level >= 2 and config.outputs.show_image_level < 4:
                image_eroded_sub = ImageUtils.resize_util_h(
                    image_eroded_sub, image.shape[0]
                )
                image_eroded_sub[:, -5:] = 0
                h_stack = np.hstack((image_eroded_sub, image))
                InteractionUtils.show(
                    f"Warped: {file_path}",
                    ImageUtils.resize_util(
                        h_stack, int(config.dimensions.display_width * 1.6)
                    ),
                    0,
                    0,
                    [0, 0],
                    config=config,
                )
        
        return image

    def load_marker(self, marker_ops, config):
        if not os.path.exists(self.marker_path):
            logger.error(
                "Marker not found at path provided in template:",
                self.marker_path,
            )
            exit(31)

        marker = cv2.imread(self.marker_path, cv2.IMREAD_GRAYSCALE)

        if "sheetToMarkerWidthRatio" in marker_ops:
            marker = ImageUtils.resize_util(
                marker,
                config.dimensions.processing_width
                / int(marker_ops["sheetToMarkerWidthRatio"]),
            )
        # Use smaller kernel for faster processing
        marker = cv2.GaussianBlur(marker, (3, 3), 0)
        marker = cv2.normalize(
            marker, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )

        if self.apply_erode_subtract:
            # Use smaller kernel and fewer iterations for faster processing
            marker -= cv2.erode(marker, kernel=np.ones((3, 3)), iterations=3)

        return marker

    # Resizing the marker within scaleRange at rate of descent_per_step to
    # find the best match.
    def getBestMatch(self, image_eroded_sub):
        config = self.tuning_config
        descent_per_step = (
            self.marker_rescale_range[1] - self.marker_rescale_range[0]
        ) // self.marker_rescale_steps
        
        _h, _w = self.marker.shape[:2]
        res, best_scale = None, None
        all_max_t = 0
        
        # Use downscaled images for faster matching
        scale_factor = 0.5
        small_image_eroded_sub = cv2.resize(image_eroded_sub, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        for r0 in np.arange(
            self.marker_rescale_range[1],
            self.marker_rescale_range[0],
            -1 * descent_per_step,
        ):  # reverse order
            s = float(r0 * 1 / 100)
            if s == 0.0:
                continue
                
            # Use cached marker
            rescaled_marker = self._cached_markers.get(s)
            if rescaled_marker is None:
                rescaled_marker = ImageUtils.resize_util_h(
                    self.marker, u_height=int(_h * s)
                )
                
            # Downscale marker for faster template matching
            small_marker = cv2.resize(rescaled_marker, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            
            # Use TM_CCORR_NORMED which is faster than TM_CCOEFF_NORMED
            res = cv2.matchTemplate(
                small_image_eroded_sub, small_marker, cv2.TM_CCORR_NORMED
            )

            max_t = res.max()
            if all_max_t < max_t:
                best_scale, all_max_t = s, max_t

        if all_max_t < self.min_matching_threshold:
            logger.warning(
                "\tTemplate matching too low! Consider rechecking preProcessors applied before this."
            )
            if config.outputs.show_image_level >= 1:
                InteractionUtils.show("res", res, 1, 0, config=config)

        if best_scale is None:
            logger.warning(
                "No matchings for given scaleRange:", self.marker_rescale_range
            )
        return best_scale, all_max_t
