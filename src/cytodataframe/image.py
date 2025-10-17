"""
Helper functions for working with images in the context of CytoDataFrames.
"""

from typing import Any, Dict, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import skimage
import skimage.measure
from PIL import Image, ImageEnhance
from skimage import draw, exposure
from skimage import draw as skdraw
from skimage.util import img_as_ubyte


def is_image_too_dark(
    image: Image.Image, pixel_brightness_threshold: float = 10.0
) -> bool:
    """
    Check if the image is too dark based on the mean brightness.
    By "too dark" we mean not as visible to the human eye.

    Args:
        image (Image):
            The input PIL Image.
        threshold (float):
            The brightness threshold below which the image is considered too dark.

    Returns:
        bool:
            True if the image is too dark, False otherwise.
    """
    # Convert the image to a numpy array and then to grayscale
    img_array = np.array(image)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

    # Calculate the mean brightness
    mean_brightness = np.mean(gray_image)

    return mean_brightness < pixel_brightness_threshold


def adjust_image_brightness(image: Image.Image) -> Image.Image:
    """
    Adjust the brightness of an image using histogram equalization.

    Args:
        image (Image):
            The input PIL Image.

    Returns:
        Image:
            The brightness-adjusted PIL Image.
    """
    # Convert the image to numpy array and then to grayscale
    img_array = np.array(image)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

    # Apply histogram equalization to improve the contrast
    equalized_image = cv2.equalizeHist(gray_image)

    # Convert back to RGBA
    img_array[:, :, 0] = equalized_image  # Update only the R channel
    img_array[:, :, 1] = equalized_image  # Update only the G channel
    img_array[:, :, 2] = equalized_image  # Update only the B channel

    # Convert back to PIL Image
    enhanced_image = Image.fromarray(img_array)

    # Slightly reduce the brightness
    enhancer = ImageEnhance.Brightness(enhanced_image)
    reduced_brightness_image = enhancer.enhance(0.7)

    return reduced_brightness_image


def draw_outline_on_image_from_outline(
    orig_image: np.ndarray, outline_image_path: str, outline_color: tuple = (0, 255, 0)
) -> np.ndarray:
    """
    Draws green outlines on an image based on a provided outline image and returns
    the combined result.

    Args:
        orig_image (np.ndarray):
            The original image on which the outlines will be drawn.
            It must be a grayscale or RGB image with shape `(H, W)` for
            grayscale or `(H, W, 3)` for RGB.
        outline_image_path (str):
            The file path to the outline image. This image will be used
            to determine the areas where the outlines will be drawn.
            It can be grayscale or RGB.
        outline_color (tuple):
            RGB color for the outline (default: green).

    Returns:
        np.ndarray:
            The original image with green outlines drawn on the non-black areas from
            the outline image. The result is returned as an RGB image with shape
            `(H, W, 3)`.
    """

    # Load the outline image
    outline_image = imageio.imread(outline_image_path)

    # Resize if necessary
    if outline_image.shape[:2] != orig_image.shape[:2]:
        outline_image = skimage.transform.resize(
            outline_image,
            orig_image.shape[:2],
            preserve_range=True,
            anti_aliasing=True,
        ).astype(orig_image.dtype)

    # Create a mask for non-black areas (with threshold)
    threshold = 10  # Adjust as needed
    # Grayscale
    if outline_image.ndim == 2:
        non_black_mask = outline_image > threshold
    else:  # RGB/RGBA
        non_black_mask = np.any(outline_image[..., :3] > threshold, axis=-1)

    # Ensure the original image is RGB
    if orig_image.ndim == 2:
        orig_image = np.stack([orig_image] * 3, axis=-1)
    elif orig_image.shape[-1] != 3:
        raise ValueError("Original image must have 3 channels (RGB).")

    # Ensure uint8 data type
    if orig_image.dtype != np.uint8:
        orig_image = (orig_image * 255).astype(np.uint8)

    # Apply the green outline
    combined_image = orig_image.copy()
    combined_image[non_black_mask] = outline_color

    return combined_image


def draw_outline_on_image_from_mask(
    orig_image: np.ndarray, mask_image_path: str, outline_color: tuple = (0, 255, 0)
) -> np.ndarray:
    """
    Draws green outlines on an image based on a binary mask and returns
    the combined result.

    Please note: masks are inherently challenging to use when working with
    multi-compartment datasets and may result in outlines that do not
    pertain to the precise compartment. For example, if an object mask
    overlaps with one or many other object masks the outlines may not
    differentiate between objects.

    Args:
        orig_image (np.ndarray):
            Image which a mask will be applied to. Must be a NumPy array.
        mask_image_path (str):
            Path to the binary mask image file.
        outline_color (tuple):
            RGB color for the outline (default: green).

    Returns:
        np.ndarray:
            The resulting image with the green outline applied.
    """
    # Load the binary mask image
    mask_image = imageio.imread(mask_image_path)

    # Ensure the original image is RGB
    # Grayscale input
    if orig_image.ndim == 2:
        orig_image = np.stack([orig_image] * 3, axis=-1)
    # Unsupported input
    elif orig_image.shape[-1] != 3:
        raise ValueError("Original image must have 3 channels (RGB).")

    # Ensure the mask is 2D (binary)
    if mask_image.ndim > 2:
        mask_image = mask_image[..., 0]  # Take the first channel if multi-channel

    # Detect contours from the mask
    contours = skimage.measure.find_contours(mask_image, level=0.5)

    # Create an outline image with the same shape as the original image
    outline_image = np.zeros_like(orig_image)

    # Draw contours as green lines
    for contour in contours:
        rr, cc = draw.polygon_perimeter(
            np.round(contour[:, 0]).astype(int),
            np.round(contour[:, 1]).astype(int),
            shape=orig_image.shape[:2],
        )
        # Assign green color to the outline in all three channels
        outline_image[rr, cc, :] = outline_color

    # Combine the original image with the green outline
    combined_image = orig_image.copy()
    mask = np.any(outline_image > 0, axis=-1)  # Non-zero pixels in the outline
    combined_image[mask] = outline_image[mask]

    return combined_image


def adjust_with_adaptive_histogram_equalization(
    image: np.ndarray, brightness: int = 50
) -> np.ndarray:
    """
    Adaptive histogram equalization with brightness and contrast tuning via gamma.

    Parameters:
        image (np.ndarray): Input image.
        brightness (int): 0 = dark, 50 = neutral, 100 = bright.

    Returns:
        np.ndarray: Adjusted image.
    """
    b = np.clip(brightness, 0, 100) / 100.0

    # Contrast settings (same as before)
    kernel_frac = (1 / 4) * (1 - b) + (1 / 12) * b
    kernel_size = (
        max(int(image.shape[0] * kernel_frac), 1),
        max(int(image.shape[1] * kernel_frac), 1),
    )

    clip_limit = 0.1 * (1 - b) + 0.01 * b
    nbins = int(128 * (1 - b) + 1024 * b)

    def equalize_and_adjust(channel: np.ndarray) -> np.ndarray:
        """
        Internal function to equalize and adjust a single channel.

        Args:
            channel (np.ndarray):
                The input channel to be processed.
        Returns:
            np.ndarray:
                The processed channel.
        """
        eq = exposure.equalize_adapthist(
            channel,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
        )
        brightness_shift = (b - 0.5) * 2  # [-1, 1]
        gamma = 1.0 - brightness_shift * 0.8  # e.g. 1.8 → dark, 0.2 → bright
        return exposure.adjust_gamma(eq, gamma=gamma)

    if image.ndim == 2:
        result = equalize_and_adjust(image)
        return img_as_ubyte(result)

    elif image.ndim == 3 and image.shape[2] == 3:
        result = np.stack(
            [equalize_and_adjust(image[:, :, i]) for i in range(3)], axis=-1
        )
        return img_as_ubyte(result)

    elif image.ndim == 3 and image.shape[2] == 4:
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]
        result = np.stack(
            [equalize_and_adjust(rgb[:, :, i]) for i in range(3)], axis=-1
        )
        return np.dstack([img_as_ubyte(result), alpha])

    else:
        raise ValueError("Unsupported image format. Use grayscale, RGB, or RGBA.")


def get_pixel_bbox_from_offsets(
    center_x: float, center_y: float, rel_bbox: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """
    Compute a pixel bbox given a center point and pixel offsets.

    Args:
        center_x: Center x-coordinate in pixels.
        center_y: Center y-coordinate in pixels.
        rel_bbox: 4-tuple of pixel offsets:
            (dx_min, dy_min, dx_max, dy_max)

    Returns:
        A 4-tuple (x_min, x_max, y_min, y_max), where x_min and y_min
        are clamped to be ≥ 0.  x_max and y_max are returned as-is.
    """
    dx_min, dy_min, dx_max, dy_max = rel_bbox

    # apply offsets
    x_min = center_x + dx_min
    x_max = center_x + dx_max
    y_min = center_y + dy_min
    y_max = center_y + dy_max

    # clamp lower bounds to zero and round to int
    x_min = max(0, round(x_min))
    y_min = max(0, round(y_min))

    # round upper bounds to int (no max-image clamp)
    x_max = round(x_max)
    y_max = round(y_max)

    # guard against inverted boxes (just in case)
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    return x_min, y_min, x_max, y_max


def add_image_scale_bar(  # noqa: PLR0913
    img: np.ndarray,
    um_per_pixel: float,
    *,
    length_um: float = 10.0,
    thickness_px: int = 4,
    color: Tuple[int, int, int] = (255, 255, 255),
    location: str = "lower right",
    margin_px: int = 10,
    **_: Dict[Any, Any],
) -> np.ndarray:
    """
    Add a scale bar to the lower or upper corner of an image.

    The function overlays a solid rectangular scale bar onto a grayscale,
    RGB, or RGBA image.  The bar's physical length in micrometers is
    converted to pixels using the provided microns-per-pixel value.
    Non-finite or out-of-range input values are sanitized before drawing.

    Args:
        img (np.ndarray):
            Input image, either 2-D grayscale or 3-D RGB/RGBA.
        um_per_pixel (float):
            Micrometers per pixel.  If None or ≤ 0, the image is returned
            unchanged.
        length_um (float, optional):
            Desired length of the scale bar in micrometers.  Defaults to
            10.0.
        thickness_px (int, optional):
            Thickness of the bar in pixels.  Defaults to 4.
        color (Tuple[int, int, int], optional):
            RGB color of the bar.  Defaults to white ``(255, 255, 255)``.
        location (str, optional):
            Placement of the bar: ``"lower right"``, ``"lower left"``,
            ``"upper right"``, or ``"upper left"``.  Defaults to
            ``"lower right"``.
        margin_px (int, optional):
            Distance in pixels between the bar and the image edges.
            Defaults to 10.
        **_ (Dict[Any, Any]):
            Additional keyword arguments ignored for forward
            compatibility.

    Returns:
        np.ndarray:
            A new RGB image with the scale bar drawn.  The array is always
            of type ``uint8`` and has shape ``(H, W, 3)``.

    Raises:
        ValueError: If the input image has an unsupported number of
            channels.

    Notes:
        The bar length is clamped to fit within image bounds after
        margins.  The function does not rely on ``skimage.color`` and is
        safe against NaN or Inf input values.
    """

    if um_per_pixel is None or um_per_pixel <= 0:
        return img

    # --- Sanitize input first: replace non-finite, clip to valid range, cast to uint8
    out = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)

    # If float image, try to guess range and bring to 0..255
    if np.issubdtype(out.dtype, np.floating):
        # If values look like 0..1, scale to 0..255; otherwise clip to 0..255
        if out.max() <= 1.0:
            out = out * 255.0
        out = np.clip(out, 0, 255)

    out = out.astype(np.uint8, copy=False)

    # ensure RGB without using skimage.color.* conversions
    if out.ndim == 2:  # grayscale -> stack
        out = np.dstack([out, out, out])
    elif out.ndim == 3 and out.shape[2] == 4:  # RGBA -> drop alpha for drawing
        out = out[:, :, :3]
    elif out.ndim != 3 or out.shape[2] != 3:
        raise ValueError(
            "Unsupported image shape for scale bar; expected gray/RGB/RGBA."
        )

    H, W = out.shape[:2]
    length_px = max(1, round(length_um / um_per_pixel))
    thickness_px = max(1, int(thickness_px))
    margin_px = max(0, int(margin_px))

    # clamp length so we never exceed width after margins
    length_px = min(length_px, max(1, W - 2 * margin_px))

    loc = location.lower().strip()

    y0 = (
        max(0, H - margin_px - thickness_px)
        if "lower" in loc
        else min(H - 1, margin_px)
    )
    x0 = max(0, W - margin_px - length_px) if "right" in loc else min(W - 1, margin_px)

    # Ensure extent stays inside image
    if y0 + thickness_px > H:
        thickness_px = H - y0
    if x0 + length_px > W:
        length_px = W - x0

    rr, cc = skdraw.rectangle(
        start=(y0, x0),
        extent=(thickness_px, length_px),
        shape=out.shape[:2],
    )
    out[rr, cc] = color

    return out
