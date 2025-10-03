"""
Tests cosmicqc image module
"""

import os
import pathlib

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image
from skimage.draw import disk

from cytodataframe.image import (
    add_image_scale_bar,
    adjust_image_brightness,
    adjust_with_adaptive_histogram_equalization,
    draw_outline_on_image_from_mask,
    draw_outline_on_image_from_outline,
    get_pixel_bbox_from_offsets,
    is_image_too_dark,
)


def test_is_image_too_dark_with_dark_image(fixture_dark_image: Image):
    assert is_image_too_dark(fixture_dark_image, pixel_brightness_threshold=10.0)


def test_is_image_too_dark_with_bright_image(fixture_bright_image: Image):
    assert not is_image_too_dark(fixture_bright_image, pixel_brightness_threshold=10.0)


def test_is_image_too_dark_with_mid_brightness_image(
    fixture_mid_brightness_image: Image,
):
    assert not is_image_too_dark(
        fixture_mid_brightness_image, pixel_brightness_threshold=10.0
    )


def test_adjust_image_brightness_with_dark_image(fixture_dark_image: Image):
    adjusted_image = adjust_image_brightness(fixture_dark_image)
    # we expect that image to be too dark (it's all dark, so there's no adjustments)
    assert is_image_too_dark(adjusted_image, pixel_brightness_threshold=10.0)


def test_adjust_image_brightness_with_bright_image(fixture_bright_image: Image):
    adjusted_image = adjust_image_brightness(fixture_bright_image)
    # Since the image was already bright, it should remain bright
    assert not is_image_too_dark(adjusted_image, pixel_brightness_threshold=10.0)


def test_adjust_image_brightness_with_mid_brightness_image(
    fixture_mid_brightness_image: Image,
):
    adjusted_image = adjust_image_brightness(fixture_mid_brightness_image)
    # The image should still not be too dark after adjustment
    assert not is_image_too_dark(adjusted_image, pixel_brightness_threshold=10.0)


def test_adjust_nuclear_speckle_image_brightness(
    fixture_nuclear_speckle_example_image: Image,
):
    assert is_image_too_dark(fixture_nuclear_speckle_example_image)
    assert not is_image_too_dark(
        adjust_image_brightness(fixture_nuclear_speckle_example_image),
        pixel_brightness_threshold=3.0,
    )


@pytest.mark.parametrize(
    "orig_image, outline_image, expected_non_black_mask",
    [
        (
            np.zeros((10, 10, 3), dtype=np.uint8),  # All-black original image
            np.full((10, 10, 3), 255, dtype=np.uint8),  # White outline image
            True,  # All pixels are non-black
        ),
        (
            np.zeros((10, 10), dtype=np.uint8),  # Grayscale original image
            np.full((10, 10), 255, dtype=np.uint8),  # Grayscale white outline
            True,  # All pixels are non-black
        ),
        (
            np.zeros((10, 10, 3), dtype=np.uint8),  # RGB original image
            np.zeros((5, 5, 3), dtype=np.uint8),  # Mismatched size outline
            False,  # All pixels remain black after resizing
        ),
    ],
)
def test_draw_outline_on_image_from_outline(
    tmp_path: pathlib.Path,
    orig_image: np.ndarray,
    outline_image: np.ndarray,
    expected_non_black_mask: bool,
) -> None:
    """
    Tests draw_outline_on_image_from_outline.
    """
    # Save the outline image to a temporary path
    outline_image_path = tmp_path / "outline.png"
    imageio.imwrite(outline_image_path, outline_image)

    # Call the method
    result_image = draw_outline_on_image_from_outline(
        orig_image, str(outline_image_path)
    )

    # Validate results
    non_black_mask = np.any(result_image[..., :3] != 0, axis=-1)

    if expected_non_black_mask:
        assert np.any(non_black_mask), "Expected a non-black outline but got none."
    else:
        assert not np.any(non_black_mask), (
            "Expected no outline but got a non-black area."
        )


@pytest.mark.parametrize(
    "orig_image, mask_image, expected_outlines",
    [
        (
            np.zeros((10, 10, 3), dtype=np.uint8),  # RGB black original image
            np.zeros((10, 10), dtype=np.uint8),  # Binary mask with no objects
            False,  # No outline expected
        ),
        (
            np.zeros((10, 10, 3), dtype=np.uint8),  # RGB black original image
            np.pad(np.ones((6, 6), dtype=np.uint8), 2) * 255,  # Square mask
            True,  # Outline expected
        ),
        (
            np.zeros((10, 10), dtype=np.uint8),  # Grayscale original image
            np.zeros((10, 10), dtype=np.uint8),  # Binary mask with no objects
            False,  # No outline expected
        ),
        (
            np.zeros((20, 20, 3), dtype=np.uint8),  # Larger RGB black original image
            np.zeros((20, 20), dtype=np.uint8),  # Binary mask with a circle
            True,  # Outline expected
        ),
    ],
)
def test_draw_outline_on_image_from_mask(
    tmp_path: pathlib.Path,
    orig_image: np.ndarray,
    mask_image: np.ndarray,
    expected_outlines: bool,
) -> None:
    """
    Tests draw_outline_on_image_from_mask.
    """
    # Create a valid circular mask for case 3
    if mask_image.shape == (20, 20) and expected_outlines:
        rr, cc = disk((10, 10), 5)
        mask_image[rr, cc] = 255

    # Save the mask image to a temporary path
    mask_image_path = tmp_path / "mask.png"
    imageio.imwrite(mask_image_path, mask_image)

    # Call the method
    result_image = draw_outline_on_image_from_mask(orig_image, str(mask_image_path))

    # Check for green outlines in the result
    green_color = [0, 255, 0]
    mask = (
        (result_image == green_color).all(axis=-1) if result_image.ndim == 3 else None
    )

    if expected_outlines:
        assert mask is not None and mask.any(), "Expected outlines but found none."
    else:
        assert mask is None or not mask.any(), "Unexpected outlines found."

    # check that we can change the outline color
    result_image = draw_outline_on_image_from_mask(
        orig_image, str(mask_image_path), outline_color=(red_color := (255, 0, 0))
    )

    # Check for red outlines in the result
    mask = (result_image == red_color).all(axis=-1) if result_image.ndim == 3 else None


# Sample test data for different image types
@pytest.mark.parametrize(
    "input_image, expected_shape, is_exception_expected",
    [
        # Grayscale image (2D array)
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), (3, 3), False),
        # RGB image (3D array)
        (
            np.array(
                [
                    [[0, 0, 0], [255, 0, 0], [0, 255, 0]],
                    [[0, 0, 0], [0, 255, 0], [255, 0, 0]],
                    [[255, 0, 0], [0, 0, 0], [0, 255, 0]],
                ]
            ),
            (3, 3, 3),
            False,
        ),
        # RGBA image (4D array)
        (
            np.array(
                [
                    [[0, 0, 0, 255], [255, 0, 0, 255], [0, 255, 0, 255]],
                    [[0, 0, 0, 255], [0, 255, 0, 255], [255, 0, 0, 255]],
                    [[255, 0, 0, 255], [0, 0, 0, 255], [0, 255, 0, 255]],
                ]
            ),
            (3, 3, 4),
            False,
        ),
        # Invalid input (image with 5 channels or unsupported format)
        (np.array([[[0, 0, 0, 0, 0]]]), None, True),
    ],
)
def test_adjust_with_adaptive_histogram_equalization(
    input_image: np.ndarray, expected_shape: np.ndarray, is_exception_expected: bool
):
    """
    Test adjust_with_adaptive_histogram_equalization
    """
    if is_exception_expected:
        # Test if the function raises an exception for invalid input
        with pytest.raises(ValueError):
            adjust_with_adaptive_histogram_equalization(input_image)
    else:
        # Test if the function processes the image and
        # returns a result with the expected shape
        result = adjust_with_adaptive_histogram_equalization(input_image)
        assert result.shape == expected_shape, (
            f"Expected shape {expected_shape}, but got {result.shape}"
        )


def test_get_pixel_bbox_from_offsets():
    """
    Test get_pixel_bbox_from_offsets function.
    """

    # Get pixel bounding box
    bbox = get_pixel_bbox_from_offsets(
        center_x=300, center_y=400, rel_bbox=(-50, -50, 50, 50)
    )

    # Check if the bounding box is correct
    assert bbox == (250, 350, 350, 450), (
        f"Expected (250, 350, 350, 450), but got {bbox}"
    )


def test_add_image_scale_bar_lower_right_bbox_and_area(tmp_path: pathlib.Path):
    # --- Arrange ---
    H, W = 100, 80
    img = np.zeros((H, W), dtype=np.uint8)  # grayscale, all zeros

    um_per_pixel = 0.5
    length_um = 10.0  # => 20 px at 0.5 µm/px
    length_px = round(length_um / um_per_pixel)
    thickness_px = 4
    margin_px = 10
    color = (255, 255, 255)
    location = "lower right"

    # --- Act ---
    out = add_image_scale_bar(
        img,
        um_per_pixel,
        length_um=length_um,
        thickness_px=thickness_px,
        color=color,
        location=location,
        margin_px=margin_px,
        label=False,  # keep dependencies minimal for the test
    )

    # --- Assert: input unchanged & output type/shape ---
    assert out is not img, "Function should return a new array (not modify in-place)."
    assert img.ndim == 2 and img.dtype == np.uint8 and img.shape == (H, W)
    assert out.ndim == 3 and out.dtype == np.uint8 and out.shape == (H, W, 3)

    # --- Assert: detect bar pixels robustly ---
    # If anti-aliasing ever appears, use a high threshold; here we expect solid color.
    bar_mask = (out[:, :, 0] >= 250) & (out[:, :, 1] >= 250) & (out[:, :, 2] >= 250)
    bar_coords = np.argwhere(bar_mask)

    assert bar_coords.size > 0, "No bright pixels found for the scale bar."

    # Bounding box of detected bar pixels
    ys = bar_coords[:, 0]
    xs = bar_coords[:, 1]
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Dimensions inferred from bbox (inclusive indices)
    inferred_height = (y_max - y_min) + 1
    inferred_width = (x_max - x_min) + 1

    # Check area matches exactly length_px * thickness_px
    # (prevents extra stray pixels from counting)
    expected_area = length_px * thickness_px
    actual_area = int(bar_mask.sum())
    assert actual_area == expected_area, (
        f"Bar area mismatch: got {actual_area}, expected {expected_area}"
    )

    # Check inferred dimensions (order can be height x width or
    # width x height depending on draw)
    dims = sorted((inferred_height, inferred_width))
    expected_dims = sorted((thickness_px, length_px))
    assert dims == expected_dims, (
        f"Bar dims mismatch: got (h={inferred_height}, w={inferred_width}), "
        f"expected thickness={thickness_px}, length={length_px}"
    )

    # --- Assert: location near lower-right with small tolerance (±1 px) ---
    tol = 1
    # Expected bottom-most row and right-most col if anchored by margin
    exp_y_max = H - margin_px - 1
    exp_x_max = W - margin_px - 1

    assert abs(int(y_max) - exp_y_max) <= tol, (
        f"Bar not at expected vertical position: y_max={y_max}, exp≈{exp_y_max}"
    )
    assert abs(int(x_max) - exp_x_max) <= tol, (
        f"Bar not at expected horizontal position: x_max={x_max}, exp≈{exp_x_max}"
    )

    # --- Assert: immediate neighbors outside the bbox remain
    # background (when in-bounds) ---
    if y_min - 1 >= 0:
        assert np.all(out[y_min - 1, x_min : x_max + 1, :] == 0)
    if x_min - 1 >= 0:
        assert np.all(out[y_min : y_max + 1, x_min - 1, :] == 0)

    H, W = 100, 80
    img = np.zeros((H, W), dtype=np.uint8)

    out = add_image_scale_bar(
        img,
        um_per_pixel=0.5,
        length_um=10,
        thickness_px=4,
        color=(255, 255, 255),
        location="lower right",
        margin_px=10,
        label=False,
    )

    # always save to tmp so you can inspect even without opening
    out_path = tmp_path / "with_bar.png"
    Image.fromarray(out).save(out_path)

    # optionally show (useful locally; safe-off in CI)
    if os.environ.get("SHOW_TEST_IMAGES") == "1":
        Image.fromarray(out).show()  # opens Preview/Photos/etc.
