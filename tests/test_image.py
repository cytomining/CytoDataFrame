"""
Tests cosmicqc image module
"""

import os
import pathlib
import warnings

import imageio.v2 as imageio
import numpy as np
import pytest
import skimage
from PIL import Image
from skimage.draw import disk

from cytodataframe.image import (
    add_image_scale_bar,
    adjust_image_brightness,
    adjust_with_adaptive_histogram_equalization,
    draw_outline_on_image_from_mask,
    draw_outline_on_image_from_outline,
    get_pixel_bbox_from_offsets,
    image_array_to_grayscale,
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


def test_colorconv_control_may_emit_warning():
    """
    CONTROL: Try a few skimage color conversions on a bad array to see if any
    emit the classic matmul divide-by-zero warning. If none do on this version
    of skimage, skip the control.
    """
    bad = np.array([[0.0, np.inf], [np.nan, 1.0]], dtype=float)

    def _gray2rgb(a: np.ndarray) -> np.ndarray:
        return skimage.color.gray2rgb(a)

    def _rgb2lab(a: np.ndarray) -> np.ndarray:
        return skimage.color.rgb2lab(np.dstack([a, a, a]))

    def _lab2rgb(a: np.ndarray) -> np.ndarray:
        stacked = np.dstack([a * 100.0, a * 255.0, a * 255.0])
        return skimage.color.lab2rgb(stacked)

    def _rgb2hsv(a: np.ndarray) -> np.ndarray:
        return skimage.color.rgb2hsv(np.dstack([a, a, a]))

    funcs = [_gray2rgb, _rgb2lab, _lab2rgb, _rgb2hsv]

    saw_warning = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=RuntimeWarning)
        for fn in funcs:
            try:
                _ = fn(bad)
            except Exception:
                # We only care about warnings, not exceptions
                pass

    for w in caught:
        if issubclass(
            w.category, RuntimeWarning
        ) and "divide by zero encountered in matmul" in str(w.message):
            saw_warning = True
            break

    if not saw_warning:
        pytest.skip(
            "No skimage colorconv matmul warning emitted in this environment; "
            "control not applicable."
        )


def test_add_image_scale_bar_avoids_colorconv_warning_with_bad_values():
    """add_image_scale_bar should not emit the colorconv matmul warning."""
    H, W = 60, 50
    img = np.zeros((H, W), dtype=float)
    img[0, 0] = np.nan
    img[0, 1] = np.inf
    img[1, 0] = -np.inf
    img[1, 1] = 2.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=RuntimeWarning)
        out = add_image_scale_bar(
            img,
            um_per_pixel=0.5,
            length_um=10.0,
            thickness_px=3,
            color=(255, 255, 255),
            location="lower right",
            margin_px=5,
            label=False,  # ignored via **kwargs
        )

    assert not any(
        (
            issubclass(w.category, RuntimeWarning)
            and "divide by zero encountered in matmul" in str(w.message)
        )
        for w in caught
    ), "Should not trigger the skimage colorconv matmul warning."
    assert out.ndim == 3 and out.shape == (H, W, 3) and out.dtype == np.uint8


def test_draw_outline_on_image_from_outline_raises_for_non_rgb(tmp_path: pathlib.Path):
    outline_image_path = tmp_path / "outline.png"
    imageio.imwrite(outline_image_path, np.zeros((5, 5), dtype=np.uint8))

    with pytest.raises(ValueError, match="3 channels"):
        draw_outline_on_image_from_outline(
            np.zeros((5, 5, 4), dtype=np.uint8),
            str(outline_image_path),
        )


def test_draw_outline_on_image_from_mask_raises_for_non_rgb(tmp_path: pathlib.Path):
    mask_image_path = tmp_path / "mask.png"
    imageio.imwrite(mask_image_path, np.zeros((5, 5), dtype=np.uint8))

    with pytest.raises(ValueError, match="3 channels"):
        draw_outline_on_image_from_mask(
            np.zeros((5, 5, 4), dtype=np.uint8),
            str(mask_image_path),
        )


def test_draw_outline_on_image_from_outline_converts_non_uint8(tmp_path: pathlib.Path):
    outline_image_path = tmp_path / "outline.png"
    imageio.imwrite(outline_image_path, np.full((5, 5), 255, dtype=np.uint8))
    orig = np.zeros((5, 5, 3), dtype=np.float32)
    out = draw_outline_on_image_from_outline(orig, str(outline_image_path))
    assert out.dtype == np.uint8


def test_draw_outline_on_image_from_mask_handles_multichannel_mask(
    tmp_path: pathlib.Path,
):
    mask = np.zeros((6, 6, 3), dtype=np.uint8)
    mask[2:4, 2:4, :] = 255
    mask_image_path = tmp_path / "mask_rgb.png"
    imageio.imwrite(mask_image_path, mask)
    out = draw_outline_on_image_from_mask(
        np.zeros((6, 6, 3), dtype=np.uint8), str(mask_image_path)
    )
    assert out.shape == (6, 6, 3)


def test_get_pixel_bbox_from_offsets_handles_inverted_axes():
    bbox = get_pixel_bbox_from_offsets(
        center_x=10, center_y=20, rel_bbox=(5, 8, -5, -8)
    )
    assert bbox == (5, 12, 15, 28)


def test_add_image_scale_bar_returns_original_when_um_per_pixel_nonpositive():
    img = np.zeros((8, 8), dtype=np.uint8)
    out = add_image_scale_bar(img, um_per_pixel=0.0)
    assert out is img


def test_add_image_scale_bar_scales_float_unit_range():
    img = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = add_image_scale_bar(
        img,
        um_per_pixel=0.5,
        length_um=2.0,
        thickness_px=1,
        margin_px=0,
    )
    assert out.dtype == np.uint8
    assert out.shape == (4, 4, 3)
    assert int(out.max()) <= 255


def test_add_image_scale_bar_raises_on_unsupported_shape():
    with pytest.raises(ValueError, match="Unsupported image shape"):
        add_image_scale_bar(
            np.zeros((2, 2, 2), dtype=np.uint8),
            um_per_pixel=0.5,
        )


def test_add_image_scale_bar_clips_extent_for_small_images():
    img = np.zeros((5, 5), dtype=np.uint8)
    out = add_image_scale_bar(
        img,
        um_per_pixel=0.1,
        length_um=50.0,
        thickness_px=4,
        location="upper left",
        margin_px=4,
    )
    # The clipped bar still exists and remains within bounds.
    assert out.shape == (5, 5, 3)
    assert np.any(np.all(out == (255, 255, 255), axis=-1))


def test_add_image_scale_bar_accepts_rgba_input():
    img = np.zeros((6, 6, 4), dtype=np.uint8)
    img[..., 3] = 200
    out = add_image_scale_bar(
        img,
        um_per_pixel=0.5,
        length_um=2.0,
        thickness_px=1,
        margin_px=0,
    )
    assert out.shape == (6, 6, 3)


def test_image_array_to_grayscale_supports_channel_layouts():
    """Grayscale conversion supports 2D, single-channel, RGB, and RGBA."""
    gray = np.full((4, 5), 7, dtype=np.uint8)
    assert image_array_to_grayscale(gray) is gray

    single = np.full((4, 5, 1), 9, dtype=np.uint8)
    out_single = image_array_to_grayscale(single)
    assert out_single.shape == (4, 5)
    assert np.all(out_single == 9)

    rgb = np.full((4, 5, 3), 100, dtype=np.uint8)
    assert image_array_to_grayscale(rgb).shape == (4, 5)

    rgba = np.full((4, 5, 4), 100, dtype=np.uint8)
    assert image_array_to_grayscale(rgba).shape == (4, 5)


def test_image_array_to_grayscale_raises_on_unsupported_shape():
    with pytest.raises(ValueError, match="Unsupported image format"):
        image_array_to_grayscale(np.zeros((4, 5, 5), dtype=np.uint8))


def test_is_image_too_dark_handles_grayscale_image():
    """Single-channel grayscale images are common in microscopy and must work."""
    dark = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
    bright = Image.fromarray(np.full((10, 10), 255, dtype=np.uint8))
    assert is_image_too_dark(dark)
    assert not is_image_too_dark(bright)


def test_adjust_image_brightness_handles_grayscale_image():
    """Brightness adjustment should not crash on single-channel grayscale input."""
    gray = Image.fromarray(np.full((20, 20), 5, dtype=np.uint8))
    adjusted = adjust_image_brightness(gray)
    # output is rebuilt as an RGB image
    assert np.array(adjusted).shape == (20, 20, 3)


def test_adjust_image_brightness_preserves_rgba_alpha():
    arr = np.full((10, 10, 4), 5, dtype=np.uint8)
    arr[..., 3] = 123  # distinct alpha
    adjusted = np.array(adjust_image_brightness(Image.fromarray(arr)))
    assert adjusted.shape == (10, 10, 4)
    # alpha channel is preserved through the adjustment
    assert np.all(adjusted[..., 3] == 123)


def test_adjust_with_adaptive_histogram_equalization_brightness_affects_output():
    """The brightness parameter must measurably change output intensities.

    Existing tests only checked output shape; this guards the brightness/gamma
    math so regressions in the tuning curve are caught.
    """
    ramp = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    dark = adjust_with_adaptive_histogram_equalization(ramp, brightness=10)
    bright = adjust_with_adaptive_histogram_equalization(ramp, brightness=90)
    # A higher brightness setting yields a brighter (higher mean) image.
    assert float(np.mean(bright)) > float(np.mean(dark))
    # The default (50) should land between the two extremes.
    neutral = adjust_with_adaptive_histogram_equalization(ramp, brightness=50)
    assert float(np.mean(dark)) <= float(np.mean(neutral)) <= float(np.mean(bright))


def test_draw_outline_on_image_from_outline_scales_float_values_correctly(
    tmp_path: pathlib.Path,
):
    """A non-zero float image must be scaled by 255 (not 256) into uint8."""
    path = tmp_path / "outline.png"
    # outline image with no bright pixels so the original is preserved
    imageio.imwrite(path, np.zeros((4, 4), dtype=np.uint8))
    orig = np.full((4, 4, 3), 0.5, dtype=np.float32)
    out = draw_outline_on_image_from_outline(orig, str(path))
    assert out.dtype == np.uint8
    # 0.5 * 255 -> 127 (truncated). 0.5 * 256 would give 128.
    assert int(out[0, 0, 0]) == 127


def test_draw_outline_on_image_from_mask_preserves_non_outline_pixels(
    tmp_path: pathlib.Path,
):
    """Pixels outside the drawn outline must retain their original values."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    rr, cc = disk((10, 10), 4)
    mask[rr, cc] = 255
    mask_path = tmp_path / "mask.png"
    imageio.imwrite(mask_path, mask)

    orig = np.full((20, 20, 3), 80, dtype=np.uint8)
    out = draw_outline_on_image_from_mask(orig, str(mask_path))

    outline_mask = (out == [0, 255, 0]).all(axis=-1)
    assert outline_mask.any(), "Expected an outline to be drawn."
    # corner pixel is far from the central circle outline; must be unchanged
    assert tuple(int(v) for v in out[0, 0]) == (80, 80, 80)
    # every non-outline pixel keeps the original intensity
    assert np.all(out[~outline_mask] == 80)


def test_add_image_scale_bar_preserves_float_unit_range_values():
    """A 0..1 float image must be scaled up by 255, not divided by 255."""
    img = np.full((10, 10), 0.5, dtype=np.float32)
    out = add_image_scale_bar(img, um_per_pixel=0.5, length_um=2.0, margin_px=0)
    assert out.dtype == np.uint8
    # background (non-bar) pixels should reflect 0.5 * 255 ~= 127, not ~0.
    assert int(out[0, 0, 0]) == 127


def test_add_image_scale_bar_upper_left_placement():
    """The scale bar honors an 'upper left' location request."""
    H, W = 60, 60
    out = add_image_scale_bar(
        np.zeros((H, W), dtype=np.uint8),
        um_per_pixel=0.5,
        length_um=5.0,
        thickness_px=3,
        location="upper left",
        margin_px=5,
    )
    bar = np.argwhere((out >= 250).all(axis=-1))
    assert bar.size > 0
    ys, xs = bar[:, 0], bar[:, 1]
    # bar sits near the top-left, not the bottom-right
    assert ys.min() < H // 2
    assert xs.min() < W // 2


def test_get_pixel_bbox_from_offsets_clamps_negative_lower_bounds_to_zero():
    """Compartments near an image edge must clamp lower bounds to 0, not 1."""
    # center near the top-left with offsets that push x_min/y_min negative
    bbox = get_pixel_bbox_from_offsets(
        center_x=0, center_y=0, rel_bbox=(-50, -50, 50, 50)
    )
    x_min, y_min, x_max, y_max = bbox
    assert x_min == 0
    assert y_min == 0
    assert x_max == 50
    assert y_max == 50
