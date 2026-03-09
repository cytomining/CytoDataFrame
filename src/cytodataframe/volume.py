import base64
import html
import io
import os
import pathlib
import uuid
from typing import Any, Callable, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

FALLBACK_NDIM = 2
MIN_VOLUME_NDIM = 3
DEFAULT_MAX_INLINE_VOLUME_BYTES = 16 * 1024 * 1024
VTK_JS_CDN_URL = "https://unpkg.com/@kitware/vtk.js@34.9.1/dist/vtk.js"
VTK_JS_URL_ENV_VAR = "CYTODATAFRAME_VTK_JS_URL"
VTK_JS_TRANSFER_FUNCTION_SNIPPET = (
    "const ctfun=vtk.Rendering.Core.vtkColorTransferFunction.newInstance();"
    "ctfun.addRGBPoint(0,0,0,0);"
    "ctfun.addRGBPoint(1,1,1,1);"
    "ctfun.addRGBPoint(255,1,1,1);"
    "const ofun=vtk.Common.DataModel.vtkPiecewiseFunction.newInstance();"
    "ofun.addPoint(0,0.0);"
    "ofun.addPoint(1,0.15);"
    "ofun.addPoint(255,0.2);"
)


def build_3d_image_html_stub(
    data_value: str,
    candidate_path: pathlib.Path,
    display_options: Optional[dict],
    message: str = "3D image",
) -> str:
    """Build a fallback HTML block for a 3D image cell.

    Args:
        data_value: Original cell value associated with the image.
        candidate_path: Resolved filesystem path for the image source.
        display_options: Display configuration containing optional width/height.
        message: Text shown in the fallback block.

    Returns:
        An HTML ``div`` string representing a non-interactive 3D placeholder.
    """
    display_options = display_options or {}
    width = display_options.get("width", "300px")
    height = display_options.get("height", "300px")

    html_style = [f"width:{width}"]
    if height is not None:
        html_style.append(f"height:{height}")
    html_style.extend(
        [
            "display:flex",
            "align-items:center",
            "justify-content:center",
            "background:#f6f6f6",
            "border:1px solid #ddd",
            "color:#555",
            "font-size:12px",
        ]
    )

    html_style_joined = ";".join(html_style)
    path_attr = html.escape(str(candidate_path), quote=True)
    value_attr = html.escape(str(data_value), quote=True)

    return (
        f'<div class="cyto-3d-image" data-image-path="{path_attr}" '
        f'data-image-value="{value_attr}" style="{html_style_joined}">'
        f"{html.escape(message)}"
        "</div>"
    )


def _resolve_vtk_js_url(display_options: Optional[dict]) -> str:
    display_options = display_options or {}
    configured = display_options.get("vtk_js_url")
    if not configured:
        configured = os.getenv(VTK_JS_URL_ENV_VAR)
    return str(configured) if configured else VTK_JS_CDN_URL


def build_3d_image_html_view(  # noqa: C901, PLR0912, PLR0913, PLR0915
    volume: np.ndarray,
    dims: Tuple[int, int, int],
    data_value: str,
    candidate_path: pathlib.Path,
    display_options: Optional[dict],
    label_volume: Optional[np.ndarray] = None,
) -> str:
    display_options = display_options or {}
    width = display_options.get("width", "300px")
    height = display_options.get("height", "300px")

    html_style = [
        f"width:{width}",
        f"height:{height}",
        "background:#f6f6f6",
        "border:1px solid #ddd",
    ]
    html_style_joined = ";".join(html_style)

    max_inline_volume_bytes = display_options.get(
        "max_inline_volume_bytes",
        DEFAULT_MAX_INLINE_VOLUME_BYTES,
    )
    try:
        max_inline_volume_bytes = max(1, int(max_inline_volume_bytes))
    except (TypeError, ValueError):
        max_inline_volume_bytes = DEFAULT_MAX_INLINE_VOLUME_BYTES

    volume_uint8 = np.array(volume, dtype=np.uint8, copy=True)
    label_binary: Optional[np.ndarray] = None
    if label_volume is not None:
        label_arr = np.asarray(label_volume)
        if label_arr.shape == volume_uint8.shape:
            label_binary = np.where(label_arr > 0, 255, 0).astype(np.uint8, copy=False)

    inline_payload_nbytes = volume_uint8.nbytes
    if label_binary is not None:
        inline_payload_nbytes += label_binary.nbytes
    if inline_payload_nbytes > max_inline_volume_bytes:
        return build_3d_image_html_stub(
            data_value=data_value,
            candidate_path=candidate_path,
            display_options=display_options,
            message=(
                "3D image too large for inline rendering "
                f"({inline_payload_nbytes} bytes > {max_inline_volume_bytes} bytes)"
            ),
        )

    volume_bytes = volume_uint8.tobytes()
    volume_b64 = base64.b64encode(volume_bytes).decode("utf-8")
    label_b64 = ""
    label_color_attr = "0,1,0"
    label_opacity_attr = "0.95"
    if label_binary is not None:
        label_b64 = base64.b64encode(label_binary.tobytes()).decode("utf-8")
        overlay_color = display_options.get("label_overlay_color", (0, 255, 0))
        if (
            isinstance(overlay_color, (list, tuple))
            and len(overlay_color) >= MIN_VOLUME_NDIM
        ):
            try:
                color = np.asarray(overlay_color[:MIN_VOLUME_NDIM], dtype=np.float32)
                if np.max(color, initial=0.0) > 1.0:
                    color = np.clip(color / 255.0, 0.0, 1.0)
                else:
                    color = np.clip(color, 0.0, 1.0)
                label_color_attr = ",".join(f"{float(v):.6f}" for v in color)
            except Exception:
                label_color_attr = "0,1,0"
        try:
            opacity = float(display_options.get("label_overlay_opacity", 0.95))
            opacity = min(1.0, max(0.0, opacity))
            label_opacity_attr = f"{opacity:.6f}"
        except (TypeError, ValueError):
            label_opacity_attr = "0.95"
    dims_attr = ",".join(str(value) for value in dims)
    element_id = f"cyto-3d-{uuid.uuid4().hex}"
    path_attr = html.escape(str(candidate_path), quote=True)
    value_attr = html.escape(str(data_value), quote=True)

    fallback_html = ""
    try:
        if volume.ndim >= MIN_VOLUME_NDIM:
            fallback = volume.max(axis=0)
            if fallback.ndim == FALLBACK_NDIM:
                fallback = np.asarray(fallback)
                if fallback.size:
                    try:
                        lo, hi = np.percentile(fallback, (1.0, 99.9))
                        if hi <= lo:
                            hi = lo + 1.0
                        fallback = np.clip((fallback - lo) / (hi - lo), 0, 1)
                    except Exception:
                        fallback = fallback.astype(np.float32, copy=False)
                        vmin = float(np.min(fallback))
                        vmax = float(np.max(fallback))
                        if vmax <= vmin:
                            vmax = vmin + 1.0
                        fallback = np.clip((fallback - vmin) / (vmax - vmin), 0, 1)
                fallback_uint8 = (fallback * 255).astype(np.uint8, copy=False)
                png_bytes_io = io.BytesIO()
                imageio.imwrite(png_bytes_io, fallback_uint8, format="png")
                png_bytes = png_bytes_io.getvalue()
                png_b64 = base64.b64encode(png_bytes).decode("utf-8")
                fallback_html = (
                    '<img class="cyto-3d-fallback" '
                    f'src="data:image/png;base64,{png_b64}" '
                    f'style="{html_style_joined}"/>'
                )
    except Exception:
        fallback_html = ""

    vtk_js_url = _resolve_vtk_js_url(display_options)
    return (
        f'<div id="{element_id}" class="cyto-3d-image" '
        f'data-image-path="{path_attr}" data-image-value="{value_attr}" '
        f'data-volume="{volume_b64}" data-dims="{dims_attr}" '
        f'data-label-volume="{label_b64}" '
        f'data-label-color="{label_color_attr}" '
        f'data-label-opacity="{label_opacity_attr}" '
        f'style="{html_style_joined}">'
        f"{fallback_html}</div>"
        + build_3d_vtk_js_script(element_id, vtk_js_url=vtk_js_url)
    )


def build_3d_vtk_js_script(element_id: str, vtk_js_url: Optional[str] = None) -> str:
    vtk_js_url = vtk_js_url or VTK_JS_CDN_URL
    return (
        "<script>"
        "(function(){"
        f"const container=document.getElementById('{element_id}');"
        "if(!container){return;}"
        "const init=function(){"
        "if(container.dataset.vtkInit){return;}"
        "container.dataset.vtkInit='1';"
        "const dims=container.dataset.dims.split(',').map(Number);"
        "const raw=atob(container.dataset.volume);"
        "const bytes=new Uint8Array(raw.length);"
        "for(let i=0;i<raw.length;i+=1){bytes[i]=raw.charCodeAt(i);}"
        "const labelRawB64=container.dataset.labelVolume||'';"
        "const vtk=window.vtk;"
        f"{_build_vtk_js_renderer_core(include_container_size=False)}"
        "};"
        "if(window.vtk){init();return;}"
        "if(!window._cytoVtkLoading){"
        "window._cytoVtkLoading=true;"
        "const script=document.createElement('script');"
        f"script.src='{vtk_js_url}';"
        "script.async=true;"
        "script.onload=function(){init();};"
        "document.head.appendChild(script);"
        "}else{"
        "const wait=setInterval(function(){"
        "if(window.vtk){clearInterval(wait);init();}"
        "},50);"
        "setTimeout(function(){clearInterval(wait);},10000);"
        "}"
        "})();"
        "</script>"
    )


def _build_vtk_js_renderer_core(*, include_container_size: bool) -> str:
    size_config = ""
    if include_container_size:
        size_config = (
            "const width=container.clientWidth||300;"
            "const height=container.clientHeight||300;"
            "openGL.setSize(width,height);"
        )
    return (
        "const fallback=container.querySelector('.cyto-3d-fallback');"
        "if(fallback){fallback.remove();}"
        "const imageData=vtk.Common.DataModel.vtkImageData.newInstance();"
        "imageData.setDimensions(dims);"
        "imageData.getPointData().setScalars("
        "vtk.Common.Core.vtkDataArray.newInstance({"
        "name:'Scalars',values:bytes,numberOfComponents:1"
        "})"
        ");"
        "const mapper=vtk.Rendering.Core.vtkVolumeMapper.newInstance();"
        "mapper.setInputData(imageData);"
        "const volume=vtk.Rendering.Core.vtkVolume.newInstance();"
        "volume.setMapper(mapper);"
        f"{VTK_JS_TRANSFER_FUNCTION_SNIPPET}"
        "volume.getProperty().setRGBTransferFunction(0,ctfun);"
        "volume.getProperty().setScalarOpacity(0,ofun);"
        "volume.getProperty().setShade(false);"
        "volume.getProperty().setInterpolationTypeToFastLinear();"
        "const renderer=vtk.Rendering.Core.vtkRenderer.newInstance({"
        "background:[1,1,1]"
        "});"
        "const renderWindow=vtk.Rendering.Core.vtkRenderWindow.newInstance();"
        "renderWindow.addRenderer(renderer);"
        "const openGL=vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();"
        "openGL.setContainer(container);"
        f"{size_config}"
        "renderWindow.addView(openGL);"
        "const interactor=vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();"
        "interactor.setView(openGL);"
        "interactor.initialize();"
        "interactor.bindEvents(container);"
        "const style=vtk.Interaction.Style.vtkInteractorStyleTrackballCamera"
        ".newInstance();"
        "interactor.setInteractorStyle(style);"
        "renderer.addVolume(volume);"
        "if(labelRawB64){"
        "const labelRaw=atob(labelRawB64);"
        "if(labelRaw.length===bytes.length){"
        "const labelBytes=new Uint8Array(labelRaw.length);"
        "for(let i=0;i<labelRaw.length;i+=1){labelBytes[i]=labelRaw.charCodeAt(i);}"
        "const labelData=vtk.Common.DataModel.vtkImageData.newInstance();"
        "labelData.setDimensions(dims);"
        "labelData.getPointData().setScalars("
        "vtk.Common.Core.vtkDataArray.newInstance({"
        "name:'LabelScalars',values:labelBytes,numberOfComponents:1"
        "})"
        ");"
        "const labelMapper=vtk.Rendering.Core.vtkVolumeMapper.newInstance();"
        "labelMapper.setInputData(labelData);"
        "if(labelMapper.setBlendModeToMaximumIntensity){"
        "labelMapper.setBlendModeToMaximumIntensity();"
        "}"
        "const labelVolume=vtk.Rendering.Core.vtkVolume.newInstance();"
        "labelVolume.setMapper(labelMapper);"
        "const labelColor=(container.dataset.labelColor||'0,1,0')"
        ".split(',').map(Number);"
        "const r=Math.min(1,Math.max(0,labelColor[0]||0));"
        "const g=Math.min(1,Math.max(0,labelColor[1]||1));"
        "const b=Math.min(1,Math.max(0,labelColor[2]||0));"
        "const labelOpacityRaw=Number(container.dataset.labelOpacity||0.95);"
        "const labelOpacity=Math.min(1,Math.max(0,labelOpacityRaw));"
        "const labelCtfun=vtk.Rendering.Core.vtkColorTransferFunction.newInstance();"
        "labelCtfun.addRGBPoint(0,0,0,0);"
        "labelCtfun.addRGBPoint(1,r,g,b);"
        "labelCtfun.addRGBPoint(255,r,g,b);"
        "const labelOfun=vtk.Common.DataModel.vtkPiecewiseFunction.newInstance();"
        "labelOfun.addPoint(0,0.0);"
        "labelOfun.addPoint(1,labelOpacity);"
        "labelOfun.addPoint(255,labelOpacity);"
        "labelVolume.getProperty().setRGBTransferFunction(0,labelCtfun);"
        "labelVolume.getProperty().setScalarOpacity(0,labelOfun);"
        "labelVolume.getProperty().setShade(false);"
        "labelVolume.getProperty().setInterpolationTypeToNearest();"
        "renderer.addVolume(labelVolume);"
        "}"
        "}"
        "renderer.resetCamera();"
        "renderWindow.render();"
    )


def build_3d_vtk_js_initializer(display_options: Optional[dict] = None) -> str:
    vtk_js_url = _resolve_vtk_js_url(display_options)
    return (
        "(function(){"
        "const init=function(container){"
        "if(!container||container.dataset.vtkInit){return;}"
        "container.dataset.vtkInit='1';"
        "const dims=container.dataset.dims.split(',').map(Number);"
        "const raw=atob(container.dataset.volume);"
        "const bytes=new Uint8Array(raw.length);"
        "for(let i=0;i<raw.length;i+=1){bytes[i]=raw.charCodeAt(i);}"
        "const labelRawB64=container.dataset.labelVolume||'';"
        "const vtk=window.vtk;"
        f"{_build_vtk_js_renderer_core(include_container_size=True)}"
        "};"
        "const initAll=function(){"
        "const containers=document.querySelectorAll("
        "'.cyto-3d-image[data-volume][data-dims]');"
        "containers.forEach(function(container){init(container);});"
        "};"
        "if(window.vtk){initAll();return;}"
        "if(!window._cytoVtkLoading){"
        "window._cytoVtkLoading=true;"
        "const script=document.createElement('script');"
        f"script.src='{vtk_js_url}';"
        "script.async=true;"
        "script.onload=function(){initAll();};"
        "document.head.appendChild(script);"
        "}else{"
        "const wait=setInterval(function(){"
        "if(window.vtk){clearInterval(wait);initAll();}"
        "},50);"
        "setTimeout(function(){clearInterval(wait);},10000);"
        "}"
        "})();"
    )


def extract_volume_from_ome_arrow(  # noqa: C901, PLR0912
    data_value: Any,
    ensure_uint8: Callable[[np.ndarray], np.ndarray],
    is_ome_arrow_value: Callable[[Any], bool],
    logger: Any,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int]]]:
    if not is_ome_arrow_value(data_value):
        return None

    try:
        pixels_meta = data_value.get("pixels_meta", {})
        size_x = int(pixels_meta.get("size_x") or 0)
        size_y = int(pixels_meta.get("size_y") or 0)
        size_z = int(pixels_meta.get("size_z") or 0)
        planes = data_value.get("planes")

        if size_x <= 0 or size_y <= 0 or size_z <= 1 or planes is None:
            return None

        if isinstance(planes, np.ndarray):
            plane_entries = planes.tolist()
        else:
            plane_entries = list(planes)
        if not plane_entries:
            return None

        base = size_x * size_y
        volume = None
        filled = 0

        for plane_idx, plane in enumerate(plane_entries):
            if not isinstance(plane, dict):
                continue
            c_val = int(plane.get("c") or 0)
            t_val = int(plane.get("t") or 0)
            if c_val != 0 or t_val != 0:
                continue
            z_val = plane.get("z")
            z_idx = int(z_val) if z_val is not None else plane_idx
            if z_idx < 0 or z_idx >= size_z:
                continue
            pixels = plane.get("pixels")
            if pixels is None:
                continue
            np_pixels = np.asarray(pixels)
            if np_pixels.size != base:
                continue
            if volume is None:
                volume = np.zeros((size_z, size_y, size_x), dtype=np_pixels.dtype)
            volume[z_idx] = np_pixels.reshape((size_y, size_x))
            filled += 1

        if filled == 0 or volume is None:
            return None

        volume = ensure_uint8(volume)
        return volume, (size_x, size_y, size_z)
    except Exception as exc:
        logger.debug("Unable to decode 3D OME-Arrow struct: %s", exc)
        return None


def build_3d_html_from_path(  # noqa: PLR0913
    data_value: str,
    candidate_path: pathlib.Path,
    display_options: Optional[dict],
    ensure_uint8: Callable[[np.ndarray], np.ndarray],
    is_ome_arrow_value: Callable[[Any], bool],
    logger: Any,
) -> Optional[str]:
    try:
        from ome_arrow import OMEArrow  # type: ignore
    except Exception:
        logger.debug("ome-arrow not available for 3D rendering.")
        return None

    try:
        ome_struct = OMEArrow(data=str(candidate_path)).data
        if hasattr(ome_struct, "as_py"):
            ome_struct = ome_struct.as_py()
    except Exception as exc:
        logger.debug("Failed to load OME-Arrow for 3D rendering: %s", exc)
        return None

    volume_data = extract_volume_from_ome_arrow(
        ome_struct, ensure_uint8, is_ome_arrow_value, logger
    )
    if volume_data is None:
        return None

    volume, dims = volume_data
    return build_3d_image_html_view(
        volume=volume,
        dims=dims,
        data_value=data_value,
        candidate_path=candidate_path,
        display_options=display_options,
    )
