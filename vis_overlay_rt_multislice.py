#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import matplotlib.pyplot as plt

# (Optional) for GIF
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


# -------------------------
# I/O helpers
# -------------------------

def dcmread_header(path: str):
    return pydicom.dcmread(path, stop_before_pixels=True, force=True)

def dcmread_full(path: str):
    return pydicom.dcmread(path, force=True)

def safe_float_list(x) -> Optional[List[float]]:
    try:
        return [float(v) for v in x]
    except Exception:
        return None

def scan_dicom_paths(dicom_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(dicom_dir):
        for fn in files:
            out.append(os.path.join(root, fn))
    return out

def mkdirp(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def unit_normal_from_iop(iop6: List[float]) -> np.ndarray:
    # DICOM: iop[0:3]=col_dir, iop[3:6]=row_dir
    col_dir = np.array(iop6[0:3], dtype=np.float64)
    row_dir = np.array(iop6[3:6], dtype=np.float64)
    n = np.cross(col_dir, row_dir)
    norm = np.linalg.norm(n)
    return n if norm < 1e-8 else n / norm

def is_ct_header(ds) -> bool:
    if getattr(ds, "Modality", None) != "CT":
        return False
    required = ["SeriesInstanceUID", "SOPInstanceUID", "ImagePositionPatient",
                "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns"]
    return all(hasattr(ds, t) for t in required)

def apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Return display-ready image scaled 0..1."""
    low = center - width / 2.0
    high = center + width / 2.0
    x = np.clip(hu, low, high)
    x = (x - low) / max(high - low, 1e-6)
    return x


# -------------------------
# CT loading
# -------------------------

@dataclass
class CTSlice:
    path: str
    sop_uid: str
    ipp: np.ndarray
    iop: np.ndarray
    rows: int
    cols: int
    ps_row: float
    ps_col: float
    slope: float
    intercept: float

@dataclass
class CTVolume:
    vol_hu: np.ndarray            # (Z,Y,X)
    slices: List[CTSlice]
    series_uid: str
    frame_of_ref_uid: Optional[str]
    row_dir: np.ndarray           # (3,)
    col_dir: np.ndarray           # (3,)
    normal: np.ndarray            # (3,)
    slice_positions: np.ndarray   # (Z,)

def load_ct_volume(dicom_dir: str) -> CTVolume:
    paths = scan_dicom_paths(dicom_dir)
    series_map: Dict[str, List[CTSlice]] = {}
    for_uid_map: Dict[str, Optional[str]] = {}

    for p in paths:
        try:
            ds = dcmread_header(p)
        except Exception:
            continue
        if not is_ct_header(ds):
            continue

        series_uid = str(ds.SeriesInstanceUID)
        sop_uid = str(ds.SOPInstanceUID)
        ipp = np.array(safe_float_list(ds.ImagePositionPatient), dtype=np.float64)
        iop = np.array(safe_float_list(ds.ImageOrientationPatient), dtype=np.float64)
        ps = safe_float_list(ds.PixelSpacing)
        rows = int(ds.Rows)
        cols = int(ds.Columns)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        series_map.setdefault(series_uid, []).append(
            CTSlice(
                path=p, sop_uid=sop_uid, ipp=ipp, iop=iop,
                rows=rows, cols=cols,
                ps_row=float(ps[0]), ps_col=float(ps[1]),
                slope=slope, intercept=intercept
            )
        )
        if series_uid not in for_uid_map:
            for_uid_map[series_uid] = str(getattr(ds, "FrameOfReferenceUID", None)) if hasattr(ds, "FrameOfReferenceUID") else None

    if not series_map:
        raise RuntimeError("No CT slices found under dicom_dir.")

    series_uid = max(series_map.keys(), key=lambda k: len(series_map[k]))
    slices = series_map[series_uid]
    frame_of_ref = for_uid_map.get(series_uid, None)

    # dirs from first slice
    ref_iop = slices[0].iop.tolist()
    col_dir = np.array(ref_iop[0:3], dtype=np.float64)
    row_dir = np.array(ref_iop[3:6], dtype=np.float64)
    normal = unit_normal_from_iop(ref_iop)

    projs = np.array([float(np.dot(s.ipp, normal)) for s in slices], dtype=np.float64)
    order = np.argsort(projs)
    slices_sorted = [slices[i] for i in order]
    projs_sorted = projs[order]

    vol = []
    for s in slices_sorted:
        ds = dcmread_full(s.path)
        arr = ds.pixel_array.astype(np.float32)
        hu = arr * s.slope + s.intercept
        vol.append(hu)
    vol_hu = np.stack(vol, axis=0)
    return CTVolume(vol_hu, slices_sorted, series_uid, frame_of_ref, row_dir, col_dir, normal, projs_sorted)


# -------------------------
# RTSTRUCT parsing
# -------------------------

@dataclass
class ROIContours:
    roi_number: int
    roi_name: str
    contours_xyz: List[np.ndarray]

def load_rtstruct_contours(dicom_dir: str, roi_name: Optional[str] = None) -> Tuple[str, Optional[str], ROIContours]:
    paths = scan_dicom_paths(dicom_dir)
    rtstruct_files = []
    for p in paths:
        try:
            ds = dcmread_header(p)
        except Exception:
            continue
        if getattr(ds, "Modality", None) == "RTSTRUCT":
            rtstruct_files.append(p)
    if not rtstruct_files:
        raise RuntimeError("No RTSTRUCT found.")

    # choose most ROIs
    best_path, best_roi_count = None, -1
    for p in rtstruct_files:
        ds = dcmread_full(p)
        count = len(getattr(ds, "StructureSetROISequence", [])) if hasattr(ds, "StructureSetROISequence") else 0
        if count > best_roi_count:
            best_roi_count = count
            best_path = p
    assert best_path is not None

    ds = dcmread_full(best_path)
    for_uid = str(getattr(ds, "FrameOfReferenceUID", None)) if hasattr(ds, "FrameOfReferenceUID") else None

    roi_defs: Dict[int, str] = {}
    if hasattr(ds, "StructureSetROISequence"):
        for roi in ds.StructureSetROISequence:
            num = int(getattr(roi, "ROINumber", -1))
            name = str(getattr(roi, "ROIName", f"ROI_{num}"))
            roi_defs[num] = name

    if not hasattr(ds, "ROIContourSequence"):
        raise RuntimeError("RTSTRUCT missing ROIContourSequence.")

    roi_contours: List[ROIContours] = []
    for rc in ds.ROIContourSequence:
        rnum = int(getattr(rc, "ReferencedROINumber", -1))
        name = roi_defs.get(rnum, f"ROI_{rnum}")
        contour_seq = getattr(rc, "ContourSequence", None)
        if contour_seq is None:
            continue

        contours_xyz = []
        for c in contour_seq:
            cd = getattr(c, "ContourData", None)
            if cd is None or len(cd) < 9 or (len(cd) % 3 != 0):
                continue
            pts = np.array([float(x) for x in cd], dtype=np.float64).reshape((-1, 3))
            contours_xyz.append(pts)

        if contours_xyz:
            roi_contours.append(ROIContours(rnum, name, contours_xyz))

    if not roi_contours:
        raise RuntimeError("No valid contours in RTSTRUCT.")

    selected = None
    if roi_name:
        for r in roi_contours:
            if r.roi_name == roi_name:
                selected = r
                break
        if selected is None:
            low = roi_name.lower()
            for r in roi_contours:
                if low in r.roi_name.lower():
                    selected = r
                    break
    if selected is None:
        selected = roi_contours[0]

    return best_path, for_uid, selected


# -------------------------
# RTDOSE loading + sampling
# -------------------------

@dataclass
class DoseGrid:
    dose_gy: np.ndarray
    ipp: np.ndarray
    row_dir: np.ndarray
    col_dir: np.ndarray
    normal: np.ndarray
    ps_row: float
    ps_col: float
    offsets: np.ndarray

def load_rtdose(dicom_dir: str) -> Tuple[str, Optional[str], DoseGrid]:
    paths = scan_dicom_paths(dicom_dir)
    rtdose_files = []
    for p in paths:
        try:
            ds = dcmread_header(p)
        except Exception:
            continue
        if getattr(ds, "Modality", None) == "RTDOSE":
            rtdose_files.append(p)
    if not rtdose_files:
        raise RuntimeError("No RTDOSE found.")

    # choose largest voxel
    best_path, best_vox, best_ds = None, -1, None
    for p in rtdose_files:
        ds = dcmread_full(p)
        try:
            arr = np.asarray(ds.pixel_array)
        except Exception:
            continue
        vox = int(arr.size)
        if vox > best_vox:
            best_vox = vox
            best_path = p
            best_ds = ds
    assert best_path is not None and best_ds is not None
    ds = best_ds

    for_uid = str(getattr(ds, "FrameOfReferenceUID", None)) if hasattr(ds, "FrameOfReferenceUID") else None
    dgs = float(getattr(ds, "DoseGridScaling", np.nan)) if hasattr(ds, "DoseGridScaling") else np.nan
    if not np.isfinite(dgs) or dgs <= 0:
        raise RuntimeError("RTDOSE missing/invalid DoseGridScaling.")

    ps = safe_float_list(getattr(ds, "PixelSpacing", None))
    ipp = safe_float_list(getattr(ds, "ImagePositionPatient", None))
    iop = safe_float_list(getattr(ds, "ImageOrientationPatient", None))
    gfov = safe_float_list(getattr(ds, "GridFrameOffsetVector", None))
    if ps is None or ipp is None or iop is None or gfov is None:
        raise RuntimeError("RTDOSE missing required geometry tags.")

    ipp = np.array(ipp, dtype=np.float64)
    iop = np.array(iop, dtype=np.float64)
    col_dir = iop[0:3].astype(np.float64)
    row_dir = iop[3:6].astype(np.float64)
    normal = unit_normal_from_iop(iop.tolist())
    offsets = np.array(gfov, dtype=np.float64)

    arr = np.asarray(ds.pixel_array).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    dose_gy = arr * float(dgs)

    return best_path, for_uid, DoseGrid(dose_gy, ipp, row_dir, col_dir, normal, float(ps[0]), float(ps[1]), offsets)


def contour_to_ct_slice_and_pixels(ct: CTVolume, contour_xyz: np.ndarray) -> Tuple[int, np.ndarray]:
    projs = np.dot(contour_xyz, ct.normal)
    p_mean = float(np.mean(projs))
    z_idx = int(np.argmin(np.abs(ct.slice_positions - p_mean)))

    ipp = ct.slices[z_idx].ipp
    v = contour_xyz - ipp[None, :]

    r = np.dot(v, ct.row_dir) / ct.slices[z_idx].ps_row
    c = np.dot(v, ct.col_dir) / ct.slices[z_idx].ps_col
    pts_rc = np.stack([r, c], axis=1)
    return z_idx, pts_rc


def sample_dose_on_ct_slice(dose: DoseGrid, ct: CTVolume, z_idx: int) -> np.ndarray:
    ct_slice = ct.slices[z_idx]
    Y, X = ct_slice.rows, ct_slice.cols

    rr = np.arange(Y, dtype=np.float64)
    cc = np.arange(X, dtype=np.float64)
    R, C = np.meshgrid(rr, cc, indexing="ij")

    P = (ct_slice.ipp[None, None, :]
         + (R[..., None] * ct_slice.ps_row) * ct.row_dir[None, None, :]
         + (C[..., None] * ct_slice.ps_col) * ct.col_dir[None, None, :])

    v = P - dose.ipp[None, None, :]

    r_f = np.dot(v, dose.row_dir) / dose.ps_row
    c_f = np.dot(v, dose.col_dir) / dose.ps_col
    o_f = np.dot(v, dose.normal)

    offsets = dose.offsets
    frames = dose.dose_gy
    if offsets[0] > offsets[-1]:
        offsets = offsets[::-1]
        frames = frames[::-1, ...]

    K = frames.shape[0]
    o_clipped = np.clip(o_f, offsets[0], offsets[-1])
    k1 = np.searchsorted(offsets, o_clipped, side="right") - 1
    k1 = np.clip(k1, 0, K - 2)
    k2 = k1 + 1

    o1 = offsets[k1]
    o2 = offsets[k2]
    w = (o_clipped - o1) / np.maximum(o2 - o1, 1e-8)

    def bilinear(frame: np.ndarray, rf: np.ndarray, cf: np.ndarray) -> np.ndarray:
        Yd, Xd = frame.shape
        rf = np.clip(rf, 0, Yd - 1 - 1e-6)
        cf = np.clip(cf, 0, Xd - 1 - 1e-6)
        r0 = np.floor(rf).astype(np.int32)
        c0 = np.floor(cf).astype(np.int32)
        r1 = np.clip(r0 + 1, 0, Yd - 1)
        c1 = np.clip(c0 + 1, 0, Xd - 1)
        dr = rf - r0
        dc = cf - c0

        v00 = frame[r0, c0]
        v01 = frame[r0, c1]
        v10 = frame[r1, c0]
        v11 = frame[r1, c1]
        v0 = v00 * (1 - dc) + v01 * dc
        v1 = v10 * (1 - dc) + v11 * dc
        return v0 * (1 - dr) + v1 * dr

    out = np.zeros((Y, X), dtype=np.float32)
    for k in np.unique(k1):
        mask = (k1 == k)
        d1 = bilinear(frames[k], r_f[mask], c_f[mask])
        d2 = bilinear(frames[k + 1], r_f[mask], c_f[mask])
        ww = w[mask].astype(np.float32)
        out[mask] = d1 * (1 - ww) + d2 * ww
    return out


# -------------------------
# Multi-slice export
# -------------------------

def export_multislice(dicom_dir: str, out_dir: str, roi_name: Optional[str],
                      dose_alpha: float, window_center: float, window_width: float,
                      make_gif: bool, make_mp4: bool, fps: int,
                      margin_slices: int) -> None:
    mkdirp(out_dir)
    frames_dir = os.path.join(out_dir, "frames")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    mkdirp(frames_dir)

    ct = load_ct_volume(dicom_dir)
    _, _, roi = load_rtstruct_contours(dicom_dir, roi_name=roi_name)
    _, _, dose = load_rtdose(dicom_dir)

    # 1) 找到 ROI 覆盖的 slice 集合
    slice_to_contours: Dict[int, List[np.ndarray]] = {}
    for cont in roi.contours_xyz:
        z, pts_rc = contour_to_ct_slice_and_pixels(ct, cont)
        slice_to_contours.setdefault(z, []).append(pts_rc)

    covered = sorted(slice_to_contours.keys())
    if not covered:
        raise RuntimeError("Selected ROI has no contours mapped to CT slices.")

    # 2) 扩展上下 margin_slices 层（让视频更连贯）
    z_min = max(min(covered) - margin_slices, 0)
    z_max = min(max(covered) + margin_slices, ct.vol_hu.shape[0] - 1)
    z_list = list(range(z_min, z_max + 1))

    # 3) 预计算一个全局剂量显示 vmax（避免每帧颜色跳变）
    dose_samples = []
    for z in covered:
        d = sample_dose_on_ct_slice(dose, ct, z)
        if np.any(d > 0):
            dose_samples.append(np.percentile(d[d > 0], 99.0))
    global_vmax = float(np.median(dose_samples)) if dose_samples else 1.0
    if not np.isfinite(global_vmax) or global_vmax <= 0:
        global_vmax = 1.0

    # 4) 逐层输出 PNG
    png_paths = []
    for idx, z in enumerate(z_list):
        ct_slice = ct.vol_hu[z, :, :]
        ct_disp = apply_window(ct_slice, window_center, window_width)

        dose_slice = sample_dose_on_ct_slice(dose, ct, z)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(ct_disp, cmap="gray", interpolation="nearest")
        ax.imshow(dose_slice, cmap="jet", alpha=dose_alpha, interpolation="nearest", vmin=0, vmax=global_vmax)

        # contours only on slices that have them
        for pts_rc in slice_to_contours.get(z, []):
            ax.plot(pts_rc[:, 1], pts_rc[:, 0], linewidth=1.5)

        ax.set_title(f"ROI: {roi.roi_name} | z={z}")
        ax.axis("off")
        plt.tight_layout()

        out_png = os.path.join(frames_dir, f"frame_{idx:03d}_z{z:04d}.png")
        plt.savefig(out_png, dpi=160)
        plt.close(fig)

        png_paths.append(out_png)

    print(f"Saved {len(png_paths)} frames to: {frames_dir}")

    # 5) 合成 GIF
    if make_gif:
        if imageio is None:
            print("[WARN] imageio not installed, skip GIF. Install: pip install imageio")
        else:
            gif_path = os.path.join(out_dir, "overlay.gif")
            imgs = [imageio.imread(p) for p in png_paths]
            imageio.mimsave(gif_path, imgs, duration=1.0 / max(fps, 1))
            print("Saved GIF:", gif_path)

    # 6) 合成 MP4（需要 ffmpeg）
    if make_mp4:
        mp4_path = os.path.join(out_dir, "overlay.mp4")
        # use ffmpeg if available
        # pattern must match frame numbering (frame_%03d_*.png is not directly supported by ffmpeg)
        # so we also export a pure sequential set:
        seq_dir = os.path.join(out_dir, "frames_seq")
        if os.path.exists(seq_dir):
            shutil.rmtree(seq_dir)
        mkdirp(seq_dir)
        for i, p in enumerate(png_paths):
            shutil.copyfile(p, os.path.join(seq_dir, f"frame_{i:03d}.png"))

        cmd = f'ffmpeg -y -framerate {fps} -i "{seq_dir}/frame_%03d.png" -pix_fmt yuv420p "{mp4_path}"'
        ret = os.system(cmd)
        if ret != 0:
            print("[WARN] ffmpeg failed or not installed. You can install ffmpeg, or rely on PNG/GIF output.")
        else:
            print("Saved MP4:", mp4_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--roi_name", type=str, default=None)
    ap.add_argument("--dose_alpha", type=float, default=0.35)
    ap.add_argument("--window_center", type=float, default=40.0)
    ap.add_argument("--window_width", type=float, default=400.0)
    ap.add_argument("--make_gif", action="store_true")
    ap.add_argument("--make_mp4", action="store_true")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--margin_slices", type=int, default=3, help="Extra slices above/below ROI coverage for context.")
    args = ap.parse_args()

    export_multislice(
        dicom_dir=args.dicom_dir,
        out_dir=args.out_dir,
        roi_name=args.roi_name,
        dose_alpha=args.dose_alpha,
        window_center=args.window_center,
        window_width=args.window_width,
        make_gif=args.make_gif,
        make_mp4=args.make_mp4,
        fps=args.fps,
        margin_slices=args.margin_slices
    )


if __name__ == "__main__":
    main()

