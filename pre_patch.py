#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DICOM CT reading + basic cleaning (rectal cancer RT pipeline friendly).

Features:
- Scan a folder, group CT slices by SeriesInstanceUID
- Pick the "best" CT series (most slices) or specify series uid
- Sort slices robustly (ImagePositionPatient along slice normal; fallback InstanceNumber)
- Consistency checks: geometry, orientation, spacing, missing/duplicate slices warnings
- Apply RescaleSlope/Intercept to produce HU volume
- Export: ct_hu.npy + ct_meta.json (+ optional ct_hu.nii.gz)

Usage:
  python dicom_ct_clean.py --dicom_dir /path/to/patient --out_dir ./out
  python dicom_ct_clean.py --dicom_dir /path --out_dir ./out --series_uid <UID>

Notes:
- Assumes single-frame CT DICOM slices (typical planning CT).
- Does not perform resampling/cropping (this file is "read & clean" only).
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


# -----------------------------
# Utilities
# -----------------------------

def is_ct_slice(ds: pydicom.dataset.FileDataset) -> bool:
    """Heuristic checks for CT image slice."""
    modality = getattr(ds, "Modality", None)
    if modality != "CT":
        return False
    # Must have basic geometry tags (PixelData check removed when using stop_before_pixels=True)
    for tag in ["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns"]:
        if not hasattr(ds, tag):
            return False
    return True


def safe_float_list(x) -> Optional[List[float]]:
    try:
        return [float(v) for v in x]
    except Exception:
        return None


def unit_normal_from_iop(iop: List[float]) -> np.ndarray:
    """
    DICOM IOP: first 3 = row direction cosines, next 3 = column direction cosines.
    Slice normal = row x col.
    """
    row = np.array(iop[:3], dtype=np.float64)
    col = np.array(iop[3:], dtype=np.float64)
    n = np.cross(row, col)
    norm = np.linalg.norm(n)
    if norm < 1e-8:
        return n
    return n / norm


def almost_equal(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class CTSliceInfo:
    path: str
    sop_instance_uid: str
    instance_number: Optional[int]
    image_position_patient: List[float]  # [x,y,z] in mm
    image_orientation_patient: List[float]  # 6 floats
    rows: int
    cols: int
    pixel_spacing: List[float]  # [row_spacing, col_spacing]
    slice_thickness: Optional[float]
    spacing_between_slices: Optional[float]
    rescale_slope: float
    rescale_intercept: float


@dataclass
class CTSeriesMeta:
    series_instance_uid: str
    study_instance_uid: Optional[str]
    frame_of_reference_uid: Optional[str]
    patient_id: Optional[str]
    patient_name: Optional[str]
    sop_class_uid: Optional[str]

    rows: int
    cols: int
    num_slices: int

    pixel_spacing_row_col: List[float]  # [row, col]
    slice_spacing: float  # inferred from positions (mm)
    slice_thickness: Optional[float]

    image_orientation_patient: List[float]  # 6 floats
    slice_normal: List[float]  # 3 floats (unit)

    origin_ipp_first_slice: List[float]  # [x,y,z] of first slice in sorted order
    positions_along_normal: List[float]  # projections (mm) for QA

    warnings: List[str]


# -----------------------------
# Core: scanning + reading
# -----------------------------

def scan_ct_slices(dicom_dir: str) -> Dict[str, List[CTSliceInfo]]:
    """
    Walk folder, read minimal headers, collect CT slices grouped by SeriesInstanceUID.
    """
    series_map: Dict[str, List[CTSliceInfo]] = {}

    for root, _, files in os.walk(dicom_dir):
        for fn in files:
            path = os.path.join(root, fn)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            except (InvalidDicomError, Exception):
                continue

            if not is_ct_slice(ds):
                continue

            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if not series_uid:
                continue

            ipp = safe_float_list(getattr(ds, "ImagePositionPatient", None))
            iop = safe_float_list(getattr(ds, "ImageOrientationPatient", None))
            ps = safe_float_list(getattr(ds, "PixelSpacing", None))
            if ipp is None or iop is None or ps is None:
                continue

            info = CTSliceInfo(
                path=path,
                sop_instance_uid=str(getattr(ds, "SOPInstanceUID", "")),
                instance_number=int(getattr(ds, "InstanceNumber", 0)) if hasattr(ds, "InstanceNumber") else None,
                image_position_patient=ipp,
                image_orientation_patient=iop,
                rows=int(getattr(ds, "Rows")),
                cols=int(getattr(ds, "Columns")),
                pixel_spacing=ps,
                slice_thickness=float(getattr(ds, "SliceThickness")) if hasattr(ds, "SliceThickness") else None,
                spacing_between_slices=float(getattr(ds, "SpacingBetweenSlices")) if hasattr(ds, "SpacingBetweenSlices") else None,
                rescale_slope=float(getattr(ds, "RescaleSlope", 1.0)),
                rescale_intercept=float(getattr(ds, "RescaleIntercept", 0.0)),
            )

            series_map.setdefault(series_uid, []).append(info)

    return series_map


def pick_best_series(series_map: Dict[str, List[CTSliceInfo]], series_uid: Optional[str]) -> Tuple[str, List[CTSliceInfo]]:
    if not series_map:
        raise RuntimeError("No CT slices found under dicom_dir.")

    if series_uid:
        if series_uid not in series_map:
            raise RuntimeError(f"Specified series_uid not found: {series_uid}")
        return series_uid, series_map[series_uid]

    # Default: pick series with most slices
    best_uid = max(series_map.keys(), key=lambda uid: len(series_map[uid]))
    return best_uid, series_map[best_uid]


def sort_slices(slices: List[CTSliceInfo]) -> Tuple[List[CTSliceInfo], np.ndarray]:
    """
    Robust sort using projection of IPP onto slice normal (derived from IOP).
    Fallback to InstanceNumber if needed.
    Returns sorted slices and positions_along_normal array.
    """
    # Use first slice orientation as reference
    ref_iop = slices[0].image_orientation_patient
    n = unit_normal_from_iop(ref_iop)

    # Compute projections
    projs = []
    for s in slices:
        ipp = np.array(s.image_position_patient, dtype=np.float64)
        projs.append(float(np.dot(ipp, n)))
    projs = np.array(projs, dtype=np.float64)

    # If projections are all identical (degenerate), fallback InstanceNumber
    if float(np.max(projs) - np.min(projs)) < 1e-6:
        # fallback
        slices_sorted = sorted(slices, key=lambda s: (s.instance_number is None, s.instance_number or 0, s.path))
        projs_sorted = np.array([0.0] * len(slices_sorted), dtype=np.float64)
        return slices_sorted, projs_sorted

    order = np.argsort(projs)
    slices_sorted = [slices[i] for i in order]
    projs_sorted = projs[order]
    return slices_sorted, projs_sorted


# -----------------------------
# QA checks
# -----------------------------

def check_consistency(sorted_slices: List[CTSliceInfo], projs: np.ndarray) -> List[str]:
    warnings: List[str] = []

    # Check constant rows/cols/pixel spacing/iop
    rows0, cols0 = sorted_slices[0].rows, sorted_slices[0].cols
    ps0 = sorted_slices[0].pixel_spacing
    iop0 = sorted_slices[0].image_orientation_patient

    for idx, s in enumerate(sorted_slices):
        if s.rows != rows0 or s.cols != cols0:
            warnings.append(f"Rows/Cols mismatch at slice {idx}: ({s.rows},{s.cols}) vs ({rows0},{cols0})")
        if any(not almost_equal(float(a), float(b), tol=1e-4) for a, b in zip(s.pixel_spacing, ps0)):
            warnings.append(f"PixelSpacing mismatch at slice {idx}: {s.pixel_spacing} vs {ps0}")
        if any(not almost_equal(float(a), float(b), tol=1e-4) for a, b in zip(s.image_orientation_patient, iop0)):
            warnings.append(f"IOP mismatch at slice {idx}: {s.image_orientation_patient} vs {iop0}")

    # Check duplicates (same SOPInstanceUID repeated) and projection duplicates
    sop_uids = [s.sop_instance_uid for s in sorted_slices]
    if len(set(sop_uids)) != len(sop_uids):
        warnings.append("Duplicate SOPInstanceUID detected (repeated slice files).")

    # Slice spacing inference and missing slice warning
    if len(projs) >= 2 and float(np.max(projs) - np.min(projs)) >= 1e-6:
        diffs = np.diff(projs)
        # Some scanners store descending order; after sort diffs should be >=0
        diffs_abs = np.abs(diffs)

        # Robust estimate spacing: median of diffs_abs ignoring zeros
        nonzero = diffs_abs[diffs_abs > 1e-6]
        if len(nonzero) == 0:
            warnings.append("All slice position projections are identical or nearly identical; cannot infer slice spacing.")
        else:
            spacing = float(np.median(nonzero))
            # detect irregular gaps
            # if any diff deviates > 20% from median, warn
            for i, d in enumerate(nonzero):
                if d > 1.2 * spacing + 1e-6:
                    warnings.append(f"Potential missing slices: gap at sorted index {i}->{i+1}, gap={d:.4f}mm, median={spacing:.4f}mm")
                if d < 0.8 * spacing - 1e-6:
                    warnings.append(f"Irregular slice spacing: small gap at {i}->{i+1}, gap={d:.4f}mm, median={spacing:.4f}mm")
    else:
        warnings.append("Cannot infer slice spacing from projections (insufficient variation).")

    return warnings


# -----------------------------
# Volume building
# -----------------------------

def read_pixel_array_hu(ds: pydicom.dataset.FileDataset, slope: float, intercept: float) -> np.ndarray:
    """
    Read pixel_array and convert to HU.
    Handles signed/unsigned representation via pydicom.
    """
    arr = ds.pixel_array.astype(np.float32)
    # Apply rescale
    hu = arr * float(slope) + float(intercept)
    return hu


def load_ct_volume_hu(sorted_slices: List[CTSliceInfo]) -> np.ndarray:
    """
    Load full pixel data for each slice, convert to HU, stack into (Z, Y, X).
    """
    vol = []
    for s in sorted_slices:
        ds = pydicom.dcmread(s.path, force=True)
        hu = read_pixel_array_hu(ds, s.rescale_slope, s.rescale_intercept)
        if hu.shape != (s.rows, s.cols):
            raise RuntimeError(f"Pixel array shape mismatch in {s.path}: {hu.shape} vs ({s.rows},{s.cols})")
        vol.append(hu)
    vol = np.stack(vol, axis=0)  # (Z, Y, X)
    return vol


def infer_slice_spacing_mm(projs: np.ndarray) -> float:
    if len(projs) < 2:
        return float("nan")
    diffs = np.abs(np.diff(projs))
    nonzero = diffs[diffs > 1e-6]
    if len(nonzero) == 0:
        return float("nan")
    return float(np.median(nonzero))


def read_series_level_meta(any_slice_path: str) -> Dict[str, Optional[str]]:
    ds = pydicom.dcmread(any_slice_path, stop_before_pixels=True, force=True)
    return {
        "study_instance_uid": str(getattr(ds, "StudyInstanceUID", None)) if hasattr(ds, "StudyInstanceUID") else None,
        "frame_of_reference_uid": str(getattr(ds, "FrameOfReferenceUID", None)) if hasattr(ds, "FrameOfReferenceUID") else None,
        "patient_id": str(getattr(ds, "PatientID", None)) if hasattr(ds, "PatientID") else None,
        "patient_name": str(getattr(ds, "PatientName", None)) if hasattr(ds, "PatientName") else None,
        "sop_class_uid": str(getattr(ds, "SOPClassUID", None)) if hasattr(ds, "SOPClassUID") else None,
    }


def export_nifti_if_available(vol_zyx: np.ndarray,
                             meta: CTSeriesMeta,
                             out_path: str) -> None:
    """
    Optional NIfTI export if nibabel installed.
    Note: NIfTI uses RAS by convention; DICOM is LPS.
    Here we export with a simple affine that treats axes as:
      X = columns, Y = rows, Z = slices
    If you need strict DICOM->NIfTI world mapping, do it in later "geometry harmonization" step.
    """
    try:
        import nibabel as nib
    except Exception:
        return

    # spacing
    sy, sx = meta.pixel_spacing_row_col  # row, col
    sz = meta.slice_spacing
    if not np.isfinite(sz) or sz <= 0:
        sz = meta.slice_thickness or 1.0

    # Basic affine (not full DICOM direction-cosine affine)
    # Users typically replace this with direction-aware affine later.
    affine = np.array([
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1],
    ], dtype=np.float64)

    img = nib.Nifti1Image(np.transpose(vol_zyx, (2, 1, 0)), affine)  # to (X,Y,Z)
    nib.save(img, out_path)



# -----------------------------
# RTSTRUCT -> Masks (PTV / BODY)
# -----------------------------

def is_rtstruct(ds: pydicom.dataset.FileDataset) -> bool:
    return getattr(ds, "Modality", None) == "RTSTRUCT" and hasattr(ds, "StructureSetROISequence")

def find_rtstruct_files(dicom_dir: str) -> List[str]:
    rt_paths: List[str] = []
    for root, _, files in os.walk(dicom_dir):
        for fn in files:
            path = os.path.join(root, fn)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if is_rtstruct(ds):
                rt_paths.append(path)
    return rt_paths

def choose_rtstruct(rt_paths: List[str], ct_for_uid: Optional[str]) -> Optional[str]:
    """Pick RTSTRUCT matching CT FrameOfReferenceUID if possible."""
    if not rt_paths:
        return None
    if ct_for_uid:
        for p in rt_paths:
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            except Exception:
                continue
            for_uid = str(getattr(ds, "FrameOfReferenceUID", "")) if hasattr(ds, "FrameOfReferenceUID") else ""
            if for_uid and for_uid == ct_for_uid:
                return p
    # fallback: first
    return rt_paths[0]

def _compile_name_regex(pat: str) -> re.Pattern:
    try:
        return re.compile(pat, flags=re.IGNORECASE)
    except re.error:
        # fallback to literal
        return re.compile(re.escape(pat), flags=re.IGNORECASE)

def extract_roi_contours(rtstruct_ds: pydicom.dataset.FileDataset, name_regex: re.Pattern) -> List[np.ndarray]:
    """Return list of contours, each contour is (N,3) in patient coords (mm)."""
    # Map ROINumber -> ROIName
    roi_num_to_name: Dict[int, str] = {}
    for roi in getattr(rtstruct_ds, "StructureSetROISequence", []):
        try:
            num = int(roi.ROINumber)
            name = str(roi.ROIName)
            roi_num_to_name[num] = name
        except Exception:
            continue

    # Find matching ROI numbers
    matched_nums = {num for num, nm in roi_num_to_name.items() if name_regex.search(nm or "")}
    if not matched_nums:
        return []

    contours: List[np.ndarray] = []
    for roi_contour in getattr(rtstruct_ds, "ROIContourSequence", []):
        try:
            ref_num = int(roi_contour.ReferencedROINumber)
        except Exception:
            continue
        if ref_num not in matched_nums:
            continue

        for c in getattr(roi_contour, "ContourSequence", []):
            data = getattr(c, "ContourData", None)
            if not data:
                continue
            pts = np.array([float(x) for x in data], dtype=np.float64).reshape(-1, 3)
            if pts.shape[0] >= 3:
                contours.append(pts)
    return contours

def rasterize_contours_to_mask(
    contours_xyz: List[np.ndarray],
    ct_slices_sorted: List[CTSliceInfo],
    vol_shape_zyx: Tuple[int, int, int],
    fill_value: int = 1,
) -> np.ndarray:
    """Rasterize 3D RTSTRUCT contours onto CT grid -> mask (Z,Y,X) uint8."""
    from PIL import Image, ImageDraw

    Z, Y, X = vol_shape_zyx
    mask = np.zeros((Z, Y, X), dtype=np.uint8)

    # Reference orientation & spacings
    iop = ct_slices_sorted[0].image_orientation_patient
    row_dir = np.array(iop[:3], dtype=np.float64)
    col_dir = np.array(iop[3:], dtype=np.float64)
    row_spacing = float(ct_slices_sorted[0].pixel_spacing[0])
    col_spacing = float(ct_slices_sorted[0].pixel_spacing[1])
    n = unit_normal_from_iop(iop)

    # Slice projection positions
    slice_projs = []
    slice_ipps = []
    for s in ct_slices_sorted:
        ipp = np.array(s.image_position_patient, dtype=np.float64)
        slice_ipps.append(ipp)
        slice_projs.append(float(np.dot(ipp, n)))
    slice_projs = np.array(slice_projs, dtype=np.float64)

    # Tolerance for assigning contour to slice
    # Use median spacing as tolerance; fallback to thickness
    diffs = np.abs(np.diff(np.sort(slice_projs)))
    nonzero = diffs[diffs > 1e-6]
    tol = float(np.median(nonzero)) / 2.0 if len(nonzero) else float(ct_slices_sorted[0].slice_thickness or 3.0) / 2.0
    tol = max(tol, 0.5)  # at least 0.5mm

    for pts in contours_xyz:
        # Determine closest slice index by projection
        proj = float(np.dot(pts.mean(axis=0), n))
        k = int(np.argmin(np.abs(slice_projs - proj)))
        if abs(slice_projs[k] - proj) > tol:
            # contour plane does not match any slice well; skip
            continue

        ipp = slice_ipps[k]
        # Convert to (row, col) in pixel coordinates
        v = pts - ipp[None, :]
        rows = np.dot(v, row_dir) / row_spacing
        cols = np.dot(v, col_dir) / col_spacing
        # NOTE: DICOM IOP defines iop[:3] as the direction of +col and iop[3:] as +row.
        # We therefore project to (row,col) using (row_dir=iop[3:], col_dir=iop[:3]).

        poly = [(float(c), float(r)) for r, c in zip(rows, cols)]
        # Rasterize with PIL on (X,Y) image where x=col, y=row
        img = Image.new("L", (X, Y), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(poly, outline=fill_value, fill=fill_value)
        mask[k] |= (np.array(img, dtype=np.uint8) > 0).astype(np.uint8)

    return mask

def generate_ptv_and_body_masks(
    dicom_dir: str,
    out_dir: str,
    ct_vol_hu_zyx: np.ndarray,
    ct_meta: CTSeriesMeta,
    ct_slices_sorted: List[CTSliceInfo],
    ptv_name_regex: str = r"\bPTV\b|PTV\d*",
    body_name_regex: str = r"\bBODY\b|\bEXTERNAL\b|\bBODYCONTOUR\b|\bEXTERN\b",
    body_hu_threshold: int = -400,
    body_keep_largest_cc: bool = True,
    body_bottom_crop_mm: float = 60.0,
    body_erode_iters: int = 2,
    body_dilate_iters: int = 2,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Try RTSTRUCT-based masks first; fallback BODY to HU threshold if needed."""
    notes: List[str] = []

    rt_paths = find_rtstruct_files(dicom_dir)
    rt_path = choose_rtstruct(rt_paths, ct_meta.frame_of_reference_uid)

    ptv_mask = None
    body_mask = None

    if rt_path:
        try:
            rt = pydicom.dcmread(rt_path, force=True)
            notes.append(f"RTSTRUCT used: {rt_path}")
            ptv_re = _compile_name_regex(ptv_name_regex)
            body_re = _compile_name_regex(body_name_regex)

            ptv_contours = extract_roi_contours(rt, ptv_re)
            body_contours = extract_roi_contours(rt, body_re)

            if ptv_contours:
                ptv_mask = rasterize_contours_to_mask(ptv_contours, ct_slices_sorted, ct_vol_hu_zyx.shape, fill_value=1)
                notes.append(f"PTV contours found: {len(ptv_contours)}")
            else:
                notes.append("PTV ROI not found in RTSTRUCT (by name regex).")

            if body_contours:
                body_mask = rasterize_contours_to_mask(body_contours, ct_slices_sorted, ct_vol_hu_zyx.shape, fill_value=1)
                notes.append(f"BODY/EXTERNAL contours found: {len(body_contours)}")
            else:
                notes.append("BODY/EXTERNAL ROI not found in RTSTRUCT (by name regex).")
        except Exception as e:
            notes.append(f"Failed to parse RTSTRUCT: {e}")

    else:
        notes.append("No RTSTRUCT found under dicom_dir.")
    # Fallback for BODY: HU threshold segmentation (remove couch + fill holes)
    # Why couch remains: couch often touches patient, making "largest CC" include both.
    # Strategy:
    #   1) HU threshold -> candidate
    #   2) optionally mask out bottom rows (where couch sits)
    #   3) (optional) erode to break thin connections, keep largest CC, then dilate back
    #   4) fill holes slice-wise
    if body_mask is None:
        vol = ct_vol_hu_zyx
        cand = vol > int(body_hu_threshold)

        try:
            from scipy import ndimage as ndi

            # --- Step A: remove bottom band (couch usually appears at bottom of axial slice) ---
            # Works even when couch touches patient.
            # Convert mm -> pixels using row spacing.
            sy = float(ct_meta.pixel_spacing_row_col[0]) if ct_meta and ct_meta.pixel_spacing_row_col else 1.0
            bottom_crop_px = int(round(float(body_bottom_crop_mm) / max(sy, 1e-6)))
            if bottom_crop_px > 0 and bottom_crop_px < cand.shape[1]:
                cand[:, -bottom_crop_px:, :] = False

            # --- Step B: morphology to disconnect couch from body ---
            if body_erode_iters > 0:
                cand2 = ndi.binary_erosion(cand, iterations=int(body_erode_iters))
            else:
                cand2 = cand

            if body_keep_largest_cc:
                lab, nlab = ndi.label(cand2)
                if nlab > 0:
                    counts = np.bincount(lab.ravel())
                    counts[0] = 0
                    keep = counts.argmax()
                    body_mask = (lab == keep)
                else:
                    body_mask = cand2
            else:
                body_mask = cand2

            if body_dilate_iters > 0:
                body_mask = ndi.binary_dilation(body_mask, iterations=int(body_dilate_iters))

            # --- Step C: fill holes slice-wise (helps lung/bowel gas) ---
            body_mask = body_mask.astype(bool)
            for k in range(body_mask.shape[0]):
                body_mask[k] = ndi.binary_fill_holes(body_mask[k])

            body_mask = body_mask.astype(np.uint8)
            notes.append(
                f"BODY mask generated by HU fallback (thr={int(body_hu_threshold)}; bottom_crop_mm={float(body_bottom_crop_mm)}; "
                f"erode={int(body_erode_iters)}; dilate={int(body_dilate_iters)}; largest_cc={bool(body_keep_largest_cc)})."
            )
        except Exception:
            # If scipy is unavailable, we at least do threshold + optional bottom crop
            sy = float(ct_meta.pixel_spacing_row_col[0]) if ct_meta and ct_meta.pixel_spacing_row_col else 1.0
            bottom_crop_px = int(round(float(body_bottom_crop_mm) / max(sy, 1e-6)))
            if bottom_crop_px > 0 and bottom_crop_px < cand.shape[1]:
                cand[:, -bottom_crop_px:, :] = False
            body_mask = cand.astype(np.uint8)
            notes.append(
                f"BODY mask generated by HU fallback (thr={int(body_hu_threshold)}; bottom_crop_mm={float(body_bottom_crop_mm)}; scipy not available)."
            )

    # Save
    if ptv_mask is not None:
        np.save(os.path.join(out_dir, "ptv_mask.npy"), ptv_mask.astype(np.uint8))
    if body_mask is not None:
        np.save(os.path.join(out_dir, "body_mask.npy"), body_mask.astype(np.uint8))

    return ptv_mask, body_mask, notes



# -----------------------------
# RTDOSE: load + resample to CT grid
# -----------------------------
def find_rtdose_files(dicom_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(dicom_dir):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if getattr(ds, "Modality", None) == "RTDOSE" and hasattr(ds, "SOPInstanceUID"):
                paths.append(p)
    return paths

def choose_rtdose(paths: List[str], frame_of_reference_uid: Optional[str]) -> Optional[str]:
    if not paths:
        return None
    if frame_of_reference_uid:
        for p in paths:
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                if str(getattr(ds, "FrameOfReferenceUID", "")) == str(frame_of_reference_uid):
                    return p
            except Exception:
                continue
    return paths[0]

def load_rtdose_dicom(dose_path: str) -> Tuple[np.ndarray, dict]:
    """
    Returns:
      dose_arr: float32 [K, Y, X] in Gy (best-effort conversion using DoseGridScaling and DoseUnits)
      info: dict with geometry fields (ipp0, iop, pixel_spacing, offsets_mm)
    """
    ds = pydicom.dcmread(dose_path, force=True)
    if "PixelData" not in ds:
        raise ValueError(f"RTDOSE has no PixelData: {dose_path}")

    scaling = float(getattr(ds, "DoseGridScaling", 1.0))
    arr = ds.pixel_array.astype(np.float32) * scaling  # usually (frames, rows, cols)

    units = str(getattr(ds, "DoseUnits", "") or "").upper()
    # DICOM DoseUnits can be "GY" or "CGY" (sometimes absent)
    if units == "CGY":
        arr = arr / 100.0

    iop = safe_float_list(getattr(ds, "ImageOrientationPatient", None))
    ipp0 = safe_float_list(getattr(ds, "ImagePositionPatient", None))
    ps = safe_float_list(getattr(ds, "PixelSpacing", None))
    offsets = safe_float_list(getattr(ds, "GridFrameOffsetVector", None))

    if iop is None or ipp0 is None or ps is None or offsets is None:
        raise ValueError(f"Missing geometry tags in RTDOSE: {dose_path}")

    info = {
        "iop": [float(x) for x in iop],
        "ipp0": [float(x) for x in ipp0],
        "pixel_spacing_row_col": [float(ps[0]), float(ps[1])],
        "offsets_mm": [float(x) for x in offsets],
        "dose_units": units or None,
        "dose_grid_scaling": float(scaling),
        "path": dose_path,
    }
    return arr, info

def resample_rtdose_to_ct_grid(
    dose_kyx: np.ndarray,
    dose_info: dict,
    ct_meta: CTSeriesMeta,
    ct_slices_sorted: List[CTSliceInfo],
    out_shape_zyx: Tuple[int, int, int],
) -> np.ndarray:
    """
    Resample RTDOSE onto CT voxel centers using trilinear interpolation.
    Assumptions:
      - CT grid is orthogonal (true for typical planning CT)
      - Uses DICOM direction cosines for both RTDOSE and CT.
    """
    try:
        from scipy import ndimage as ndi
    except Exception as e:
        raise RuntimeError("scipy is required for RTDOSE resampling (scipy.ndimage.map_coordinates).") from e

    K, Yd, Xd = dose_kyx.shape
    Z, Yc, Xc = out_shape_zyx
    if Yc <= 0 or Xc <= 0 or Z <= 0:
        raise ValueError(f"Invalid CT shape: {out_shape_zyx}")

    # Dose basis
    iop_d = dose_info["iop"]
    rd = np.array(iop_d[:3], dtype=np.float64)
    cd = np.array(iop_d[3:], dtype=np.float64)
    nd = unit_normal_from_iop(iop_d)
    ipp0_d = np.array(dose_info["ipp0"], dtype=np.float64)
    syd, sxd = dose_info["pixel_spacing_row_col"]  # row, col
    offsets = np.array(dose_info["offsets_mm"], dtype=np.float64)

    # Make offsets monotonic for interpolation
    order = np.argsort(offsets)
    offsets_sorted = offsets[order]
    dose_sorted = dose_kyx[order, :, :]

    # CT basis
    iop_c = ct_slices_sorted[0].image_orientation_patient
    # DICOM IOP: iop[:3] points along +col, iop[3:] points along +row.
    # Here we define directions corresponding to numpy indices:
    #   yy (row_mm) multiplies row_dir_c, xx (col_mm) multiplies col_dir_c.
    row_dir_c = np.array(iop_c[3:], dtype=np.float64)  # +row index
    col_dir_c = np.array(iop_c[:3], dtype=np.float64)  # +col index
    # We'll use per-slice IPP for z positions (more robust than assuming constant spacing)
    syc = float(ct_meta.pixel_spacing_row_col[0])
    sxc = float(ct_meta.pixel_spacing_row_col[1])

    # Precompute y/x grids once
    yy = (np.arange(Yc, dtype=np.float64) * syc).reshape(Yc, 1)  # (Y,1)
    xx = (np.arange(Xc, dtype=np.float64) * sxc).reshape(1, Xc)  # (1,X)

    # Dot products between CT in-plane dirs and dose dirs
    a_rr = float(np.dot(row_dir_c, rd))
    a_cr = float(np.dot(col_dir_c, rd))
    a_rc = float(np.dot(row_dir_c, cd))
    a_cc = float(np.dot(col_dir_c, cd))
    a_rn = float(np.dot(row_dir_c, nd))
    a_cn = float(np.dot(col_dir_c, nd))

    out = np.zeros((Z, Yc, Xc), dtype=np.float32)

    # Helper: map projection distance (mm along dose normal) -> continuous k index
    idx_axis = np.arange(len(offsets_sorted), dtype=np.float64)

    for z in range(Z):
        ipp_c = np.array(ct_slices_sorted[z].image_position_patient, dtype=np.float64)
        base = ipp_c - ipp0_d  # vector from dose origin to this CT slice origin

        # Components along dose basis at CT slice origin
        b_r = float(np.dot(base, rd))  # mm along dose row_dir
        b_c = float(np.dot(base, cd))  # mm along dose col_dir
        b_n = float(np.dot(base, nd))  # mm along dose normal

        # mm coordinates in dose row/col directions for each CT pixel
        r_mm = b_r + a_rr * yy + a_cr * xx   # (Y,X)
        c_mm = b_c + a_rc * yy + a_cc * xx   # (Y,X)
        n_mm = b_n + a_rn * yy + a_cn * xx   # (Y,X)

        # Convert to dose index space
        r_idx = r_mm / float(syd)
        c_idx = c_mm / float(sxd)

        # Map n_mm to k index via 1D interpolation on offsets
        k_idx = np.interp(n_mm, offsets_sorted, idx_axis, left=np.nan, right=np.nan)

        # map_coordinates expects coords in order (k, y, x)
        coords = np.array([
            k_idx,
            r_idx,
            c_idx
        ], dtype=np.float64)

        # Replace NaNs with out-of-bounds values; use cval=0
        coords = np.where(np.isfinite(coords), coords, -1e6)

        slice_res = ndi.map_coordinates(
            dose_sorted,
            coords,
            order=1,            # trilinear
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32)
        out[z] = slice_res

    return out

def normalize_dose(
    dose_zyx: np.ndarray,
    body_mask_zyx: Optional[np.ndarray],
    mode: str = "none",
    rx_gy: Optional[float] = None,
    p99: float = 99.0,
    eps: float = 1e-6
) -> Tuple[np.ndarray, dict]:
    """
    mode:
      - 'none' : no normalization
      - 'dmax' : divide by max dose within body (if provided) else global max
      - 'p99'  : divide by percentile(p99) within body (if provided) else global
      - 'rx'   : divide by rx_gy (required)
    Returns normalized dose and a dict of stats.
    """

    d = dose_zyx.astype(np.float32)
    if body_mask_zyx is not None:
        mask = (body_mask_zyx > 0)
        vals = d[mask]
        if vals.size == 0:
            vals = d.ravel()
    else:
        vals = d.ravel()

    stats = {
        "mode": mode,
        "rx_gy": float(rx_gy) if rx_gy is not None else None,
        "p99": float(p99),
        "scale": 1.0,
        "dmax_body": float(np.max(vals)) if vals.size else float(np.max(d)),
    }

    mode_l = (mode or "none").lower()
    if mode_l == "none":
        return d, stats
    if mode_l == "rx":
        if rx_gy is None:
            raise ValueError("rx_gy is required when dose_norm_mode='rx'")
        scale = max(float(rx_gy), eps)
        stats["scale"] = scale
        return d / scale, stats
    if mode_l == "dmax":
        scale = max(float(np.max(vals)), eps)
        stats["scale"] = scale
        return d / scale, stats
    if mode_l == "p99":
        scale = max(float(np.percentile(vals, p99)), eps)
        stats["scale"] = scale
        return d / scale, stats
    raise ValueError(f"Unknown dose_norm_mode: {mode}")


# -----------------------------
# Optional: crop volumes to body bbox (remove large air/couch context)
# -----------------------------

def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int,int,int,int,int,int]]:
    """Return (z0,z1,y0,y1,x0,x1) with z1/y1/x1 as exclusive."""
    if mask is None or mask.size == 0:
        return None
    idx = np.where(mask > 0)
    if len(idx[0]) == 0:
        return None
    z0, z1 = int(idx[0].min()), int(idx[0].max()) + 1
    y0, y1 = int(idx[1].min()), int(idx[1].max()) + 1
    x0, x1 = int(idx[2].min()), int(idx[2].max()) + 1
    return z0, z1, y0, y1, x0, x1


def expand_bbox(b: Tuple[int,int,int,int,int,int], shape_zyx: Tuple[int,int,int],
                margin_zyx: Tuple[int,int,int]) -> Tuple[int,int,int,int,int,int]:
    z0,z1,y0,y1,x0,x1 = b
    mz,my,mx = margin_zyx
    z0 = max(0, z0 - mz); y0 = max(0, y0 - my); x0 = max(0, x0 - mx)
    z1 = min(shape_zyx[0], z1 + mz); y1 = min(shape_zyx[1], y1 + my); x1 = min(shape_zyx[2], x1 + mx)
    return z0,z1,y0,y1,x0,x1


def crop_zyx(arr: np.ndarray, b: Tuple[int,int,int,int,int,int]) -> np.ndarray:
    z0,z1,y0,y1,x0,x1 = b
    return arr[z0:z1, y0:y1, x0:x1]


def crop_outputs_to_body(out_dir: str,
                         meta: CTSeriesMeta,
                         crop_margin_mm: float = 10.0) -> List[str]:
    """Crop ct_hu/dose/masks in out_dir to body bbox (+margin). Updates ct_meta.json with crop info."""
    notes: List[str] = []
    body_path = os.path.join(out_dir, "body_mask.npy")
    ct_path = os.path.join(out_dir, "ct_hu.npy")
    if not (os.path.exists(body_path) and os.path.exists(ct_path)):
        return notes

    body = np.load(body_path)
    ct = np.load(ct_path)

    b = bbox_from_mask(body)
    if b is None:
        notes.append("Skip cropping: body_mask is empty.")
        return notes

    sy, sx = (meta.pixel_spacing_row_col if meta and meta.pixel_spacing_row_col else [1.0, 1.0])
    sz = float(meta.slice_spacing) if meta and np.isfinite(meta.slice_spacing) and meta.slice_spacing > 0 else float(meta.slice_thickness or 1.0)

    mz = int(math.ceil(float(crop_margin_mm) / max(sz, 1e-6)))
    my = int(math.ceil(float(crop_margin_mm) / max(float(sy), 1e-6)))
    mx = int(math.ceil(float(crop_margin_mm) / max(float(sx), 1e-6)))

    b2 = expand_bbox(b, ct.shape, (mz,my,mx))

    # crop and overwrite
    np.save(ct_path, crop_zyx(ct, b2).astype(np.float32))
    np.save(body_path, crop_zyx(body, b2).astype(np.uint8))

    # ptv
    ptv_path = os.path.join(out_dir, "ptv_mask.npy")
    if os.path.exists(ptv_path):
        ptv = np.load(ptv_path)
        if ptv.shape == body.shape:
            np.save(ptv_path, crop_zyx(ptv, b2).astype(np.uint8))

    # dose
    dose_path = os.path.join(out_dir, "dose.npy")
    if os.path.exists(dose_path):
        dose = np.load(dose_path)
        if dose.shape == body.shape:
            np.save(dose_path, crop_zyx(dose, b2).astype(np.float32))

    # update meta with crop info
    meta_path = os.path.join(out_dir, "ct_meta.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            md = json.load(f)
    except Exception:
        md = asdict(meta) if meta else {}
    md["crop_bbox_zyx"] = [int(v) for v in b2]  # z0,z1,y0,y1,x0,x1 (exclusive)
    md["original_shape_zyx"] = [int(v) for v in ct.shape]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(md, f, ensure_ascii=False, indent=2)

    notes.append(f"Cropped outputs to body bbox (+{float(crop_margin_mm)}mm). New shape={tuple(np.load(ct_path).shape)} bbox_zyx={b2}")
    return notes


# -----------------------------
# Main pipeline
# -----------------------------

def run_ct_read_and_clean(dicom_dir: str, out_dir: str, series_uid: Optional[str] = None) -> Tuple[np.ndarray, CTSeriesMeta, List[CTSliceInfo]]:
    mkdirp(out_dir)

    series_map = scan_ct_slices(dicom_dir)
    chosen_uid, slices = pick_best_series(series_map, series_uid)

    if len(slices) < 2:
        raise RuntimeError(f"CT series {chosen_uid} has too few slices: {len(slices)}")

    slices_sorted, projs_sorted = sort_slices(slices)
    warnings = check_consistency(slices_sorted, projs_sorted)

    slice_spacing = infer_slice_spacing_mm(projs_sorted)

    # Build meta
    series_level = read_series_level_meta(slices_sorted[0].path)
    n = unit_normal_from_iop(slices_sorted[0].image_orientation_patient)

    meta = CTSeriesMeta(
        series_instance_uid=chosen_uid,
        study_instance_uid=series_level["study_instance_uid"],
        frame_of_reference_uid=series_level["frame_of_reference_uid"],
        patient_id=series_level["patient_id"],
        patient_name=series_level["patient_name"],
        sop_class_uid=series_level["sop_class_uid"],
        rows=slices_sorted[0].rows,
        cols=slices_sorted[0].cols,
        num_slices=len(slices_sorted),
        pixel_spacing_row_col=[float(slices_sorted[0].pixel_spacing[0]), float(slices_sorted[0].pixel_spacing[1])],
        slice_spacing=float(slice_spacing) if np.isfinite(slice_spacing) else float("nan"),
        slice_thickness=slices_sorted[0].slice_thickness,
        image_orientation_patient=[float(x) for x in slices_sorted[0].image_orientation_patient],
        slice_normal=[float(x) for x in n.tolist()],
        origin_ipp_first_slice=[float(x) for x in slices_sorted[0].image_position_patient],
        positions_along_normal=[float(x) for x in projs_sorted.tolist()],
        warnings=warnings,
    )

    # Load volume HU
    vol_hu_zyx = load_ct_volume_hu(slices_sorted)

    # Save outputs
    np.save(os.path.join(out_dir, "ct_hu.npy"), vol_hu_zyx.astype(np.float32))
    with open(os.path.join(out_dir, "ct_meta.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    # Optional NIfTI
    export_nifti_if_available(vol_hu_zyx, meta, os.path.join(out_dir, "ct_hu.nii.gz"))

    return vol_hu_zyx, meta, slices_sorted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_dir", type=str, required=True, help="Root folder containing DICOM files (CT/RT...).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder.")
    ap.add_argument("--series_uid", type=str, default=None, help="Optional CT SeriesInstanceUID to force.")

    # ---- extra outputs to match downstream np.load(...) script ----
    ap.add_argument("--dose_npy", type=str, default=None, help="Path to dose.npy to copy into out_dir as dose.npy")
    ap.add_argument("--ptv_mask_npy", type=str, default=None, help="Path to ptv_mask.npy to copy into out_dir as ptv_mask.npy")
    ap.add_argument("--body_mask_npy", type=str, default=None, help="Path to body_mask.npy to copy into out_dir as body_mask.npy")

    # ---- RTDOSE DICOM -> dose.npy (resample to CT grid) ----
    ap.add_argument("--rtdose_dicom", type=str, default=None, help="Path to RTDOSE DICOM file. If not set, will try to auto-find under dicom_dir.")
    ap.add_argument("--dose_norm_mode", type=str, default="p99", choices=["none","dmax","p99","rx"], help="Dose normalization mode for saved dose.npy. Default: p99 (within body).")
    ap.add_argument("--rx_gy", type=float, default=None, help="Prescription dose in Gy (required if dose_norm_mode='rx').")
    ap.add_argument("--dose_norm_p99", type=float, default=99.0, help="Percentile for dose_norm_mode='p99'. Default: 99.")
    ap.add_argument("--zero_outside_body", action="store_true", help="Apply dose[body==0]=0 before normalization & saving (recommended).")
    ap.add_argument("--resample_target_spacing", type=str, default=None, help="Optional target spacing 'sz,sy,sx' in mm to resample CT/dose/masks after all generation. Example: 3.0,1.5,1.5")

    ap.add_argument("--ptv_name_regex", type=str, default=r"\bPTV\b|PTV\d*", help="Regex (case-insensitive) to match PTV ROI name in RTSTRUCT.")
    ap.add_argument("--body_name_regex", type=str, default=r"\bBODY\b|\bEXTERNAL\b|\bBODYCONTOUR\b|\bEXTERN\b", help="Regex (case-insensitive) to match BODY/EXTERNAL ROI name in RTSTRUCT.")

    # When no BODY/EXTERNAL ROI exists in RTSTRUCT, we create body_mask from CT HU.
    ap.add_argument("--body_hu_threshold", type=int, default=-400, help="HU threshold for BODY fallback segmentation. Common: -500~-250. Default: -400")
    ap.add_argument("--no_body_largest_cc", action="store_true", help="Disable keeping largest 3D connected component in BODY fallback (not recommended unless couch removal fails).")
    ap.add_argument("--body_bottom_crop_mm", type=float, default=60.0, help="In BODY HU-fallback, zero-out bottom band (mm) to remove couch. Set 0 to disable.")
    ap.add_argument("--body_erode_iters", type=int, default=2, help="In BODY HU-fallback, erosion iterations to break couch-body connection. 0 disables.")
    ap.add_argument("--body_dilate_iters", type=int, default=2, help="In BODY HU-fallback, dilation iterations after CC selection to restore body thickness.")
    ap.add_argument("--crop_to_body", action="store_true", help="After generating body_mask, crop ct_hu/dose/masks to body bbox (+margin).")
    ap.add_argument("--crop_margin_mm", type=float, default=10.0, help="Cropping margin in mm when --crop_to_body is set.")

    args = ap.parse_args()

    vol, meta, slices_sorted = run_ct_read_and_clean(args.dicom_dir, args.out_dir, args.series_uid)

    # ---- copy dose / masks so that downstream can do np.load(".../dose.npy") etc ----
    def _maybe_copy(src_path: str, dst_name: str) -> None:
        if not src_path:
            return
        if not os.path.exists(src_path):
            print(f"[WARN] {dst_name} not copied: file not found: {src_path}")
            return
        try:
            arr = np.load(src_path)
        except Exception as e:
            print(f"[WARN] {dst_name} not copied: cannot np.load({src_path}): {e}")
            return
        np.save(os.path.join(args.out_dir, dst_name), arr)
        print(f"[OK] saved {dst_name} -> {os.path.join(args.out_dir, dst_name)}  shape={getattr(arr,'shape',None)} dtype={getattr(arr,'dtype',None)}")

    # NOTE: dose.npy will be created from RTDOSE (or copied from --dose_npy) after masks are prepared.
    _maybe_copy(args.ptv_mask_npy, "ptv_mask.npy")
    _maybe_copy(args.body_mask_npy, "body_mask.npy")


    # ---- generate PTV/BODY masks if not provided ----
    need_ptv = args.ptv_mask_npy is None
    need_body = args.body_mask_npy is None
    if need_ptv or need_body:
        ptv_mask, body_mask, notes = generate_ptv_and_body_masks(
            dicom_dir=args.dicom_dir,
            out_dir=args.out_dir,
            ct_vol_hu_zyx=vol,
            ct_meta=meta,
            ct_slices_sorted=slices_sorted,
            ptv_name_regex=args.ptv_name_regex,
            body_name_regex=args.body_name_regex,
            body_hu_threshold=args.body_hu_threshold,
            body_keep_largest_cc=(not args.no_body_largest_cc),
            body_bottom_crop_mm=args.body_bottom_crop_mm,
            body_erode_iters=args.body_erode_iters,
            body_dilate_iters=args.body_dilate_iters,
        )
        for n in notes:
            print("[MASK]", n)
        if need_ptv and ptv_mask is None:
            print("[WARN] ptv_mask.npy was not generated (PTV ROI not found).")
        if need_body and body_mask is None:
            print("[WARN] body_mask.npy was not generated.")




    # ---- create dose.npy ----
    dose_out_path = os.path.join(args.out_dir, "dose.npy")
    dose_created = False

    if args.dose_npy is not None:
        _maybe_copy(args.dose_npy, "dose.npy")
        dose_created = os.path.exists(dose_out_path)
    else:
        # Load RTDOSE DICOM and resample to CT grid
        try:
            rtdose_path = args.rtdose_dicom
            if not rtdose_path:
                candidates = find_rtdose_files(args.dicom_dir)
                rtdose_path = choose_rtdose(candidates, meta.frame_of_reference_uid)
            if not rtdose_path:
                print("[WARN] No RTDOSE found. dose.npy will NOT be generated.")
            else:
                dose_kyx, dose_info = load_rtdose_dicom(rtdose_path)
                dose_ct = resample_rtdose_to_ct_grid(
                    dose_kyx=dose_kyx,
                    dose_info=dose_info,
                    ct_meta=meta,
                    ct_slices_sorted=slices_sorted,
                    out_shape_zyx=vol.shape,
                )

                # Apply body outside zero (recommended)
                body_path = os.path.join(args.out_dir, "body_mask.npy")
                body_mask = None
                if os.path.exists(body_path):
                    try:
                        body_mask = np.load(body_path)
                    except Exception:
                        body_mask = None

                if args.zero_outside_body and body_mask is not None and dose_ct.shape == body_mask.shape:
                    dose_ct = dose_ct.astype(np.float32)
                    dose_ct[body_mask == 0] = 0.0
                    print("[DOSE] Applied dose[body==0]=0")

                # Normalize
                try:
                    dose_norm, norm_stats = normalize_dose(
                        dose_ct,
                        body_mask_zyx=(body_mask if (body_mask is not None and body_mask.shape == dose_ct.shape) else None),
                        mode=args.dose_norm_mode,
                        rx_gy=args.rx_gy,
                        p99=args.dose_norm_p99,
                    )
                    dose_ct = dose_norm
                    print(f"[DOSE] Normalized dose with mode={args.dose_norm_mode} scale={norm_stats.get('scale')}")
                    # record into meta json (append)
                    try:
                        meta_path = os.path.join(args.out_dir, "ct_meta.json")
                        with open(meta_path, "r", encoding="utf-8") as f:
                            md = json.load(f)
                        md["dose_processing"] = {
                            "source": "RTDOSE",
                            "rtdose_path": dose_info.get("path"),
                            "dose_units": dose_info.get("dose_units"),
                            "dose_grid_scaling": dose_info.get("dose_grid_scaling"),
                            "dose_norm": norm_stats,
                            "zero_outside_body": bool(args.zero_outside_body),
                        }
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(md, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[WARN] Dose normalization failed: {e}")

                np.save(dose_out_path, dose_ct.astype(np.float32))
                dose_created = True
                print(f"[OK] saved dose.npy (CT-grid) -> {dose_out_path}  shape={dose_ct.shape} dtype={dose_ct.dtype}")
        except Exception as e:
            print(f"[WARN] Failed to generate dose.npy from RTDOSE: {e}")

    # ---- optional: resample to target spacing (simple axis-aligned zoom) ----
    if args.resample_target_spacing:
        try:
            from scipy import ndimage as ndi
            sz_t, sy_t, sx_t = [float(x) for x in args.resample_target_spacing.split(",")]
            sy, sx = meta.pixel_spacing_row_col
            sz = meta.slice_spacing if np.isfinite(meta.slice_spacing) and meta.slice_spacing > 0 else (meta.slice_thickness or 1.0)
            zoom_ct = (sz / sz_t, sy / sy_t, sx / sx_t)

            def _resample(path, order):
                if not os.path.exists(path):
                    return None
                a = np.load(path)
                return ndi.zoom(a, zoom=zoom_ct, order=order)

            ct_rs = _resample(os.path.join(args.out_dir, "ct_hu.npy"), order=1)
            if ct_rs is not None:
                np.save(os.path.join(args.out_dir, "ct_hu.npy"), ct_rs.astype(np.float32))

            body_rs = _resample(os.path.join(args.out_dir, "body_mask.npy"), order=0)
            if body_rs is not None:
                np.save(os.path.join(args.out_dir, "body_mask.npy"), (body_rs > 0.5).astype(np.uint8))

            ptv_rs = _resample(os.path.join(args.out_dir, "ptv_mask.npy"), order=0)
            if ptv_rs is not None:
                np.save(os.path.join(args.out_dir, "ptv_mask.npy"), (ptv_rs > 0.5).astype(np.uint8))

            dose_rs = _resample(os.path.join(args.out_dir, "dose.npy"), order=1)
            if dose_rs is not None:
                np.save(os.path.join(args.out_dir, "dose.npy"), dose_rs.astype(np.float32))

            # update meta spacing
            meta_path = os.path.join(args.out_dir, "ct_meta.json")
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    md = json.load(f)
                md["resample_target_spacing"] = {"sz": sz_t, "sy": sy_t, "sx": sx_t}
                md["pixel_spacing_row_col"] = [sy_t, sx_t]
                md["slice_spacing"] = sz_t
                md["warnings"] = md.get("warnings", []) + [f"Resampled volumes to target spacing sz,sy,sx={sz_t},{sy_t},{sx_t} using ndi.zoom (axis-aligned)."]
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(md, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # reload vol/meta for downstream crop bbox computations
            try:
                vol = np.load(os.path.join(args.out_dir, "ct_hu.npy"))
            except Exception:
                pass
            print(f"[OK] Resampled outputs to target spacing {args.resample_target_spacing}")
        except Exception as e:
            print(f"[WARN] Failed to resample to target spacing: {e}")

    # ---- optional crop to body bbox (remove couch/air context) ----
    if args.crop_to_body:
        crop_notes = crop_outputs_to_body(args.out_dir, meta, crop_margin_mm=args.crop_margin_mm)
        for n in crop_notes:
            print("[CROP]", n)
        # reload vol for reporting
        try:
            vol = np.load(os.path.join(args.out_dir, "ct_hu.npy"))
        except Exception:
            pass

    print("=== CT Read & Clean Done ===")
    print(f"SeriesInstanceUID: {meta.series_instance_uid}")
    print(f"Shape (Z,Y,X): {vol.shape}")
    print(f"PixelSpacing (row,col): {meta.pixel_spacing_row_col}")
    print(f"Slice spacing (inferred): {meta.slice_spacing}")
    if meta.warnings:
        print("\nWarnings:")
        for w in meta.warnings:
            print(" -", w)
    else:
        print("No warnings.")


if __name__ == "__main__":
    main()
