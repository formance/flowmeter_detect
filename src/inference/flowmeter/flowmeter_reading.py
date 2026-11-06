"""Register a meter crop to a calibrated dial and interpolate its needle reading."""
import json
from pathlib import Path

import numpy as np


def angle_calibration(pivot, anchors):
    vectors = anchors[:, 1:] - pivot
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    radii = np.linalg.norm(vectors, axis=1)
    if np.any(radii < 1) or not np.all(np.diff(anchors[:, 0]) > 0):
        raise ValueError('Calibration needs increasing values at distinct points away from the pivot')
    if np.all(np.diff(angles) < 0):
        angles, anchors = angles[::-1], anchors[::-1]
    if not np.all(np.diff(angles) > 0) or angles[-1] - angles[0] >= np.pi:
        raise ValueError('Calibration angles must progress monotonically along a single scale arc')
    return angles, anchors[:, 0], float(np.median(radii))


def interpolate_needle(points, pivot, angles, values, scale_radius):
    vectors = points - pivot
    radii = np.linalg.norm(vectors, axis=1)
    # Exclude the hub and background far beyond the printed scale.
    shaft = (radii >= 0.35 * scale_radius) & (radii <= 1.2 * scale_radius)
    if shaft.sum() < 20:
        return None, 'needle_shaft_not_found'
    rays = np.arctan2(vectors[shaft, 1], vectors[shaft, 0])
    center = (angles[0] + angles[-1]) / 2
    rays = (rays - center + np.pi) % (2 * np.pi) - np.pi + center
    direction = float(np.median(rays))
    inliers = np.abs(rays - direction) <= np.deg2rad(3)
    if inliers.mean() < 0.75 or np.ptp(radii[shaft][inliers]) < 0.15 * scale_radius:
        return None, 'needle_direction_unreliable'
    direction = float(np.median(rays[inliers]))
    tolerance = np.deg2rad(0.75)
    if not angles[0] - tolerance <= direction <= angles[-1] + tolerance:
        return None, 'outside_calibrated_range'
    return round(float(np.interp(direction, angles, values)), 1), 'ok'


class DialCalibration:
    def __init__(self, path: Path, segment_scale, config=None):
        import cv2

        if config is None:
            config = json.loads(path.read_text(encoding='utf-8-sig'))
        self.unit = config['unit']
        reference_path = path.parent / config['reference']
        self.name = reference_path.name
        reference = cv2.imdecode(np.frombuffer(reference_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if reference is None:
            raise ValueError(f'Cannot read calibration image: {reference_path}')
        height, width = reference.shape[:2]
        self.size = (width, height)
        self.pivot = np.asarray(config['pivot'], dtype=np.float64)
        anchors = np.asarray(config['anchors'], dtype=np.float64)
        if self.pivot.shape != (2,) or anchors.ndim != 2 or anchors.shape[1] != 3 or len(anchors) < 2:
            raise ValueError('Calibration needs pivot [x,y] and anchors [[value,x,y], ...]')
        points = np.vstack((self.pivot, anchors[:, 1:]))
        if not np.isfinite(anchors).all() or not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= self.size):
            raise ValueError('Calibration coordinates must lie inside the reference crop')
        self.angles, self.values, self.radius = angle_calibration(self.pivot, anchors)

        self.sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02)
        self.keypoints, self.descriptors = self.sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
        if self.descriptors is None or len(self.descriptors) < 8:
            raise ValueError('Calibration reference has too few distinct image features')
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        if config.get('scale_mask'):
            mask_path = path.parent / config['scale_mask']
            mask = cv2.imdecode(np.frombuffer(mask_path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != reference.shape[:2] or not np.isin(mask, [0, 1, 255]).all():
                raise ValueError('Calibration scale_mask must be a binary PNG matching the reference crop')
            self.scale = mask > 0
        else:
            self.scale = segment_scale(reference)
        if not self.scale.any():
            raise ValueError('Calibration reference has an empty scale mask')
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.scale_band = cv2.dilate(self.scale.astype(np.uint8), self.kernel) > 0

    def align(self, crop):
        import cv2

        if min(crop.shape[:2]) < 32:
            return None, 0
        keypoints, descriptors = self.sift.detectAndCompute(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None)
        if descriptors is None:
            return None, 0
        matches = {}
        for pair in self.matcher.knnMatch(descriptors, self.descriptors, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                match = pair[0]
                if match.trainIdx not in matches or match.distance < matches[match.trainIdx].distance:
                    matches[match.trainIdx] = match
        if len(matches) < 8:
            return None, 0
        source = np.float32([keypoints[m.queryIdx].pt for m in matches.values()])
        target = np.float32([self.keypoints[m.trainIdx].pt for m in matches.values()])
        matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            return None, 0
        inliers = mask.ravel().astype(bool)
        count = int(inliers.sum())
        if count < 8 or inliers.mean() < 0.45:
            return None, count
        hull = cv2.convexHull(target[inliers])
        if cv2.contourArea(hull) < 0.03 * self.size[0] * self.size[1]:
            return None, count
        return matrix, count

    def read(self, crop, needle, scale):
        import cv2

        result = {'reading': None, 'unit': self.unit, 'calibration_template': self.name}
        if not needle.any() or not scale.any():
            return {**result, 'reading_status': 'pointer_missing' if not needle.any() else 'scale_missing'}
        matrix, matches = self.align(crop)
        result['alignment_inliers'] = matches
        if matrix is None:
            return {**result, 'reading_status': 'alignment_failed'}
        aligned_scale = cv2.warpPerspective(scale.astype(np.uint8), matrix, self.size, flags=cv2.INTER_NEAREST)
        aligned_band = cv2.dilate(aligned_scale, self.kernel) > 0
        coverage = (self.scale & aligned_band).sum() / self.scale.sum()
        agreement = (self.scale_band & (aligned_scale > 0)).sum() / max(int(aligned_scale.sum()), 1)
        score = float(min(coverage, agreement))
        result['scale_alignment'] = round(score, 3)
        if score < 0.3:
            return {**result, 'reading_status': 'scale_mismatch'}
        y, x = np.nonzero(needle)
        points = np.column_stack((x, y)).astype(np.float32).reshape(-1, 1, 2)
        aligned = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
        value, status = interpolate_needle(aligned, self.pivot, self.angles, self.values, self.radius)
        return {**result, 'reading': value, 'reading_status': status}


def load_calibrations(path, segment_scale):
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    profiles = [config, *({'unit': config['unit'], **profile} for profile in config.get('fallbacks', []))]
    return [DialCalibration(path, segment_scale, profile) for profile in profiles]


def read_dial(calibrations, crop, needle, scale):
    candidates = [profile.read(crop, needle, scale) for profile in calibrations]

    def alignment_quality(result):
        agreement = result.get('scale_alignment', 0)
        inliers = result.get('alignment_inliers', 0)
        return agreement >= 0.3, agreement * inliers, inliers

    return max(candidates, key=alignment_quality)
