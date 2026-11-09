"""Register a meter crop to a calibrated dial and interpolate its needle reading."""
import json
from pathlib import Path

import numpy as np

from src.common.image_preprocessing import enhance_meter_crop


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


def interpolate_needle(points, pivot, anchors, scale_radius, *, measured_zero_fraction=0.0):
    import cv2

    radii = np.linalg.norm(points - pivot, axis=1)
    shaft = points[(radii >= 0.35 * scale_radius) & (radii <= 1.2 * scale_radius)]
    if len(shaft) < 20:
        return None, 'needle_shaft_not_found'

    line = cv2.fitLine(shaft.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel().astype(np.float64)
    direction, origin = line[:2], line[2:]
    normal = np.array([-direction[1], direction[0]])
    width = max(2.0, 0.035 * scale_radius)
    inliers = np.abs((shaft - origin) @ normal) <= width
    if (inliers.mean() < 0.75 or np.ptp((shaft[inliers] - origin) @ direction) < 0.15 * scale_radius
            or abs((pivot - origin) @ normal) > 2 * width):
        return None, 'needle_direction_unreliable'

    # Start the ray at the closest point on the fitted line to the hub.
    origin += ((pivot - origin) @ direction) * direction
    if np.median((shaft[inliers] - origin) @ direction) < 0:
        direction = -direction

    # A scale segment crosses the ray where its signed distance to the line is zero.
    offsets = anchors[:, 1:] - origin
    sides = offsets @ normal
    along = offsets @ direction
    spans = sides[:-1] - sides[1:]
    fractions = np.divide(sides[:-1], spans, out=np.full_like(spans, np.nan), where=np.abs(spans) > 1e-8)
    distances = along[:-1] + fractions * np.diff(along)
    intersections = np.flatnonzero((fractions >= 0) & (fractions <= 1) & (distances > 0))
    values = anchors[:, 0]
    if len(intersections):
        index = intersections[np.argmin(distances[intersections])]
        fraction = fractions[index]
        if index == 0:
            fraction = max(0.0, (fraction - measured_zero_fraction) / (1.0 - measured_zero_fraction))
        value = values[index] + fraction * (values[index + 1] - values[index])
        return round(float(value), 1), 'ok'

    # Preserve the small endpoint allowance for masks near the ends of the scale.
    ends = np.array([0, len(anchors) - 1])
    near = ends[(along[ends] > 0) & (np.abs(sides[ends]) <= along[ends] * np.tan(np.deg2rad(0.75)))]
    if len(near):
        end = near[np.argmin(np.abs(sides[near]))]
        return float(values[end]), 'ok'

    # A ray just before the zero end points to the opposite side from the scale's sweep.
    sweep_at_zero = along[0] * sides[1] - sides[0] * along[1]
    if values[0] == 0 and along[0] > 0 and sides[0] * sweep_at_zero > 0:
        return 0.0, 'ok'
    return None, 'outside_calibrated_range'


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
        _, _, self.radius = angle_calibration(self.pivot, anchors)
        self.anchors = anchors
        self.measured_zero_fraction = config.get('measured_zero_fraction', 0.0)
        if (type(self.measured_zero_fraction) not in (int, float)
                or not 0 <= self.measured_zero_fraction < 1
                or (self.measured_zero_fraction > 0 and anchors[0, 0] != 0)):
            raise ValueError('measured_zero_fraction must be in [0, 1) on a scale starting at zero')

        self.sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02)
        self.keypoints, self.descriptors = self.sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
        if self.descriptors is None or len(self.descriptors) < 8:
            raise ValueError('Calibration reference has too few distinct image features')
        enhanced_reference = enhance_meter_crop(reference)
        self.enhanced_keypoints, self.enhanced_descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(enhanced_reference, cv2.COLOR_BGR2GRAY), None)
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

    def align(self, crop, *, max_side=None, enhanced=False):
        import cv2

        if min(crop.shape[:2]) < 32:
            return None, 0
        height, width = crop.shape[:2]
        if max_side is not None:
            factor = max_side / max(height, width)
            crop = cv2.resize(crop, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)
        scaling = np.diag([crop.shape[1] / width, crop.shape[0] / height, 1.0])
        target_keypoints = self.enhanced_keypoints if enhanced else self.keypoints
        target_descriptors = self.enhanced_descriptors if enhanced else self.descriptors
        keypoints, descriptors = self.sift.detectAndCompute(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None)
        if descriptors is None or target_descriptors is None or len(target_descriptors) < 2:
            return None, 0
        matches = {}
        for pair in self.matcher.knnMatch(descriptors, target_descriptors, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                match = pair[0]
                if match.trainIdx not in matches or match.distance < matches[match.trainIdx].distance:
                    matches[match.trainIdx] = match
        if len(matches) < 8:
            return None, 0
        source = np.float32([keypoints[m.queryIdx].pt for m in matches.values()])
        target = np.float32([target_keypoints[m.trainIdx].pt for m in matches.values()])
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
        return matrix @ scaling, count

    def read(self, crop, needle, scale, *, max_side=None, enhanced=False):
        import cv2

        result = {'reading': None, 'unit': self.unit, 'calibration_template': self.name}
        if not needle.any() or not scale.any():
            return {**result, 'reading_status': 'pointer_missing' if not needle.any() else 'scale_missing'}
        matrix, matches = self.align(crop, max_side=max_side, enhanced=enhanced)
        result['alignment_inliers'] = matches
        result['alignment_mode'] = ('contrast_' if enhanced else '') + (f'resize_{max_side}' if max_side else 'native')
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
        value, status = interpolate_needle(aligned, self.pivot, self.anchors, self.radius,
                                          measured_zero_fraction=self.measured_zero_fraction)
        return {**result, 'reading': value, 'reading_status': status}


def load_calibrations(path, segment_scale):
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    profiles = [config, *({'unit': config['unit'], **profile} for profile in config.get('fallbacks', []))]
    return [DialCalibration(path, segment_scale, profile) for profile in profiles]


def read_dial(calibrations, crop, needle, scale, *, enhanced=False):
    def alignment_quality(result):
        agreement = result.get('scale_alignment', 0)
        inliers = result.get('alignment_inliers', 0)
        return result['reading_status'] == 'ok', agreement >= 0.3, agreement * inliers, inliers

    candidates = []
    for max_side in ((320,) if enhanced else (None, 320)):
        current = [profile.read(crop, needle, scale, max_side=max_side, enhanced=enhanced)
                   for profile in calibrations]
        candidates.extend(current)
        if any(result['reading_status'] == 'ok' for result in current) or not needle.any() or not scale.any():
            break
    return max(candidates, key=alignment_quality)
