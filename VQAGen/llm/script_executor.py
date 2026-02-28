"""
Safe script executor for LLM-generated Python code.

Executes code in a restricted environment with access to common_utils
functions. Uses a persistent worker process pool to avoid re-importing
heavy libraries (open3d, torch, scipy) on every execution.
"""

import sys
import traceback
import multiprocessing
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def preprocess_code(code: str) -> str:
    """Remove import statements from LLM-generated code.

    The sandbox already provides numpy, math, and other modules
    in the namespace, so import statements are unnecessary.
    """
    lines = code.split('\n')
    processed_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            logger.debug(f"Removing import statement: {stripped}")
            continue
        processed_lines.append(line)
    return '\n'.join(processed_lines)


# Default execution timeout in seconds
DEFAULT_EXECUTION_TIMEOUT = 30

# Worker process globals (set by _init_worker)
_worker_np = None
_worker_math = None
_worker_o3d = None
_worker_cal_3d_bbox = None
_worker_sample_points = None


@dataclass
class ExecutionResult:
    """Result of script execution"""
    success: bool
    result: Any
    error_message: Optional[str] = None


def _init_worker():
    """Initialize worker process with heavy imports.

    This runs once per worker process creation. All heavy libraries
    are imported here and stored in module globals so subsequent
    task executions reuse them without re-importing.
    """
    global _worker_np, _worker_math, _worker_o3d
    global _worker_cal_3d_bbox, _worker_sample_points

    import numpy as np
    import math
    import open3d as o3d
    from utils.common_utils import (
        cal_3d_bbox_distance_between_categories,
        sample_points_in_oriented_bbox_uniform,
    )

    _worker_np = np
    _worker_math = math
    _worker_o3d = o3d
    _worker_cal_3d_bbox = cal_3d_bbox_distance_between_categories
    _worker_sample_points = sample_points_in_oriented_bbox_uniform

    logger.debug("[Worker] Initialized with heavy imports")


def _execute_in_pool_worker(code, scene_info, frame_info):
    """Execute code in a pre-initialized pool worker.

    Uses pre-imported modules from _init_worker() globals,
    avoiding the overhead of re-importing heavy libraries.

    Returns:
        dict with 'success', 'result', 'error' keys.
    """
    np = _worker_np
    math = _worker_math
    o3d = _worker_o3d
    cal_3d_bbox_distance_between_categories = _worker_cal_3d_bbox
    sample_points_in_oriented_bbox_uniform = _worker_sample_points

    try:
        # Allowed built-in functions
        ALLOWED_BUILTINS = {
            'abs', 'all', 'any', 'bool', 'dict', 'enumerate',
            'filter', 'float', 'int', 'isinstance', 'len', 'list',
            'map', 'max', 'min', 'print', 'range', 'round', 'set',
            'sorted', 'str', 'sum', 'tuple', 'type', 'zip',
            'True', 'False', 'None',
            # Exceptions (needed for assert, try/except, raise)
            'Exception', 'ValueError', 'TypeError', 'KeyError',
            'IndexError', 'AssertionError', 'ZeroDivisionError',
            'RuntimeError', 'AttributeError', 'StopIteration',
        }

        restricted_builtins = {}
        for name in ALLOWED_BUILTINS:
            if isinstance(__builtins__, dict):
                restricted_builtins[name] = __builtins__.get(name)
            else:
                restricted_builtins[name] = getattr(
                    __builtins__, name, None
                )

        # --- Helper functions ---

        def get_centroid(category, object_bboxes, index=0):
            bboxes = object_bboxes.get(category, [])
            if bboxes and index < len(bboxes):
                return np.array(bboxes[index]["centroid"])
            return None

        def get_all_centroids(category, object_bboxes):
            bboxes = object_bboxes.get(category, [])
            return [np.array(bbox["centroid"]) for bbox in bboxes]

        def euclidean_distance(pos1, pos2):
            return float(
                np.linalg.norm(np.array(pos1) - np.array(pos2))
            )

        def find_closest_object(
            reference_category, object_bboxes,
            exclude_categories=None
        ):
            exclude = set(exclude_categories or [])
            exclude.add(reference_category)
            ref_bboxes = object_bboxes.get(reference_category, [])
            if not ref_bboxes:
                return None
            ref_centroid = np.array(ref_bboxes[0]["centroid"])
            min_dist = float('inf')
            closest = None
            for category, bboxes in object_bboxes.items():
                if category in exclude:
                    continue
                for bbox in bboxes:
                    centroid = np.array(bbox["centroid"])
                    dist = np.linalg.norm(centroid - ref_centroid)
                    if dist < min_dist:
                        min_dist = dist
                        closest = category
            return closest

        def find_farthest_object(
            reference_category, object_bboxes,
            exclude_categories=None
        ):
            exclude = set(exclude_categories or [])
            exclude.add(reference_category)
            ref_bboxes = object_bboxes.get(reference_category, [])
            if not ref_bboxes:
                return None
            ref_centroid = np.array(ref_bboxes[0]["centroid"])
            max_dist = -1
            farthest = None
            for category, bboxes in object_bboxes.items():
                if category in exclude:
                    continue
                for bbox in bboxes:
                    centroid = np.array(bbox["centroid"])
                    dist = np.linalg.norm(centroid - ref_centroid)
                    if dist > max_dist:
                        max_dist = dist
                        farthest = category
            return farthest

        def count_objects_within_distance(
            reference_category, target_category,
            object_bboxes, threshold
        ):
            ref_bboxes = object_bboxes.get(reference_category, [])
            tgt_bboxes = object_bboxes.get(target_category, [])
            if not ref_bboxes:
                return 0
            ref_centroid = np.array(ref_bboxes[0]["centroid"])
            count = 0
            for bbox in tgt_bboxes:
                centroid = np.array(bbox["centroid"])
                dist = np.linalg.norm(centroid - ref_centroid)
                if dist <= threshold:
                    count += 1
            return count

        # --- Relative Direction Functions ---

        def _calculate_angle(v1, v2s):
            """Shared angle calculation for direction functions."""
            dot_products = (v1 * v2s).sum(axis=1)
            mag_v1 = np.linalg.norm(v1, axis=1)
            mag_v2s = np.linalg.norm(v2s, axis=1)
            cos_vals = np.clip(
                dot_products / (mag_v1 * mag_v2s), -1, 1
            )
            angles = np.arccos(cos_vals)
            crs_products = np.cross(v1, v2s)
            angles = np.where(
                crs_products >= 0., angles, 2 * math.pi - angles
            )
            return np.degrees(angles)

        def _get_direction_data(
            positioning_category, orienting_category,
            querying_category, object_bboxes
        ):
            """Extract positions and compute angles for direction.

            Returns (angles, querying_points) or None if data missing.
            """
            pos_bboxes = object_bboxes.get(positioning_category, [])
            ori_bboxes = object_bboxes.get(orienting_category, [])
            qry_bboxes = object_bboxes.get(querying_category, [])

            if not pos_bboxes or not ori_bboxes or not qry_bboxes:
                return None

            pos_obj = pos_bboxes[0]
            ori_obj = ori_bboxes[0]
            qry_obj = qry_bboxes[0]

            pos_pos = np.array([pos_obj['centroid']])
            ori_pos = np.array([ori_obj['centroid']])
            qry_pos = np.array([qry_obj['centroid']])

            qry_bbox = o3d.geometry.OrientedBoundingBox(
                center=np.array(
                    qry_obj["centroid"]
                ).astype(np.float64).reshape(3, 1),
                R=np.array(
                    qry_obj["normalizedAxes"]
                ).astype(np.float64).reshape(3, 3).T,
                extent=np.array(
                    qry_obj["axesLengths"]
                ).astype(np.float64).reshape(3, 1)
            )

            vertices = np.asarray(qry_bbox.get_box_points())
            qry_points = np.concatenate([qry_pos, vertices], axis=0)

            orienting_vec = ori_pos - pos_pos
            querying_vecs = qry_points - pos_pos

            angles = _calculate_angle(
                orienting_vec[:, :2], querying_vecs[:, :2]
            )
            return angles

        def get_rel_direction_quadrant(
            positioning_category, orienting_category,
            querying_category, object_bboxes
        ):
            """Relative direction in 4 quadrants.

            Returns: front-left/front-right/back-left/back-right
            or ambiguous.
            """
            angles = _get_direction_data(
                positioning_category, orienting_category,
                querying_category, object_bboxes
            )
            if angles is None:
                return "ambiguous"

            quadrant_of_centroid = (angles // 90)[0]
            quadrant_of_vertices = (angles // 90)[1:]

            if (quadrant_of_centroid != quadrant_of_vertices).sum() > 2:
                return "ambiguous"

            ambiguity_threshold = 10
            if (90 - (angles % 90)[0]) < ambiguity_threshold:
                return "ambiguous"

            if angles[0] >= 270:
                return "front-right"
            elif angles[0] >= 180:
                return "back-right"
            elif angles[0] >= 90:
                return "back-left"
            else:
                return "front-left"

        def get_rel_direction_lr(
            positioning_category, orienting_category,
            querying_category, object_bboxes
        ):
            """Relative direction: left or right.

            Returns: left, right, or ambiguous.
            """
            angles = _get_direction_data(
                positioning_category, orienting_category,
                querying_category, object_bboxes
            )
            if angles is None:
                return "ambiguous"

            bins = [0, 135, 225, 360]
            quadrants = np.digitize(angles, bins=bins)
            quadrant_of_centroid = quadrants[0]
            quadrant_of_vertices = quadrants[1:]

            if (quadrant_of_centroid != quadrant_of_vertices).sum() > 2:
                return "ambiguous"

            ambiguity_threshold = 10
            boundaries = np.array(bins)
            if np.abs(angles[0] - boundaries).min() < ambiguity_threshold:
                return "ambiguous"

            if quadrants[0] == 1:
                return "left"
            elif quadrants[0] == 3:
                return "right"
            else:
                return "ambiguous"

        def get_rel_direction_lrb(
            positioning_category, orienting_category,
            querying_category, object_bboxes
        ):
            """Relative direction: left, right, or back.

            Returns: left, right, back, or ambiguous.
            """
            angles = _get_direction_data(
                positioning_category, orienting_category,
                querying_category, object_bboxes
            )
            if angles is None:
                return "ambiguous"

            bins = [0, 135, 225, 360]
            quadrants = np.digitize(angles, bins=bins)
            quadrant_of_centroid = quadrants[0]
            quadrant_of_vertices = quadrants[1:]

            if (quadrant_of_centroid != quadrant_of_vertices).sum() > 2:
                return "ambiguous"

            ambiguity_threshold = 10
            boundaries = np.array(bins)
            if np.abs(angles[0] - boundaries).min() < ambiguity_threshold:
                return "ambiguous"

            if quadrants[0] == 1:
                return "left"
            elif quadrants[0] == 2:
                return "back"
            elif quadrants[0] == 3:
                return "right"
            else:
                return "ambiguous"

        def get_closest_among_choices(
            primary_category, choice_categories, object_bboxes
        ):
            """Find which choice category is closest to primary.

            Returns: category name or ambiguous.
            """
            primary_bboxes = object_bboxes.get(primary_category, [])
            if not primary_bboxes:
                return "ambiguous"

            distances = []
            valid_choices = []
            for category in choice_categories:
                cat_bboxes = object_bboxes.get(category, [])
                if not cat_bboxes:
                    continue
                dist = cal_3d_bbox_distance_between_categories(
                    cat_bboxes, primary_bboxes
                )
                distances.append(dist)
                valid_choices.append(category)

            if len(distances) < 2:
                return "ambiguous"

            min_distance = min(distances)
            if min_distance < 0.15:
                return "ambiguous"

            for i in range(len(distances)):
                for j in range(i + 1, len(distances)):
                    if abs(distances[i] - distances[j]) <= 0.15:
                        return "ambiguous"

            min_index = distances.index(min_distance)
            return valid_choices[min_index]

        # --- Build namespace and execute ---

        object_bboxes = scene_info.get('object_bboxes', {})
        namespace = {
            '__builtins__': restricted_builtins,
            # Data
            'scene_info': scene_info,
            'frame_info': frame_info or {},
            # Convenience accessors
            'object_counts': scene_info.get('object_counts', {}),
            'object_bboxes': object_bboxes,
            'room_size': scene_info.get('room_size'),
            'room_center': scene_info.get('room_center'),
            # Constants
            'inf': float('inf'),
            # Modules
            'np': np,
            'numpy': np,
            'math': math,
            # Functions from common_utils
            'cal_3d_bbox_distance_between_categories': (
                cal_3d_bbox_distance_between_categories
            ),
            'sample_points_in_oriented_bbox_uniform': (
                sample_points_in_oriented_bbox_uniform
            ),
            # Helper functions
            'get_centroid': get_centroid,
            'get_all_centroids': get_all_centroids,
            'euclidean_distance': euclidean_distance,
            'find_closest_object': find_closest_object,
            'find_farthest_object': find_farthest_object,
            'count_objects_within_distance': (
                count_objects_within_distance
            ),
            # Relative direction/distance functions
            'get_rel_direction_quadrant': get_rel_direction_quadrant,
            'get_rel_direction_lr': get_rel_direction_lr,
            'get_rel_direction_lrb': get_rel_direction_lrb,
            'get_closest_among_choices': get_closest_among_choices,
        }

        processed_code = preprocess_code(code)
        exec(processed_code, namespace)

        if 'compute_answer' not in namespace:
            return {
                'success': False,
                'result': None,
                'error': (
                    "Code must define a 'compute_answer()' function"
                ),
            }

        result = namespace['compute_answer']()
        return {
            'success': True,
            'result': result,
            'error': None,
        }

    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_lines = traceback.format_exception(
            exc_type, exc_value, exc_tb
        )
        error_msg = ''.join(tb_lines[-3:])
        return {
            'success': False,
            'result': None,
            'error': f"{type(e).__name__}: {e}\n{error_msg}",
        }


class ScriptExecutor:
    """Executes LLM-generated Python scripts in a sandboxed environment.

    Uses a persistent worker process pool (size=1) to avoid
    re-importing heavy libraries on every execution. The worker
    imports open3d, torch, scipy etc. once at pool creation,
    then reuses them for all subsequent executions.

    If a worker times out or crashes, the pool is automatically
    recreated on the next call.
    """

    def __init__(
        self, execution_timeout: int = DEFAULT_EXECUTION_TIMEOUT
    ):
        self.execution_timeout = execution_timeout
        self._pool = None
        # Warm up: create pool eagerly so heavy imports (open3d etc.)
        # don't count against execution timeout
        self._get_pool()

    def _get_pool(self):
        """Get or create the worker pool."""
        if self._pool is None:
            logger.debug("[ScriptExecutor] Creating worker pool...")
            self._pool = multiprocessing.Pool(
                processes=1, initializer=_init_worker
            )
            logger.debug("[ScriptExecutor] Worker pool ready")
        return self._pool

    def _reset_pool(self, recreate: bool = True):
        """Terminate and recreate the worker pool.

        Args:
            recreate: If True, immediately create a new pool so
                heavy imports are done before the next execute() call.
        """
        if self._pool is not None:
            logger.debug("[ScriptExecutor] Resetting worker pool")
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            self._pool = None
        if recreate:
            self._get_pool()

    def execute(
        self,
        code: str,
        scene_info: Dict,
        frame_info: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute LLM-generated code with timeout.

        The code should define a `compute_answer()` function
        that returns the answer.

        Args:
            code: Python code to execute
            scene_info: Scene metadata dictionary
            frame_info: Frame metadata dictionary (optional)
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with success status and result/error
        """
        timeout = timeout or self.execution_timeout

        try:
            pool = self._get_pool()
            async_result = pool.apply_async(
                _execute_in_pool_worker,
                (code, scene_info, frame_info or {})
            )

            result_data = async_result.get(timeout=timeout)

            if result_data['success']:
                return ExecutionResult(
                    success=True,
                    result=result_data['result']
                )
            else:
                logger.warning(
                    f"Code execution failed: "
                    f"{result_data['error']}\n"
                    f"Code:\n{code[:500]}..."
                )
                return ExecutionResult(
                    success=False,
                    result=None,
                    error_message=result_data['error']
                )

        except multiprocessing.TimeoutError:
            # Worker is stuck — kill and recreate on next call
            self._reset_pool()
            error_msg = (
                f"Execution timeout: code took longer than "
                f"{timeout} seconds"
            )
            logger.warning(
                f"{error_msg}\nCode:\n{code[:500]}..."
            )
            return ExecutionResult(
                success=False,
                result=None,
                error_message=error_msg
            )

        except Exception as e:
            # Pool may be in bad state — reset it
            self._reset_pool()
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_lines = traceback.format_exception(
                exc_type, exc_value, exc_tb
            )
            error_msg = ''.join(tb_lines[-3:])
            full_msg = (
                f"Process error: {type(e).__name__}: {e}\n"
                f"{error_msg}"
            )
            logger.error(
                f"{full_msg}\nCode:\n{code[:500]}..."
            )
            return ExecutionResult(
                success=False,
                result=None,
                error_message=full_msg
            )

    def shutdown(self):
        """Shutdown the worker pool. Call when done."""
        self._reset_pool(recreate=False)

    def get_available_functions_doc(self) -> str:
        """Get documentation for available functions."""
        return """
## Available Functions

### 1. Object Counting
Access object counts directly from `object_counts` dict:
```python
def compute_answer():
    category = "chair"
    count = object_counts.get(category, 0)
    return count
```

### 2. Object Distance
Calculate minimum distance between two object categories:
```python
def compute_answer():
    category1 = "table"
    category2 = "chair"
    cate1_bbox_info = object_bboxes.get(category1, [])
    cate2_bbox_info = object_bboxes.get(category2, [])
    distance = cal_3d_bbox_distance_between_categories(
        cate1_bbox_info, cate2_bbox_info
    )
    return round(distance, 1)
```

### 3. Room Size
Access pre-computed room size:
```python
def compute_answer():
    return round(room_size, 1) if room_size else 0
```

### 4. Room Center
Get the center coordinates of the room:
```python
def compute_answer():
    if room_center is not None:
        return room_center  # [x, y, z] coordinates
    return None
```

## Available Data Variables

- `scene_info`: Full scene metadata dictionary
- `frame_info`: Frame-level metadata (if available)
- `object_counts`: Dict mapping category -> count
- `object_bboxes`: Dict mapping category -> list of bbox info
  - Each bbox has: "centroid", "axesLengths", "normalizedAxes"
- `room_size`: Pre-computed room area (if available)
- `room_center`: Pre-computed room center [x, y, z]

## Important Notes

1. Always define a `compute_answer()` function
2. Use round() for float answers
3. Handle missing data gracefully
4. For distance questions, both categories must exist
"""


# Singleton instance
_executor = None


def get_executor() -> ScriptExecutor:
    """Get the singleton script executor instance."""
    global _executor
    if _executor is None:
        _executor = ScriptExecutor()
    return _executor
