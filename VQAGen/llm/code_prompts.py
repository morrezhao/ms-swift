"""Prompt templates for code-based QA generation.

LLM generates Python code to compute answers instead of direct values.
For MC question types, LLM also provides the options directly.
"""

# =============================================================================
# Valid Question Types (10 types from VSIBench)
# =============================================================================

VALID_QUESTION_TYPES = {
    'object_abs_distance',      # Absolute distance between two objects (numeric)
    'object_counting',          # Count objects of a category (numeric)
    'object_rel_direction_v1',  # 4-quadrant relative direction (MC)
    'object_rel_direction_v2',  # Left/right relative direction (MC)
    'object_rel_direction_v3',  # Left/right/back relative direction (MC)
    'object_rel_distance_v1',   # Closest/farthest among 4 candidates (MC: 4 category names)
    'object_rel_distance_v2',   # Closest/farthest among 3 candidates (MC: 3 category names)
    'object_rel_distance_v3',   # Closer to A or B (MC: 2 category names)
    'object_size_estimation',   # Object size in cm (numeric)
    'room_size_estimation',     # Room area in sqm (numeric)
}

# Question types that have numeric answers (no MC options needed)
NUMERIC_ONLY_TYPES = {
    'object_abs_distance',
    'object_counting',
    'object_size_estimation',
    'room_size_estimation',
}

# Question types that need MC options (LLM provides them directly)
MC_QUESTION_TYPES = {
    'object_rel_direction_v1',  # Options: front-left, front-right, back-left, back-right
    'object_rel_direction_v2',  # Options: left, right
    'object_rel_direction_v3',  # Options: left, right, back
    'object_rel_distance_v1',   # Options: exactly 4 object category names from the question
    'object_rel_distance_v2',   # Options: exactly 3 object category names from the question
    'object_rel_distance_v3',   # Options: exactly 2 object category names from the question
}

# =============================================================================
# System Prompt
# =============================================================================

CODE_SYSTEM_PROMPT = """You are an expert visual question-answering pair generator. Generate QA pairs based on 3D scene metadata by writing Python code to compute answers.

## Valid Question Types

### Numeric Types (no options needed):
- `object_abs_distance`: Distance between two objects in meters. Round to 1 decimal.
  - **SINGLE-INSTANCE ONLY**: Both categories must have exactly 1 instance in the scene (count == 1)
- `object_counting`: Count of objects in a category. Return integer.
- `object_size_estimation`: Object max dimension in centimeters. Return integer.
  - **SINGLE-INSTANCE ONLY**: The category must have exactly 1 instance in the scene (count == 1)
- `room_size_estimation`: Room floor area in square meters. Round to 1 decimal.

### MC Types (you MUST provide options):
- `object_rel_direction_v1`: "Standing at X and facing Y, in which direction is Z?"
  - Options MUST be: ["front-left", "front-right", "back-left", "back-right"]
  - **SINGLE-INSTANCE ONLY**: X, Y, Z must each have exactly 1 instance in the scene (count == 1)
- `object_rel_direction_v2`: "Standing at X and facing Y, is Z on your left or right?"
  - Options MUST be: ["left", "right"]
  - **SINGLE-INSTANCE ONLY**: X, Y, Z must each have exactly 1 instance in the scene (count == 1)
- `object_rel_direction_v3`: "Standing at X and facing Y, is Z on your left, right, or behind you?"
  - Options MUST be: ["left", "right", "back"]
  - **SINGLE-INSTANCE ONLY**: X, Y, Z must each have exactly 1 instance in the scene (count == 1)
- `object_rel_distance_v1`: Compare 4 candidates — "Which of A, B, C, D is closest/farthest to X?"
  - Options MUST be exactly the 4 candidate object category names mentioned in the question
  - All 4 candidates + reference X must exist in the scene metadata
  - If a category has multiple instances, measure from the closest instance
  - Question should note: "If there are multiple instances of an object category, measure to the closest."
  - Example: "Measuring from the closest point of each object, which of these objects (monitor, desk, trash bin, keyboard) is the closest to the window? If there are multiple instances of an object category, measure to the closest." → options: ["monitor", "desk", "trash bin", "keyboard"]
- `object_rel_distance_v2`: Compare 3 candidates — "Which of A, B, C is closest/farthest to X?"
  - Options MUST be exactly the 3 candidate object category names mentioned in the question
  - All 3 candidates + reference X must exist in the scene metadata
  - If a category has multiple instances, measure from the closest instance
  - Question should note: "If there are multiple instances of an object category, measure to the closest."
- `object_rel_distance_v3`: Compare 2 candidates — "Is X closer/farther to A or B?"
  - Options MUST be exactly the 2 candidate object category names mentioned in the question
  - Both candidates + reference X must exist in the scene metadata
  - Can ask about closest OR farthest
  - If a category has multiple instances, measure from the closest instance
  - Example: "Is the chair closer to the table or the sofa? If there are multiple instances, measure to the closest." → options: ["table", "sofa"]

## Available Data (already in scope, DO NOT import anything):
- `object_counts`: Dict[str, int] — category -> count
- `object_bboxes`: Dict[str, List] — category -> list of bbox dicts, each with:
  - "centroid": [x, y, z], "axesLengths": [l, w, h], "normalizedAxes": 3x3 matrix
- `room_size`: float — room area in sqm
- `room_center`: [x, y, z] — room center coordinates
- `np`: numpy module
- `math`: math module

## Available Functions (already in scope, DO NOT import):
- `cal_3d_bbox_distance_between_categories(cate1_bbox_info: List[dict], cate2_bbox_info: List[dict]) -> float`: Min surface distance between two sets of oriented bounding boxes
- `get_centroid(category: str, object_bboxes: Dict[str, List[dict]], index: int = 0) -> np.ndarray | None`: Get centroid of the index-th instance as a numpy array, or None if not found
- `get_all_centroids(category: str, object_bboxes: Dict[str, List[dict]]) -> List[np.ndarray]`: Get all centroids for a category
- `euclidean_distance(pos1: np.ndarray, pos2: np.ndarray) -> float`: Euclidean distance between two positions
- `find_closest_object(reference_category: str, object_bboxes: Dict[str, List[dict]], exclude_categories: List[str] | None = None) -> str | None`: Find closest object category by centroid distance
- `find_farthest_object(reference_category: str, object_bboxes: Dict[str, List[dict]], exclude_categories: List[str] | None = None) -> str | None`: Find farthest object category by centroid distance
- `get_rel_direction_quadrant(positioning: str, orienting: str, querying: str, object_bboxes: Dict[str, List[dict]]) -> str`: Returns "front-left"/"front-right"/"back-left"/"back-right"/"ambiguous"
- `get_rel_direction_lr(positioning: str, orienting: str, querying: str, object_bboxes: Dict[str, List[dict]]) -> str`: Returns "left"/"right"/"ambiguous"
- `get_rel_direction_lrb(positioning: str, orienting: str, querying: str, object_bboxes: Dict[str, List[dict]]) -> str`: Returns "left"/"right"/"back"/"ambiguous"
- `get_closest_among_choices(primary: str, choices: List[str], object_bboxes: Dict[str, List[dict]]) -> str`: Returns closest category name or "ambiguous"

## Code Examples

### object_counting
```python
def compute_answer():
    return object_counts.get("chair", 0)
```

### object_abs_distance (SINGLE-INSTANCE categories only)
```python
def compute_answer():
    assert object_counts.get("table", 0) == 1, "table must have exactly 1 instance"
    assert object_counts.get("sofa", 0) == 1, "sofa must have exactly 1 instance"
    cate1_bbox = object_bboxes.get("table", [])
    cate2_bbox = object_bboxes.get("sofa", [])
    return round(cal_3d_bbox_distance_between_categories(cate1_bbox, cate2_bbox), 1)
```

### object_size_estimation (SINGLE-INSTANCE category only)
```python
def compute_answer():
    assert object_counts.get("table", 0) == 1, "table must have exactly 1 instance"
    bboxes = object_bboxes.get("table", [])
    if not bboxes:
        return 0
    return int(max(bboxes[0].get("axesLengths", [0, 0, 0])) * 100)
```

### room_size_estimation
```python
def compute_answer():
    return round(room_size, 1) if room_size else 0
```

### object_rel_direction_v1 (SINGLE-INSTANCE categories only)
```python
def compute_answer():
    for cat in ["table", "door", "sofa"]:
        assert object_counts.get(cat, 0) == 1, f"{cat} must have exactly 1 instance"
    return get_rel_direction_quadrant("table", "door", "sofa", object_bboxes)
```

### object_rel_direction_v2 (SINGLE-INSTANCE categories only)
```python
def compute_answer():
    for cat in ["table", "door", "sofa"]:
        assert object_counts.get(cat, 0) == 1, f"{cat} must have exactly 1 instance"
    return get_rel_direction_lr("table", "door", "sofa", object_bboxes)
```

### object_rel_direction_v3 (SINGLE-INSTANCE categories only)
```python
def compute_answer():
    for cat in ["table", "door", "sofa"]:
        assert object_counts.get(cat, 0) == 1, f"{cat} must have exactly 1 instance"
    return get_rel_direction_lrb("table", "door", "sofa", object_bboxes)
```

### object_rel_distance_v1 (4 candidates)
```python
def compute_answer():
    # "Which of chair, lamp, sofa, desk is closest to the table?"
    # options: ["chair", "lamp", "sofa", "desk"]
    reference = "table"
    candidates = ["chair", "lamp", "sofa", "desk"]
    return get_closest_among_choices(reference, candidates, object_bboxes)
```

### object_rel_distance_v2 (3 candidates, handles multi-instance)
```python
def compute_answer():
    # "Which of chair, sofa, lamp is farthest from the table?
    #  If there are multiple instances, measure to the closest."
    # options: ["chair", "sofa", "lamp"]
    reference = "table"
    candidates = ["chair", "sofa", "lamp"]
    ref_bbox = object_bboxes.get(reference, [])
    max_dist, farthest = -1, None
    for cat in candidates:
        cat_bbox = object_bboxes.get(cat, [])
        if cat_bbox and ref_bbox:
            # cal_3d_bbox_distance handles multiple instances (finds min distance)
            dist = cal_3d_bbox_distance_between_categories(cat_bbox, ref_bbox)
            if dist > max_dist:
                max_dist, farthest = dist, cat
    return farthest
```

### object_rel_distance_v3 (2 candidates, closer or farther, handles multi-instance)
```python
def compute_answer():
    # "Is the chair closer to the table or the sofa?
    #  If there are multiple instances, measure to the closest."
    # options: ["table", "sofa"] — answer is one of the two candidates
    ref_bbox = object_bboxes.get("chair", [])
    a_bbox = object_bboxes.get("table", [])
    b_bbox = object_bboxes.get("sofa", [])
    # cal_3d_bbox_distance handles multiple instances (finds min distance)
    dist_a = cal_3d_bbox_distance_between_categories(ref_bbox, a_bbox)
    dist_b = cal_3d_bbox_distance_between_categories(ref_bbox, b_bbox)
    # For "closer": return the one with smaller distance
    return "table" if dist_a < dist_b else "sofa"
    # For "farther": return the one with larger distance
    # return "table" if dist_a > dist_b else "sofa"
```

## Output Format

Return a single JSON object (NOT wrapped in an array):

```json
{
  "question_type": "object_counting",
  "question": "How many chairs are in this room?",
  "code": "def compute_answer():\\n    return object_counts.get(\\"chair\\", 0)",
  "options": null
}
```

MC type example:

```json
{
  "question_type": "object_rel_direction_v1",
  "question": "Standing at the table and facing the door, in which direction is the sofa?",
  "code": "def compute_answer():\\n    return get_rel_direction_quadrant(\\"table\\", \\"door\\", \\"sofa\\", object_bboxes)",
  "options": ["front-left", "front-right", "back-left", "back-right"]
}
```

## Rules
1. **NEVER import anything** — all modules and functions are pre-loaded
2. **Define compute_answer()** that returns the answer
3. **Use ONLY valid question_types** listed above
4. **Use exact category names** matching the metadata
5. **Handle missing data** with .get() and empty list checks
6. **For MC types, always provide options**; for numeric types, set options to null
7. **For object_rel_distance types, options MUST be the exact candidate category names from the question** — v1 has 4 options, v2 has 3, v3 has 2. All candidates must exist in the scene.
8. **SINGLE-INSTANCE constraint**: For `object_abs_distance`, `object_size_estimation`, and all `object_rel_direction` types, ONLY use categories where `object_counts[category] == 1`. Check the Object Inventory to confirm before choosing categories.
9. **Multi-instance handling**: For `object_rel_distance` types, categories CAN have multiple instances. Use `cal_3d_bbox_distance_between_categories()` which automatically measures to the closest instance. Add a note in the question: "If there are multiple instances of an object category, measure to the closest."
10. **NEVER return "unknown", "ambiguous", or "skip"** — choose different objects if needed
11. **NEVER reference instance indices** (e.g., "the second chair")
12. **Verify data exists** before generating questions
"""

# =============================================================================
# User Prompt Template
# =============================================================================

CODE_USER_PROMPT_TEMPLATE = """Generate a QA pair based on this scene:

{scene_context}

Requirements:
- Choose a question type from the valid types
- Write Python code in the `code` field to compute the answer
- For MC types, include the `options` list; for numeric types, set `options` to null
- Use only object categories that exist in the scene
- Make the question natural and clear

Output valid JSON with the format specified in the system prompt."""

# =============================================================================
# Retry Prompt Template
# =============================================================================

CODE_RETRY_PROMPT_TEMPLATE = """Your previous code had execution errors. Fix them:

## Original Output
```json
{original_output}
```

## Errors
{execution_errors}

## Fix Instructions
1. REMOVE ALL import statements
2. Check category names match metadata exactly
3. Ensure compute_answer() is defined and returns a value
4. Use .get() with defaults, check empty lists before [0]
5. NEVER return "unknown" or "ambiguous"

Output the corrected complete JSON."""


def get_question_type_guidelines(question_type: str) -> str:
    """Get specific guidelines for a question type (for backward compatibility)."""
    return f"Generate questions of type `{question_type}` following the system prompt guidelines."
