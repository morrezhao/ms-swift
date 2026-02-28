import open3d as o3d
import numpy as np
import json
import math
import random
import tqdm
import torch
import alphashape
import logging

from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

FPS_LOOKUP = {
    "scannet": 24,
    "arkitscenes": 30,
    "scannetpp": 30
}


def load_scene_list(split_path):
    with open(split_path, "r") as f:
        scene_list = [line.strip() for line in f if line.strip()]
    return scene_list


def load_meta_info(meta_info_path):
    with open(meta_info_path, "r") as f:
        annos = json.load(f)
    return annos


def generate_multiple_choice(ground_truth, margin=0.20, lower_bound=0.25, upper_bound=1.75,
                             decimals=1, answer_counts=None, option_letters=None):
    EPS = math.ceil(ground_truth * margin * (10 ** decimals)) / (10 ** decimals) # scaling ceiling value to precision specified by decimals

    def sample_choices(ground_truth, lower_bound=lower_bound, upper_bound=upper_bound):
        lower_bound = ground_truth * lower_bound
        upper_bound = ground_truth * upper_bound
        choices = [random.uniform(lower_bound, upper_bound) for _ in range(3)]

        return [int(round(choice)) if decimals == 0 else round(choice, decimals) for choice in choices]

    def too_close(choices, ground_truth):
        return any(abs(choice - other) < EPS for i, choice in enumerate(choices) for other in choices[i+1:] + [ground_truth])

    choices = sample_choices(ground_truth)

    # Re-sample if any two choices are within epsilon of each other
    # If two choices are too close to each other it could make model's choice easier / harder
    max_attempts = 30  # to prevent an infinite loop
    attempts = 0
    while too_close(choices, ground_truth) and attempts <= max_attempts:
        # choices = sample_choices(ground_truth)

        # Re-sample only the choices that are too close to the others
        for i in range(len(choices)):
            if any(abs(choices[i] - other) < EPS for other in choices[:i] + choices[i+1:] + [ground_truth]):
                # Resample this choice
                new_choice = random.uniform(ground_truth * lower_bound, ground_truth * upper_bound)
                choices[i] = int(round(new_choice)) if decimals == 0 else round(new_choice, decimals)

        attempts += 1

    if too_close(choices, ground_truth):
        return [], "E", answer_counts  # E for Error

    # Add the GT to the three false answers
    choices.append(ground_truth)

    # Shuffle the choices
    # random.shuffle(choices)
    # correct_index = choices.index(ground_truth)

    # options = ['A', 'B', 'C', 'D']
    # correct_option = options[correct_index]
    options, mc_answer, answer_counts = from_options_to_mc_answer(
        choices, ground_truth, answer_counts, option_letters
    )

    return options, mc_answer, answer_counts


# Predefined option sets for different question types
OPTION_SETS = {
    # Direction types
    'direction_lr': ['left', 'right'],
    'direction_lrb': ['left', 'right', 'behind'],
    'direction_fb': ['in front', 'behind'],
    'direction_ud': ['above', 'below'],
    'direction_nf': ['near', 'far'],
    'direction_quadrant': ['back-right', 'front-left', 'front-right', 'back-left'],
    # Camera movement
    'camera_direction': ['forward', 'backward', 'left', 'right'],
    # Comparison
    'comparison': ['equal'],  # Will be extended with category names
}

# Invalid answer values that should cause QA to be discarded
INVALID_ANSWERS = {'unknown', 'n/a', 'none', 'null', 'undefined', 'ambiguous'}

# Placeholder patterns that should never appear in options
PLACEHOLDER_PATTERNS = {'option_1', 'option_2', 'option_3', 'option_4'}

# Comparison question types that should have 2-3 options instead of 4
COMPARISON_QUESTION_TYPES = {
    'distance_comparison', 'count_comparison', 'size_comparison',
    'object_comparison', 'which_closer', 'which_larger', 'which_more',
    'comparison', 'rel_distance', 'rel_size'
}


def generate_mc_options_from_ground_truth(
    ground_truth,
    question_type: str = None,
    scene_info: dict = None,
    num_options: int = 4,
    answer_counts: dict = None,
    option_letters: list = None,
    decimals: int = 1,
    comparison_options: list = None
):
    """
    Generate multiple choice options from ground truth answer.

    This function handles both numeric and string answers, generating appropriate
    distractor options based on the question type and scene context.

    Args:
        ground_truth: The correct answer (can be int, float, or str)
        question_type: Type of question (e.g., 'object_direction_lr', 'object_counting')
        scene_info: Scene metadata dict (used for generating category-based distractors)
        num_options: Number of options to generate (default 4)
        answer_counts: Dict tracking answer distribution (e.g., {'A': 0, 'B': 0, ...})
        option_letters: List of option letters (e.g., ['A', 'B', 'C', 'D'])
        decimals: Decimal places for numeric answers
        comparison_options: For comparison questions, explicit list of valid options
                           (e.g., ['chairs', 'tables'] or ['chairs', 'tables', 'equal'])
                           Required for comparison questions with category answers.

    Returns:
        tuple: (options_list, mc_answer, updated_answer_counts) or (None, None, answer_counts) if invalid
            - options_list: List of option values (without letter prefixes), or None if invalid
            - mc_answer: The letter of the correct answer (e.g., 'A'), or None if invalid
            - updated_answer_counts: Updated answer distribution dict
    """
    if option_letters is None:
        option_letters = ['A', 'B', 'C', 'D'][:num_options]
    if answer_counts is None:
        answer_counts = {letter: 0 for letter in option_letters}

    # Rule 1: Reject unknown/invalid ground truth values
    if isinstance(ground_truth, str):
        gt_lower = ground_truth.lower().strip()
        if gt_lower in INVALID_ANSWERS:
            logger.warning(f"Rejecting QA with invalid ground_truth: '{ground_truth}'")
            return None, None, answer_counts

    # Rule 1b: Reject ground_truth of 0.0 float (indicates invalid question)
    # Note: int 0 is valid (e.g., counting questions where count is 0)
    if isinstance(ground_truth, float) and ground_truth == 0.0:
        logger.warning(f"Rejecting QA with ground_truth=0.0 (indicates invalid question)")
        return None, None, answer_counts

    # Rule 3: Determine if this is a comparison question that should have 2-3 options
    is_comparison = False
    if question_type:
        qt_lower = question_type.lower()
        for comp_type in COMPARISON_QUESTION_TYPES:
            if comp_type in qt_lower:
                is_comparison = True
                break

    # If comparison_options is provided, use it directly (for comparison questions)
    if comparison_options is not None and len(comparison_options) >= 2:
        # Validate ground_truth is in the options
        gt_in_options = False
        gt_match = ground_truth
        for opt in comparison_options:
            if isinstance(ground_truth, str) and isinstance(opt, str):
                if opt.lower().strip() == ground_truth.lower().strip():
                    gt_in_options = True
                    gt_match = opt
                    break
            elif opt == ground_truth:
                gt_in_options = True
                gt_match = opt
                break

        if not gt_in_options:
            logger.warning(f"Ground truth '{ground_truth}' not in comparison_options: {comparison_options}")
            return None, None, answer_counts

        # Use the provided comparison options
        actual_letters = option_letters[:len(comparison_options)]
        options, mc_answer, answer_counts = from_options_to_mc_answer(
            list(comparison_options), gt_match, answer_counts, actual_letters
        )
        return options, mc_answer, answer_counts

    # Auto-detect counting questions and force integer options
    if question_type and 'counting' in question_type.lower():
        decimals = 0
    # Also force integers if ground_truth is already an integer
    elif isinstance(ground_truth, int):
        decimals = 0

    # Handle numeric answers
    if isinstance(ground_truth, (int, float)):
        # Ensure ground_truth is int when decimals=0
        if decimals == 0:
            ground_truth = int(round(ground_truth))
        return _generate_numeric_options(
            ground_truth, num_options, answer_counts, option_letters, decimals
        )

    # Handle string answers
    if isinstance(ground_truth, str):
        return _generate_string_options(
            ground_truth, question_type, scene_info, num_options,
            answer_counts, option_letters, is_comparison
        )

    # Fallback: just return the ground truth as only option
    return [ground_truth], 'A', answer_counts


def _generate_numeric_options(
    ground_truth: float,
    num_options: int,
    answer_counts: dict,
    option_letters: list,
    decimals: int
):
    """Generate options for numeric answers.

    Rule 5: For counting questions (decimals=0):
    - Represent counts as integers
    - Generate plausible distractors around ground truth (gt±1, gt±2)
    - Deduplicate options after normalization
    - Return None if deduplication yields <2 options
    """
    gt_value = ground_truth
    logger.debug(f"[NumericOptions] 开始生成选项: gt_value={gt_value}, decimals={decimals}")

    # Helper function to normalize and deduplicate
    def normalize_value(val):
        if decimals == 0:
            return int(round(val))
        return round(val, decimals)

    # For counting questions (integers), use ±1, ±2 approach
    if decimals == 0:
        # Generate distractors around ground truth
        potential_distractors = []
        for offset in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
            candidate = gt_value + offset
            if candidate >= 0:  # counts must be non-negative
                potential_distractors.append(candidate)

        # Shuffle and select unique values
        random.shuffle(potential_distractors)
        choices = []
        for c in potential_distractors:
            if c not in choices and c != gt_value:
                choices.append(c)
            if len(choices) >= num_options - 1:
                break
    else:
        # Handle zero or near-zero values specially
        if abs(gt_value) < 0.1:
            # For zero/near-zero, use absolute offsets
            base_offsets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            random.shuffle(base_offsets)
            choices = [normalize_value(gt_value + offset) for offset in base_offsets[:num_options-1]]
        else:
            # For non-zero values, use percentage-based offsets
            margin = 0.15
            min_eps = max(0.1, abs(gt_value) * margin)

            choices = []
            attempts = 0
            max_attempts = 50

            while len(choices) < num_options - 1 and attempts < max_attempts:
                # Generate random multiplier between 0.3 and 2.0
                multiplier = random.uniform(0.3, 2.0)
                candidate = gt_value * multiplier
                candidate = normalize_value(candidate)

                # Check if candidate is valid (not too close to gt or existing choices)
                if abs(candidate - gt_value) >= min_eps:
                    if all(abs(candidate - c) >= min_eps for c in choices):
                        if candidate > 0:  # Keep positive for distances/counts
                            choices.append(candidate)
                attempts += 1

            # If we couldn't generate enough choices, use fixed offsets
            fallback_attempts = 0
            max_fallback_attempts = 100
            while len(choices) < num_options - 1 and fallback_attempts < max_fallback_attempts:
                fallback_attempts += 1
                offset = random.choice([0.3, 0.5, 0.7, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0])
                candidate = normalize_value(gt_value * offset)
                if candidate > 0 and candidate not in choices and abs(candidate - gt_value) >= min_eps:
                    choices.append(candidate)
                else:
                    abs_offset = random.uniform(0.5, 3.0)
                    fallback = normalize_value(gt_value + abs_offset)
                    if fallback > 0 and fallback not in choices and fallback != gt_value:
                        choices.append(fallback)

            # 如果仍然不够，强制添加一些不同的值
            if len(choices) < num_options - 1:
                logger.debug(f"[NumericOptions] 进入强制添加阶段，当前 choices={choices}")
            force_counter = 1
            while len(choices) < num_options - 1:
                forced_value = normalize_value(gt_value + force_counter * 0.5)
                if forced_value > 0 and forced_value not in choices and forced_value != gt_value:
                    choices.append(forced_value)
                force_counter += 1
                if force_counter > 20:
                    logger.warning(f"[NumericOptions] 强制退出循环，gt_value={gt_value}, choices={choices}")
                    break

    # Final deduplication (Rule 5)
    choices = list(set(choices))
    choices = [c for c in choices if c != gt_value]

    logger.debug(f"[NumericOptions] 去重后选项: choices={choices}")

    # Rule 5: If deduplication yields <1 distractor (i.e., <2 total options), return None
    if len(choices) < 1:
        logger.warning(f"[NumericOptions] 去重后选项不足，无法生成有效 MC: gt_value={gt_value}")
        return None, None, answer_counts

    # Ensure we have at most num_options - 1 distractors
    choices = choices[:num_options - 1]

    # Add ground truth
    choices.append(gt_value)

    # Use from_options_to_mc_answer for balanced distribution
    options, mc_answer, answer_counts = from_options_to_mc_answer(
        choices, gt_value, answer_counts, option_letters[:len(choices)]
    )

    return options, mc_answer, answer_counts


def _generate_string_options(
    ground_truth: str,
    question_type: str,
    scene_info: dict,
    num_options: int,
    answer_counts: dict,
    option_letters: list,
    is_comparison: bool = False
):
    """Generate options for string answers based on question type.

    Rules:
    - Rule 2: Never output placeholder strings (option_1, option_2, etc.)
    - Rule 3: For comparison questions, options should be exactly the valid semantic outcomes (2-3 options OK)
    - Returns (None, None, answer_counts) if valid options cannot be generated
    """
    gt_lower = ground_truth.lower().strip()

    # Try to find matching option set based on question type or ground truth
    option_set = None

    # Check question type first
    if question_type:
        qt_lower = question_type.lower()
        # Handle object_rel_direction variants - check specific suffixes first
        if 'object_rel_direction' in qt_lower:
            if qt_lower.endswith('_quadrantI') or '_quadrant' in qt_lower:
                # object_rel_direction_quadrant: back-right, front-left, front-right, back-left
                option_set = list(OPTION_SETS['direction_quadrant'])
            elif qt_lower.endswith('_lrb') or '_lrb' in qt_lower:
                # object_rel_direction_lrb: left, right, behind
                option_set = list(OPTION_SETS['direction_lrb'])
            elif qt_lower.endswith('_lr') or '_lr' in qt_lower:
                # object_rel_direction_lr: left, right
                option_set = list(OPTION_SETS['direction_lr'])
            else:
                # Plain object_rel_direction without suffix should be ignored
                logger.warning(f"Ignoring question_type '{question_type}' - use specific variant like _lr, _lrb, or _quadrant")
                return None, None, answer_counts
        elif 'direction_lr' in qt_lower or 'pos_lr' in qt_lower or qt_lower.endswith('_lr'):
            option_set = list(OPTION_SETS['direction_lr'])
        elif 'direction_fb' in qt_lower or qt_lower.endswith('_fb'):
            option_set = list(OPTION_SETS['direction_fb'])
        elif 'direction_ud' in qt_lower or 'pos_ud' in qt_lower or qt_lower.endswith('_ud'):
            option_set = list(OPTION_SETS['direction_ud'])
        elif 'direction_nf' in qt_lower or 'pos_nf' in qt_lower or qt_lower.endswith('_nf'):
            option_set = list(OPTION_SETS['direction_nf'])
        elif 'quadrant' in qt_lower:
            option_set = list(OPTION_SETS['direction_quadrant'])
        elif 'camera' in qt_lower and 'direction' in qt_lower:
            option_set = list(OPTION_SETS['camera_direction'])

    # Infer from ground truth value if no match yet
    if option_set is None:
        if gt_lower in ['left', 'right']:
            option_set = list(OPTION_SETS['direction_lr'])
        elif gt_lower in ['in front', 'behind', 'front', 'back']:
            option_set = list(OPTION_SETS['direction_fb'])
        elif gt_lower in ['above', 'below', 'up', 'down']:
            option_set = list(OPTION_SETS['direction_ud'])
        elif gt_lower in ['near', 'far', 'nearer', 'farther']:
            option_set = list(OPTION_SETS['direction_nf'])
        elif gt_lower in ['front-left', 'front-right', 'back-left', 'back-right']:
            option_set = list(OPTION_SETS['direction_quadrant'])
        elif gt_lower in ['forward', 'backward']:
            option_set = list(OPTION_SETS['camera_direction'])

    # If still no option set, try to use object categories from scene
    if option_set is None and scene_info:
        object_counts = scene_info.get('object_counts', {})
        categories = list(object_counts.keys())

        # Helper function to match ground truth with categories (handles plural/singular)
        def find_matching_category(gt, cats):
            """Find matching category, handling plural/singular variations."""
            gt_l = gt.lower().strip()
            for cat in cats:
                cat_l = cat.lower()
                # Exact match
                if gt_l == cat_l:
                    return cat
                # GT is plural of category (e.g., "chairs" -> "chair")
                if gt_l == cat_l + 's' or gt_l == cat_l + 'es':
                    return cat
                # GT is singular of category (e.g., "chair" -> "chairs")
                if gt_l + 's' == cat_l or gt_l + 'es' == cat_l:
                    return cat
            return None

        matched_category = find_matching_category(ground_truth, categories)

        # Rule 3: For comparison questions (count_comparison, etc.),
        # we CANNOT generate valid options without knowing the comparison targets.
        # The options must be exactly {X, Y} or {X, Y, equal}, not random categories.
        # Since we only have ground_truth and don't know the other comparison target,
        # comparison questions with category answers should be handled by the caller
        # who knows the full question context.
        if is_comparison and matched_category:
            # For comparison questions, we cannot add random distractors.
            # Return None to signal that the caller should provide explicit options.
            logger.warning(f"Comparison question with category answer '{ground_truth}' - "
                          f"caller should provide explicit comparison options")
            return None, None, answer_counts

        # For NON-comparison questions (e.g., "What is closest to X?"),
        # using other categories as distractors is valid
        if matched_category and not is_comparison:
            # Use other categories as distractors (in same form as ground_truth)
            is_plural = ground_truth.lower().endswith('s') and not matched_category.lower().endswith('s')
            other_categories = []
            for c in categories:
                if c != matched_category:
                    # Convert to same form as ground_truth (plural or singular)
                    if is_plural and not c.endswith('s'):
                        other_categories.append(c + 's')
                    else:
                        other_categories.append(c)

            if len(other_categories) >= 1:  # Need at least 1 distractor for 2 options
                random.shuffle(other_categories)
                max_distractors = num_options - 1
                option_set = [ground_truth] + other_categories[:max_distractors]

    # Rule 3: For comparison questions (e.g., "Which is closer, X or Y?"),
    # options should be exactly the valid semantic outcomes
    if is_comparison and option_set is None:
        # Cannot determine valid options for comparison question
        logger.warning(f"Cannot generate valid comparison options for ground_truth: '{ground_truth}'")
        return None, None, answer_counts

    # Build final options
    if option_set:
        # Ensure ground truth is in the option set
        gt_in_set = None
        for opt in option_set:
            if opt.lower().strip() == gt_lower:
                gt_in_set = opt
                break

        if gt_in_set is None:
            # Ground truth not in option set - add it
            option_set = [ground_truth] + [o for o in option_set if o.lower().strip() != gt_lower]
            gt_in_set = ground_truth

        # For comparison questions, keep semantic options (don't force to 4)
        # For other questions, limit to num_options
        if not is_comparison and len(option_set) > num_options:
            # Keep ground truth, sample from rest
            others = [o for o in option_set if o.lower().strip() != gt_lower]
            random.shuffle(others)
            option_set = [gt_in_set] + others[:num_options - 1]

        # Validate: must have at least 2 options
        if len(option_set) < 2:
            logger.warning(f"Not enough options for ground_truth: '{ground_truth}', option_set: {option_set}")
            return None, None, answer_counts

        # Adjust option_letters for actual number of options
        actual_letters = option_letters[:len(option_set)]

        options, mc_answer, answer_counts = from_options_to_mc_answer(
            list(option_set), gt_in_set, answer_counts, actual_letters
        )
        return options, mc_answer, answer_counts

    # Rule 2: NO PLACEHOLDERS - return None if we can't generate valid options
    logger.warning(f"Cannot generate valid options for ground_truth: '{ground_truth}', question_type: {question_type}")
    return None, None, answer_counts


def sample_points_in_oriented_bbox_uniform(bbox, distance=0.05):
    # Calculate number of points along each dimension
    nx = int(np.ceil(bbox.extent[0] / distance))
    ny = int(np.ceil(bbox.extent[1] / distance))
    nz = int(np.ceil(bbox.extent[2] / distance))

    # Generate uniform grid
    x = np.linspace(-bbox.extent[0]/2, bbox.extent[0]/2, nx)
    y = np.linspace(-bbox.extent[1]/2, bbox.extent[1]/2, ny)
    z = np.linspace(-bbox.extent[2]/2, bbox.extent[2]/2, nz)
    
    # Create meshgrid
    xx, yy, zz = np.meshgrid(x, y, z)
    
    # Reshape to (N, 3) array
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Create a mask for points to keep (outside the inner box)
    mask = np.any(np.abs(points) > bbox.extent / 4, axis=1)
    points = points[mask]

    # Rotate points
    R = bbox.R
    points = np.dot(points, R.T)

    # Translate points to bbox center
    points += bbox.center

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Crop points to ensure they're all within the bounding box
    pcd = pcd.crop(bbox)
    
    if len(pcd.points) == 0:
        return sample_points_in_oriented_bbox_uniform(bbox, distance=distance*0.5)

    return np.asarray(pcd.points)


def cal_3d_bbox_distance_between_categories(cate1_bbox_info, cate2_bbox_info):
    # build bounding boxes list
    cate1_bbox_list = []
    for ins in cate1_bbox_info:
        cate1_bbox = o3d.geometry.OrientedBoundingBox(
            center=np.array(ins["centroid"]).astype(np.float64).reshape(3, 1),
            R=np.array(ins["normalizedAxes"]).astype(np.float64).reshape(3, 3).T,
            extent=np.array(ins["axesLengths"]).astype(np.float64).reshape(3, 1)
        )
        cate1_bbox_list.append(cate1_bbox)

    cate2_bbox_list = []
    for ins in cate2_bbox_info:
        cate2_bbox = o3d.geometry.OrientedBoundingBox(
            center=np.array(ins["centroid"]).astype(np.float64).reshape(3, 1),
            R=np.array(ins["normalizedAxes"]).astype(np.float64).reshape(3, 3).T,
            extent=np.array(ins["axesLengths"]).astype(np.float64).reshape(3, 1)
        )
        cate2_bbox_list.append(cate2_bbox)

    # calculate distance
    ins_distance_matrix = np.zeros((len(cate1_bbox_list), len(cate2_bbox_list)))
    
    for i, cate1_bbox in enumerate(cate1_bbox_list):
        for j, cate2_bbox in enumerate(cate2_bbox_list):
            point_set1 = sample_points_in_oriented_bbox_uniform(cate1_bbox)
            point_set2 = sample_points_in_oriented_bbox_uniform(cate2_bbox)

            distances = np.min(cdist(point_set1, point_set2, 'euclidean'))
            ins_distance_matrix[i, j] = distances

    min_distance = np.min(ins_distance_matrix)

    return min_distance


def extract_cared_categories_from_qa(qa_pair):
    # determine category(s) at hand
    if qa_pair['question_type'] in ['room_size_estimation', 'route_planning']:
        categories = []
    elif qa_pair['question_type'] == 'object_abs_distance':
        categories = [qa_pair['category, first object'], qa_pair['category, second object']]
    elif qa_pair['question_type'] in ['object_rel_direction_v1', 'object_rel_direction_v2', 'object_rel_direction_v3']:
        categories = [qa_pair['category, positioning object'], qa_pair['category, orienting object'], qa_pair['category, querying object']]
    elif qa_pair['question_type'] == 'object_rel_distance':
        # four options plus the main object
        categories = [item.split(". ")[1] for item in qa_pair['options']]
        categories.append(qa_pair['category'])
    elif qa_pair['question_type'] == 'obj_appearance_order':
        categories = [item.strip() for item in qa_pair['ground_truth'].split(",")]
    elif qa_pair['question_type'] in ['object_size_estimation', 'object_counting']:
        categories = [qa_pair['category']]
    else:
        raise NotImplementedError(f"Question type {qa_pair['question_type']} not implemented yet.")
    
    return categories


def from_options_to_mc_answer(options, gt, answer_counts, option_letters):
    # Find the letter with the minimum count
    min_count = min(answer_counts.values())
    min_letters = [letter for letter, count in answer_counts.items() 
                   if count == min_count and letter in option_letters[:len(options)]]
    
    if min_letters:
        # Choose one of the minimum count letters randomly
        target_letter = random.choice(min_letters)
        target_index = option_letters.index(target_letter)
        
        # Rearrange options to put the correct answer in the target position
        correct_option = options[options.index(gt)]
        options.remove(correct_option)
        random.shuffle(options)
        
        final_options = options[:target_index] + [correct_option] + options[target_index:]
        if len(final_options) < len(options) + 1:
            final_options.extend(options[:(len(options) + 1 - len(final_options))])
    else:
        # Fallback to original random behavior
        random.shuffle(options)
        target_index = options.index(gt)
        final_options = options
        target_letter = option_letters[target_index]
    
    # Update answer counts
    answer_counts[target_letter] += 1

    # Format options with letters
    # new_options = [f"{option_letters[i]}. {opt}" for i, opt in enumerate(final_options)]

    return final_options, target_letter, answer_counts


if __name__ == "__main__":
    """
    Test MC option generation to verify fixes for:
    1. Duplicated options in counting questions (e.g., "2.0", "2.0")
    2. Placeholder options (e.g., "option_1", "option_2")
    3. Unknown ground truth should be rejected
    4. Comparison questions should have 2-3 options (not forced to 4)
    5. Counting questions should use integers (not "2.0", "0.0")
    """
    import sys

    # Setup logging for tests
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=" * 60)
    print("Testing MC Option Generation")
    print("=" * 60)

    test_results = []
    option_letters = ['A', 'B', 'C', 'D']

    def run_test(test_name, ground_truth, question_type, scene_info=None, expected_fail=False,
                 comparison_options=None):
        """Run a single test case."""
        print(f"\n--- Test: {test_name} ---")
        print(f"  Ground truth: {ground_truth} (type: {type(ground_truth).__name__})")
        print(f"  Question type: {question_type}")
        if comparison_options:
            print(f"  Comparison options: {comparison_options}")

        answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        options, mc_answer, _ = generate_mc_options_from_ground_truth(
            ground_truth=ground_truth,
            question_type=question_type,
            scene_info=scene_info,
            num_options=4,
            answer_counts=answer_counts,
            option_letters=option_letters,
            decimals=1,
            comparison_options=comparison_options
        )

        # Check results
        passed = True
        issues = []

        if expected_fail:
            if options is not None:
                passed = False
                issues.append("Expected to fail (return None) but got options")
            else:
                print(f"  ✓ Correctly rejected (returned None)")
        else:
            if options is None:
                passed = False
                issues.append("Unexpectedly returned None")
            else:
                # Format options for display
                formatted = [f"{option_letters[i]}. {opt}" for i, opt in enumerate(options)]
                print(f"  Options: {formatted}")
                print(f"  MC Answer: {mc_answer}")

                # Check for duplicates
                normalized = []
                for opt in options:
                    try:
                        val = float(opt)
                        if val == int(val):
                            normalized.append(str(int(val)))
                        else:
                            normalized.append(str(opt))
                    except (ValueError, TypeError):
                        normalized.append(str(opt).lower().strip())

                if len(normalized) != len(set(normalized)):
                    passed = False
                    issues.append(f"Duplicate options found: {options}")

                # Check for placeholders
                for opt in options:
                    opt_str = str(opt).lower()
                    if opt_str in {'option_1', 'option_2', 'option_3', 'option_4', 'n/a'}:
                        passed = False
                        issues.append(f"Placeholder option found: {opt}")

                # Check at least 2 options
                if len(options) < 2:
                    passed = False
                    issues.append(f"Less than 2 options: {len(options)}")

                # For counting questions, check integer format
                if question_type and 'counting' in question_type.lower():
                    for opt in options:
                        if isinstance(opt, float) and opt != int(opt):
                            passed = False
                            issues.append(f"Non-integer in counting: {opt}")

                if passed:
                    print(f"  ✓ PASSED")

        if not passed:
            for issue in issues:
                print(f"  ✗ FAILED: {issue}")

        test_results.append((test_name, passed))
        return passed

    # Mock scene_info for testing
    mock_scene_info = {
        'object_counts': {
            'chair': 2,
            'table': 1,
            'sofa': 1,
            'lamp': 3,
        },
        'object_bboxes': {
            'chair': [{'centroid': [0, 0, 0]}],
            'table': [{'centroid': [1, 1, 0]}],
        }
    }

    # ===========================================
    # Test Case 1: Counting question with zero
    # Problem: "0" vs "0.0" duplicates
    # ===========================================
    run_test(
        test_name="Counting with zero",
        ground_truth=0,
        question_type="object_counting",
        scene_info=mock_scene_info
    )

    # ===========================================
    # Test Case 2: Counting question with small number
    # Problem: duplicates like "2.0", "2.0"
    # ===========================================
    run_test(
        test_name="Counting with small number (2)",
        ground_truth=2,
        question_type="object_counting",
        scene_info=mock_scene_info
    )

    # ===========================================
    # Test Case 3: Counting with float input
    # Should be normalized to int
    # ===========================================
    run_test(
        test_name="Counting with float input (2.0)",
        ground_truth=2.0,
        question_type="object_counting",
        scene_info=mock_scene_info
    )

    # ===========================================
    # Test Case 4: Unknown ground truth - should fail
    # ===========================================
    run_test(
        test_name="Unknown ground truth",
        ground_truth="unknown",
        question_type="distance_comparison",
        scene_info=mock_scene_info,
        expected_fail=True
    )

    # ===========================================
    # Test Case 5: Comparison question (left/right)
    # Should have 2 options, not 4
    # ===========================================
    run_test(
        test_name="Direction comparison (left/right)",
        ground_truth="left",
        question_type="object_direction_lr",
        scene_info=mock_scene_info
    )

    # ===========================================
    # Test Case 6: Comparison question (count comparison) WITH explicit options
    # For comparison questions, caller must provide comparison_options
    # ===========================================
    run_test(
        test_name="Count comparison with explicit options",
        ground_truth="chairs",
        question_type="object_count_comparison",
        scene_info=mock_scene_info,
        comparison_options=["chairs", "tables", "equal"]
    )

    # ===========================================
    # Test Case 6b: Comparison question WITHOUT explicit options - should fail
    # ===========================================
    run_test(
        test_name="Count comparison without options (should fail)",
        ground_truth="chairs",
        question_type="object_count_comparison",
        scene_info=mock_scene_info,
        expected_fail=True
    )

    # ===========================================
    # Test Case 7: Distance value
    # ===========================================
    run_test(
        test_name="Distance value (2.5m)",
        ground_truth=2.5,
        question_type="object_abs_distance",
        scene_info=mock_scene_info
    )

    # ===========================================
    # Test Case 8: Room size
    # ===========================================
    run_test(
        test_name="Room size (15.3 sqm)",
        ground_truth=15.3,
        question_type="room_size",
        scene_info=mock_scene_info
    )

    # ===========================================
    # Test Case 9: Direction front/back
    # ===========================================
    run_test(
        test_name="Direction front/back",
        ground_truth="in front",
        question_type="object_direction_fb",
        scene_info=mock_scene_info
    )

    # ===========================================
    # Test Case 10: Quadrant direction
    # ===========================================
    run_test(
        test_name="Quadrant direction",
        ground_truth="front-left",
        question_type="object_rel_direction",
        scene_info=mock_scene_info
    )

    # ===========================================
    # Test Case 11: Category answer with scene_info (NON-comparison)
    # Ground truth is a category name - can use other categories as distractors
    # ===========================================
    run_test(
        test_name="Category answer (closest object)",
        ground_truth="chair",
        question_type="closest_object",
        scene_info=mock_scene_info
    )

    # ===========================================
    # Test Case 11b: Distance comparison (which is closer) with explicit options
    # ===========================================
    run_test(
        test_name="Distance comparison with explicit options",
        ground_truth="lamp",
        question_type="distance_comparison",
        scene_info=mock_scene_info,
        comparison_options=["lamp", "sofa"]
    )

    # ===========================================
    # Test Case 12: N/A ground truth - should fail
    # ===========================================
    run_test(
        test_name="N/A ground truth",
        ground_truth="N/A",
        question_type="object_counting",
        scene_info=mock_scene_info,
        expected_fail=True
    )

    # ===========================================
    # Test Case 13: Empty string - should fail
    # ===========================================
    run_test(
        test_name="Empty string ground truth",
        ground_truth="",
        question_type="object_counting",
        scene_info=mock_scene_info,
        expected_fail=True
    )

    # ===========================================
    # Test Case 14: String without matching option set
    # Should return None (no placeholders)
    # ===========================================
    run_test(
        test_name="Unrecognized string answer (no scene context)",
        ground_truth="some_random_answer",
        question_type="unknown_type",
        scene_info=None,
        expected_fail=True
    )

    # ===========================================
    # Test Case 15: Large counting number
    # ===========================================
    run_test(
        test_name="Large counting number (10)",
        ground_truth=10,
        question_type="object_counting",
        scene_info=mock_scene_info
    )

    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed_count = sum(1 for _, passed in test_results if passed)
    total_count = len(test_results)

    for test_name, passed in test_results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total_count - passed_count} tests failed!")
        sys.exit(1)

