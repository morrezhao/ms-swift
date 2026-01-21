# Copyright (c) ModelScope Contributors. All rights reserved.
"""
VSI-Bench Evaluation Script

This script provides end-to-end evaluation of MLLMs on VSI-Bench benchmark.

Usage:
    python -m swift.pipelines.eval.run_vsi_bench \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --video_dir /path/to/vsi_bench_videos \
        --output_dir ./vsi_bench_results

For using frames instead of videos:
    python -m swift.pipelines.eval.run_vsi_bench \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --video_dir /path/to/vsi_bench_frames \
        --num_frames 32 \
        --output_dir ./vsi_bench_results
"""
import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swift.utils import get_logger

logger = get_logger()


@dataclass
class VSIBenchArguments:
    """Arguments for VSI-Bench evaluation."""
    # Model settings
    model: str = field(default='Qwen/Qwen2.5-VL-7B-Instruct', metadata={'help': 'Model name or path'})
    adapters: Optional[List[str]] = field(default=None, metadata={'help': 'LoRA adapter paths'})
    infer_backend: str = field(default='transformers', metadata={'help': 'Inference backend: transformers, vllm'})
    tensor_parallel_size: Optional[int] = field(default=None, metadata={'help': 'Tensor parallel size for vLLM'})

    # Data settings
    data_path: Optional[str] = field(default=None, metadata={'help': 'Local path to VSI-Bench dataset (JSON/JSONL file)'})
    video_dir: str = field(default='', metadata={'help': 'Directory containing VSI-Bench videos or frames'})
    num_frames: int = field(default=32, metadata={'help': 'Number of frames to sample from video'})
    eval_limit: Optional[int] = field(default=None, metadata={'help': 'Limit number of samples to evaluate'})

    # Generation settings
    max_new_tokens: int = field(default=256, metadata={'help': 'Maximum new tokens to generate'})
    temperature: float = field(default=0.0, metadata={'help': 'Temperature for generation'})
    top_p: float = field(default=1.0, metadata={'help': 'Top-p sampling'})

    # Output settings
    output_dir: str = field(default='./vsi_bench_output', metadata={'help': 'Output directory for results'})
    verbose: bool = field(default=False, metadata={'help': 'Print verbose output'})


def load_vsi_bench_dataset(
    video_dir: str,
    num_frames: int = 32,
    limit: Optional[int] = None,
    data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load VSI-Bench dataset from local file or HuggingFace.

    Args:
        video_dir: Directory containing videos or frames
        num_frames: Number of frames to sample
        limit: Optional limit on number of samples
        data_path: Local path to dataset file (JSON or JSONL)

    Returns:
        List of dataset samples with video/frame paths
    """
    import json

    if data_path:
        # Load from local file
        logger.info(f'Loading VSI-Bench dataset from local file: {data_path}')

        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Dataset file not found: {data_path}')

        data = []
        if data_path.endswith('.jsonl'):
            # Load JSONL format
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        elif data_path.endswith('.json'):
            # Load JSON format
            with open(data_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    data = loaded
                elif isinstance(loaded, dict):
                    # Handle dict format like {"data": [...]}
                    for key in ['data', 'samples', 'test', 'items']:
                        if key in loaded and isinstance(loaded[key], list):
                            data = loaded[key]
                            break
                    if not data:
                        data = [loaded]
        else:
            raise ValueError(f'Unsupported file format: {data_path}. Use .json or .jsonl')

        dataset = data
    else:
        # Load from HuggingFace
        from datasets import load_dataset as hf_load_dataset

        logger.info('Loading VSI-Bench dataset from HuggingFace...')

        try:
            dataset = hf_load_dataset('nyu-visionx/VSI-Bench', split='test')
            dataset = list(dataset)
        except Exception as e:
            logger.error(f'Failed to load VSI-Bench dataset: {e}')
            logger.info('Trying to load from local cache or alternative source...')
            raise

    if limit:
        dataset = dataset[:limit]

    samples = []
    for idx, row in enumerate(dataset):
        sample = {
            'id': row.get('id', idx),
            'dataset': row.get('dataset', ''),
            'scene_name': row.get('scene_name', ''),
            'question_type': row.get('question_type', ''),
            'question': row.get('question', ''),
            'ground_truth': row.get('ground_truth', ''),
            'options': row.get('options'),
        }

        # Find video/frames
        dataset_name = sample['dataset']
        scene_name = sample['scene_name']

        if video_dir:
            video_path = os.path.join(video_dir, dataset_name, scene_name)
            video_file = f'{video_path}.mp4'

            if os.path.exists(video_file):
                sample['video_path'] = video_file
            else:
                # Try frame directory
                frame_dir = video_path
                if os.path.isdir(frame_dir):
                    frame_files = sorted([
                        os.path.join(frame_dir, f)
                        for f in os.listdir(frame_dir)
                        if f.endswith(('.jpg', '.png', '.jpeg'))
                    ])
                    if frame_files:
                        # Sample frames uniformly
                        if len(frame_files) > num_frames:
                            indices = [int(i * len(frame_files) / num_frames) for i in range(num_frames)]
                            frame_files = [frame_files[i] for i in indices]
                        sample['frame_paths'] = frame_files

        samples.append(sample)

    logger.info(f'Loaded {len(samples)} samples from VSI-Bench')
    return samples


def build_prompt(sample: Dict[str, Any]) -> str:
    """Build prompt for a VSI-Bench sample.

    Args:
        sample: Sample dictionary

    Returns:
        Formatted prompt string
    """
    question = sample['question']
    options = sample.get('options')

    if options:
        # Multiple-choice question
        prompt = question + '\n' + '\n'.join(options)
        prompt += '\n\nPlease answer with just the letter (A, B, C, or D).'
    else:
        # Numerical answer question
        prompt = question
        prompt += '\n\nPlease provide the numerical answer only.'

    return prompt


def run_inference(
    infer_engine,
    samples: List[Dict[str, Any]],
    args: VSIBenchArguments,
) -> List[Dict[str, Any]]:
    """Run inference on VSI-Bench samples.

    Args:
        infer_engine: Inference engine instance
        samples: List of dataset samples
        args: Evaluation arguments

    Returns:
        List of samples with predictions
    """
    from tqdm import tqdm

    from swift.infer_engine import InferRequest, RequestConfig

    request_config = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    results = []
    for i, sample in enumerate(tqdm(samples, desc='Inference', disable=args.verbose)):
        if args.verbose:
            logger.info(f'Processing sample {i+1}/{len(samples)}: {sample["id"]}')

        prompt = build_prompt(sample)

        # Prepare request
        messages = [{'role': 'user', 'content': prompt}]

        # Add video or images
        videos = []
        images = []
        if 'video_path' in sample:
            videos = [sample['video_path']]
        elif 'frame_paths' in sample:
            images = sample['frame_paths']

        infer_request = InferRequest(messages=messages, videos=videos, images=images)

        # Run inference
        try:
            response_list = infer_engine.infer([infer_request], request_config, use_tqdm=False)
            response = response_list[0]
            prediction = response.choices[0].message.content
        except Exception as e:
            logger.warning(f'Inference failed for sample {sample["id"]}: {e}')
            prediction = ''

        result = {
            'id': sample['id'],
            'question_type': sample['question_type'],
            'question': sample['question'],
            'ground_truth': sample['ground_truth'],
            'prediction': prediction,
            'is_multiple_choice': sample.get('options') is not None,
            'raw_response': prediction,
        }
        results.append(result)

        if args.verbose:
            logger.info(f'  Question: {sample["question"][:100]}...')
            logger.info(f'  Ground Truth: {sample["ground_truth"]}')
            logger.info(f'  Prediction: {prediction}')

    return results


def run_evaluation(args: VSIBenchArguments):
    """Run full VSI-Bench evaluation.

    Args:
        args: Evaluation arguments
    """
    from swift.arguments import InferArguments
    from swift.pipelines.eval.vsi_bench import VSIBenchEvaluator
    from swift.pipelines.infer import SwiftInfer

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model using InferArguments pattern
    logger.info(f'Loading model: {args.model}')
    logger.info(f'Inference backend: {args.infer_backend}')

    # Create InferArguments for model loading
    infer_kwargs = {
        'model': args.model,
        'infer_backend': args.infer_backend,
    }
    if args.adapters:
        infer_kwargs['adapters'] = args.adapters

    # Add tensor parallel size for vLLM
    if args.infer_backend == 'vllm' and args.tensor_parallel_size:
        infer_kwargs['vllm_tensor_parallel_size'] = args.tensor_parallel_size

    infer_args = InferArguments(**infer_kwargs)

    # Create inference engine using SwiftInfer's method
    if args.infer_backend == 'vllm':
        template = infer_args.get_template()
        infer_engine = SwiftInfer.get_infer_engine(infer_args, template)
    else:
        from swift.infer_engine import TransformersEngine
        from swift.pipelines.utils import prepare_model_template
        model, template = prepare_model_template(infer_args)
        infer_engine = TransformersEngine(model, template=template)

    logger.info(f'Model loaded: {infer_engine.model_name}')

    # Load dataset
    samples = load_vsi_bench_dataset(
        video_dir=args.video_dir,
        num_frames=args.num_frames,
        limit=args.eval_limit,
        data_path=args.data_path,
    )

    # Run inference
    logger.info('Running inference...')
    results = run_inference(infer_engine, samples, args)

    # Evaluate
    logger.info('Evaluating results...')
    evaluator = VSIBenchEvaluator()
    for result in results:
        evaluator.add_prediction(
            sample_id=result['id'],
            question_type=result['question_type'],
            prediction=result['prediction'],
            ground_truth=result['ground_truth'],
            is_multiple_choice=result['is_multiple_choice'],
            raw_response=result['raw_response'],
        )

    # Save and print results
    output_path = os.path.join(args.output_dir, 'vsi_bench_results.json')
    evaluator.save_results(output_path)
    evaluator.print_report()

    # Save raw predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.json')
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f'Predictions saved to {predictions_path}')

    return evaluator.evaluate()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='VSI-Bench Evaluation')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', help='Model name or path')
    parser.add_argument('--adapters', type=str, nargs='*', default=None, help='LoRA adapter paths')
    parser.add_argument('--infer_backend', type=str, default='transformers', help='Inference backend: transformers, vllm')
    parser.add_argument('--tensor_parallel_size', type=int, default=None, help='Tensor parallel size for vLLM (number of GPUs)')
    parser.add_argument('--data_path', type=str, default=None, help='Local path to VSI-Bench dataset (JSON/JSONL)')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing videos or frames')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of frames to sample')
    parser.add_argument('--eval_limit', type=int, default=None, help='Limit number of samples')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum new tokens')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p')
    parser.add_argument('--output_dir', type=str, default='./vsi_bench_output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    parsed_args = parser.parse_args()
    args = VSIBenchArguments(**vars(parsed_args))

    run_evaluation(args)


if __name__ == '__main__':
    main()
