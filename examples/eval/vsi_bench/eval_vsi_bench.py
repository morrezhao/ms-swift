"""
VSI-Bench Evaluation Example

This example demonstrates how to evaluate multimodal LLMs on VSI-Bench
for spatial reasoning capabilities.

VSI-Bench consists of 5000+ QA pairs from 288 egocentric videos covering:
- Configurational questions (spatial layout)
- Measurement estimation (numeric distances/dimensions)
- Spatiotemporal questions (time and space relationships)

Usage:
    python eval_vsi_bench.py --model Qwen/Qwen2.5-VL-7B-Instruct --video_dir /path/to/videos
"""
import argparse
import os
from typing import Optional


def evaluate_with_swift_infer(
    model_name: str,
    video_dir: str,
    num_frames: int = 32,
    limit: Optional[int] = None,
    output_dir: str = './output',
):
    """Evaluate using swift's inference engine.

    Args:
        model_name: Model name or path
        video_dir: Directory containing videos or frames
        num_frames: Number of frames to sample from video
        limit: Maximum number of samples to evaluate
        output_dir: Output directory for results
    """
    import json

    from datasets import load_dataset

    from swift.arguments import InferArguments
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    from swift.pipelines.eval.vsi_bench import VSIBenchEvaluator
    from swift.pipelines.utils import prepare_model_template

    os.makedirs(output_dir, exist_ok=True)

    # Load model with swift
    print(f'Loading model: {model_name}')
    infer_args = InferArguments(model=model_name, infer_backend='pt')
    model, template = prepare_model_template(infer_args)
    engine = TransformersEngine(model, template=template)

    # Load VSI-Bench dataset
    print('Loading VSI-Bench dataset...')
    dataset = load_dataset('nyu-visionx/VSI-Bench', split='test')
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    print(f'Evaluating on {len(dataset)} samples...')

    # Prepare evaluator
    evaluator = VSIBenchEvaluator()
    results = []

    request_config = RequestConfig(max_tokens=256, temperature=0.0)

    for idx, row in enumerate(dataset):
        sample_id = row.get('id', idx)
        question = row.get('question', '')
        options = row.get('options')
        ground_truth = row.get('ground_truth', '')
        question_type = row.get('question_type', '')
        dataset_name = row.get('dataset', '')
        scene_name = row.get('scene_name', '')

        # Build prompt
        if options:
            prompt = question + '\n' + '\n'.join(options)
            prompt += '\n\nPlease answer with just the letter (A, B, C, or D).'
            is_multiple_choice = True
        else:
            prompt = question + '\n\nPlease provide the numerical answer only.'
            is_multiple_choice = False

        # Find video/frames
        videos = []
        images = []
        if video_dir:
            video_path = os.path.join(video_dir, dataset_name, scene_name)
            video_file = f'{video_path}.mp4'
            if os.path.exists(video_file):
                videos = [video_file]
            elif os.path.isdir(video_path):
                frame_files = sorted([
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.endswith(('.jpg', '.png', '.jpeg'))
                ])
                if frame_files and len(frame_files) > num_frames:
                    indices = [int(i * len(frame_files) / num_frames) for i in range(num_frames)]
                    frame_files = [frame_files[i] for i in indices]
                images = frame_files

        # Create inference request
        request = InferRequest(messages=[{'role': 'user', 'content': prompt}], videos=videos, images=images)

        # Run inference
        try:
            response = engine.infer([request], request_config, use_tqdm=False)[0]
            prediction = response.choices[0].message.content
        except Exception as e:
            print(f'Error on sample {sample_id}: {e}')
            prediction = ''

        # Add to evaluator
        evaluator.add_prediction(
            sample_id=sample_id,
            question_type=question_type,
            prediction=prediction,
            ground_truth=ground_truth,
            is_multiple_choice=is_multiple_choice,
            raw_response=prediction,
        )

        results.append({
            'id': sample_id,
            'question_type': question_type,
            'question': question,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'is_multiple_choice': is_multiple_choice,
        })

        if (idx + 1) % 10 == 0:
            print(f'Processed {idx + 1}/{len(dataset)} samples')

    # Save and print results
    evaluator.save_results(os.path.join(output_dir, 'vsi_bench_results.json'))
    evaluator.print_report()

    # Save predictions
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return evaluator.evaluate()


def evaluate_with_api(
    api_url: str,
    api_key: str,
    video_dir: str,
    num_frames: int = 32,
    limit: Optional[int] = None,
    output_dir: str = './output',
):
    """Evaluate using OpenAI-compatible API.

    Args:
        api_url: API endpoint URL
        api_key: API key
        video_dir: Directory containing videos or frames
        num_frames: Number of frames to sample
        limit: Maximum number of samples
        output_dir: Output directory
    """
    import base64

    import requests
    from datasets import load_dataset

    from swift.pipelines.eval.vsi_bench import VSIBenchEvaluator

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print('Loading VSI-Bench dataset...')
    dataset = load_dataset('nyu-visionx/VSI-Bench', split='test')
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    evaluator = VSIBenchEvaluator()

    for idx, row in enumerate(dataset):
        sample_id = row.get('id', idx)
        question = row.get('question', '')
        options = row.get('options')
        ground_truth = row.get('ground_truth', '')
        question_type = row.get('question_type', '')
        dataset_name = row.get('dataset', '')
        scene_name = row.get('scene_name', '')

        # Build prompt
        if options:
            prompt = question + '\n' + '\n'.join(options)
            prompt += '\n\nPlease answer with just the letter (A, B, C, or D).'
            is_multiple_choice = True
        else:
            prompt = question + '\n\nPlease provide the numerical answer only.'
            is_multiple_choice = False

        # Prepare content with images
        content = []

        # Find and encode frames
        if video_dir:
            video_path = os.path.join(video_dir, dataset_name, scene_name)
            if os.path.isdir(video_path):
                frame_files = sorted([
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.endswith(('.jpg', '.png', '.jpeg'))
                ])
                if frame_files and len(frame_files) > num_frames:
                    indices = [int(i * len(frame_files) / num_frames) for i in range(num_frames)]
                    frame_files = [frame_files[i] for i in indices]

                for frame_path in frame_files:
                    with open(frame_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    content.append({
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{img_data}'}
                    })

        content.append({'type': 'text', 'text': prompt})

        # Call API
        try:
            response = requests.post(
                f'{api_url}/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': 'default',
                    'messages': [{'role': 'user', 'content': content}],
                    'max_tokens': 256,
                    'temperature': 0.0,
                },
                timeout=60,
            )
            response.raise_for_status()
            prediction = response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f'Error on sample {sample_id}: {e}')
            prediction = ''

        evaluator.add_prediction(
            sample_id=sample_id,
            question_type=question_type,
            prediction=prediction,
            ground_truth=ground_truth,
            is_multiple_choice=is_multiple_choice,
        )

        if (idx + 1) % 10 == 0:
            print(f'Processed {idx + 1}/{len(dataset)} samples')

    evaluator.save_results(os.path.join(output_dir, 'vsi_bench_results.json'))
    evaluator.print_report()

    return evaluator.evaluate()


def main():
    parser = argparse.ArgumentParser(description='VSI-Bench Evaluation')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', help='Model name')
    parser.add_argument('--video_dir', type=str, required=True, help='Video/frames directory')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of frames')
    parser.add_argument('--limit', type=int, default=None, help='Sample limit')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--api_url', type=str, default=None, help='API URL (for API-based evaluation)')
    parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')

    args = parser.parse_args()

    if args.api_url:
        evaluate_with_api(
            api_url=args.api_url,
            api_key=args.api_key,
            video_dir=args.video_dir,
            num_frames=args.num_frames,
            limit=args.limit,
            output_dir=args.output_dir,
        )
    else:
        evaluate_with_swift_infer(
            model_name=args.model,
            video_dir=args.video_dir,
            num_frames=args.num_frames,
            limit=args.limit,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    main()
