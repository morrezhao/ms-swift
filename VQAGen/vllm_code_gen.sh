python -m tasks.code_qa_generator \
    --datasets scannet:/upfs/enhan/data/processed_data/ScanNet \
    --datasets scannetpp:/upfs/enhan/data/processed_data/ScanNetpp \
    --datasets arkitscenes:/upfs/enhan/data/processed_data/ARKitScenes \
    --split_type train \
    --output_dir data/qa_output \
    --llm_backend vllm --llm_model /upfs/models/Qwen/Qwen3-32B \
    --llm_base_url http://localhost:8001/v1 \
    --use_llm_mc
    2>&1 | tee -a /upfs/enhan/data/processed_data/ScanNet/code_qa_output/code_qa_generator.log

python -m utils.format_qa \
    --datasets scannet:data/qa_output/train/qa_code_generated_scannet.json:/upfs/enhan/data/processed_data/ScanNet/color/train \
    --datasets scannetpp:data/qa_output/train/qa_code_generated_scannetpp.json:/upfs/enhan/data/processed_data/ScanNetpp/color/train \
    --datasets arkitscenes:data/qa_output/train/qa_code_generated_arkitscenes.json:/upfs/enhan/data/processed_data/ARKitScenes/color/train \
    --output_path data/qa_output/train_grpo.json --shuffle