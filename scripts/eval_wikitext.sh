# !/bin/bash

export HF_MODEL=/group/archive_dataset_6_nobkup/archive_modelzoo/sequence_learning/weights/nlp-pretrained-model/meta-llama/Llama-3.2-1B
export EXP=outputs
export STEP=0
export OUTPUT_FOLDER=./$EXP
export TASK=wikitext

mkdir $OUTPUT_FOLDER/hf
python 3rdparty/torchtitan/scripts/checkpoint_conversion/convert_to_hf.py \
    $OUTPUT_FOLDER/checkpoint/step-$STEP \
    $OUTPUT_FOLDER/hf \
    --model_name llama3 --model_flavor 1B \
    --hf_assets_path $HF_MODEL \
    --export_dtype bfloat16
rm -rf $OUTPUT_FOLDER/hf/sharded
cp $HF_MODEL/*.json $OUTPUT_FOLDER/hf/

CUDA_VISIBLE_DEVICES=2 python -m lm_eval --model hf \
    --model_args pretrained=$OUTPUT_FOLDER/hf \
    --tasks $TASK \
    --device cuda:0 \
    --batch_size 8 2>&1 | tee $TASK-$EXP.log
