
# metauas on mvtec with 256x256 inputs
model_name='weights/metauas-256.ckpt'
test_json='./data/MVTec-AD/test.json'
image_dir='../../datasets/mvtec/'
for seed in 1 3 5 7 9
do
      prompt_json="./data/MVTec-AD/oneprompt_seed${seed}.json"
      CUDA_VISIBLE_DEVICES="0" python3 test.py \
            --checkpoint $model_name  \
            --seed $seed \
            --img_size 256 \
            --image_dir $image_dir \
            --test_json $test_json \
            --prompt_json $prompt_json 
done
# metauas-star on mvtec with 256x256 inputs
prompt_json='./data/MVTec-AD/test-train-top10pair-eb4.json'
CUDA_VISIBLE_DEVICES="0" python3 test.py \
            --checkpoint $model_name  \
            --seed $seed \
            --img_size 256 \
            --image_dir $image_dir \
            --test_json $test_json \
            --prompt_json $prompt_json 





# metauas on visa with 512x512 inputs
model_name='weights/metauas-512.ckpt'
test_json='./data/VisA-AD/test.json'
image_dir='../../datasets/visa/'
for seed in 1 3 5 7 9
do
      prompt_json="./data/VisA-AD/oneprompt_seed${seed}.json"
      CUDA_VISIBLE_DEVICES="0" python3 test.py \
            --checkpoint $model_name  \
            --seed $seed \
            --img_size 512 \
            --image_dir $image_dir \
            --test_json $test_json \
            --prompt_json $prompt_json 
done
# metauas-star on visa with 512x512 inputs
prompt_json='./data/VisA-AD/test-train-top10pair-eb4.json'
CUDA_VISIBLE_DEVICES="0" python3 test.py \
            --checkpoint $model_name  \
            --seed $seed \
            --img_size 512 \
            --image_dir $image_dir \
            --test_json $test_json \
            --prompt_json $prompt_json 
      

# metauas on goodsad with 512x512 inputs
model_name='weights/metauas-512.ckpt'
test_json='./data/GoodsAD/test.json'
image_dir='/home/tione/notebook/datasets/LSAD/GoodsAD/'
for seed in 1 3 5 7 9
do
      prompt_json="./data/GoodsAD/oneprompt_seed${seed}.json"
      CUDA_VISIBLE_DEVICES="0" python3 test.py \
            --checkpoint $model_name  \
            --seed $seed \
            --img_size 512 \
            --image_dir $image_dir \
            --test_json $test_json \
            --prompt_json $prompt_json 
done
# metauas-star on goodsad with 512x512 inputs
prompt_json='./data/GoodsAD/test-train-top10pair-eb4.json'
CUDA_VISIBLE_DEVICES="0" python3 test.py \
            --checkpoint $model_name  \
            --seed $seed \
            --img_size 512 \
            --image_dir $image_dir \
            --test_json $test_json \
            --prompt_json $prompt_json 