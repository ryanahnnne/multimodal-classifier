set -e

# === Multimodal ===

# ft_layer=all, mean vs attn
echo "ft_layer=all, mean vs attn"
CUDA_VISIBLE_DEVICES=0 python train.py +experiment=directFT_lightning_1_allLayers_mean &
CUDA_VISIBLE_DEVICES=3 python train.py +experiment=directFT_lightning_2_allLayers_attn &
wait


