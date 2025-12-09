## Theory

- [SAM2](https://github.com/facebookresearch/sam2) (Meta's Segment Anything Model 2, 39M-224M params): Meta’s first unified model that can segment objects in both images and videos in real time. Fine-tuning SAM2 allows you to adapt SAM2 to your specific needs, improving its accuracy and efficiency for your particular use case. As of Dec 2025, we already have [SAM3](https://github.com/facebookresearch/sam3)

- [LoRA](https://arxiv.org/abs/2106.09685): Research showed that weight changes during fine-tuning lie in a low-dimensional subspace. This means you don't need to update all dimensions - most of the "movement" happens in a small subset. 
    - When finetuning a model weight `W_pretrained`, we do `W_finetuned = W_pretrained + ΔW` where `ΔW` has very low "intrinsic rank" (most singular values are near zero, only ~8-64 dimensions actually matter). LoRA insight: `ΔW ≈ B × A` - we approximate `ΔW` with `B` and `A` where `A` is the "down-projection" matrix `(r × d_in), with r small (8-64)`, and `B` is the "up-projection" matrix `(d_out × r)`. So the total params to fine tune would be `r × d_in + d_out × r = r × (d_in + d_out)`
    - Original delta weight: `ΔW` e.g. `4096 × 4096 = 16M` params. With `r=16`: `A` is (`4096 × 16`) + `B` is (`16 × 4096`) = `131K` params (99.2% reduction)

- `fl-sam2-lora`: Finetuning SAM2 using LoRA technique in a federated setting where data never leaves hospitals, and since only LoRA adapters are transfered (instead of the full models), communication cost is very low