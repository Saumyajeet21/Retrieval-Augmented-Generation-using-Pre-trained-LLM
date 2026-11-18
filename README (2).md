# Text-to-Image Generation using Stable Diffusion

**Generative AI — Practical No. 05**

This project uses a pretrained **Stable Diffusion** model with the **Hugging Face Diffusers** library to generate images from text descriptions. It investigates how **inference steps**, **classifier-free guidance (CFG) scale**, **sampling methods**, and **prompt specificity** affect the generated results. It also evaluates prompt–image alignment using **CLIP similarity**.

## Google Colab Notebook

[**Open the project in Google Colab**](https://colab.research.google.com/drive/1dnHIXL6XGyIGzYoAVCv9yVAtzdTWTH5L?usp=sharing)

## Objective

- Generate an image from a natural-language prompt using a pretrained diffusion model.
- Compare output quality and generation time at different inference step counts.
- Study how CFG scale changes prompt adherence and visual appearance.
- Compare image generation with different schedulers.
- Test prompts of varying specificity.
- Measure text–image similarity with a pretrained CLIP model.

## Technology Stack

| Tool / Library | Purpose |
| --- | --- |
| Python | Implementation |
| Google Colab (GPU) | Notebook execution |
| PyTorch | Model inference |
| Hugging Face Diffusers | Stable Diffusion pipeline and schedulers |
| `segmind/tiny-sd` | Lightweight pretrained Stable Diffusion model |
| Hugging Face Transformers | CLIP evaluation |
| Matplotlib | Image galleries and charts |
| Pandas | Comparison tables |

## How It Works

1. A text prompt is tokenized and encoded into text embeddings.
2. The diffusion model begins with a random latent noise representation.
3. The U-Net and scheduler iteratively denoise the latent representation, conditioned on the text prompt.
4. A VAE decoder converts the final latent into an image.
5. Images are compared visually and using runtime measurements and CLIP similarity.

**Pipeline:** `Text Prompt → Text Encoder → Denoising (U-Net + Scheduler) → VAE Decoder → Generated Image`

## Example Prompt

```python
PROMPT = (
    "A futuristic glass library in a lush forest, solar panels on the roof, "
    "wooden walkway, golden morning light, cinematic architectural photography, "
    "highly detailed"
)
NEGATIVE = "blurry, low resolution, distorted architecture, text, watermark"
SEED = 123
```

The baseline image uses **25 inference steps**, **CFG scale 7.5**, and a fixed random seed for reproducible comparisons. Images are generated at **384 × 384** resolution.

## Experiments

| Experiment | Settings tested | What is compared |
| --- | --- | --- |
| Inference steps | 8, 16, 28, 40 | Detail, consistency, generation time |
| CFG guidance scale | 1.0, 3.5, 7.5, 11.0, 16.0 | Prompt adherence, contrast, artifacts |
| Sampling methods | DDIM, Euler, Euler Ancestral, DPM-Solver++ | Visual differences and runtime |
| Prompt specificity | Short, descriptive, and detailed prompts | Effect of text description on output |
| CLIP similarity | Generated images from the comparisons | Semantic alignment with the prompt |

### CLIP Evaluation

The notebook uses `openai/clip-vit-base-patch32` to calculate **cosine similarity × 100** between image and text embeddings. This is a relative similarity measure, **not** a percentage of image quality or a comprehensive assessment of realism.

## Run in Google Colab

1. Open the [Colab notebook](https://colab.research.google.com/drive/1dnHIXL6XGyIGzYoAVCv9yVAtzdTWTH5L?usp=sharing).
2. Choose **Runtime → Change runtime type → T4 GPU** (or another available CUDA GPU).
3. Run the installation cell, then execute the notebook cells in order.
4. Review the generated images, comparison tables, timing charts, and CLIP similarity plots.
5. Run the optional save-results cell to write images and timing CSV files to `stable_diffusion_results/`.

**Main dependencies:**

```bash
pip install "diffusers>=0.30,<1" "transformers>=4.40,<5" accelerate safetensors matplotlib pandas
```

PyTorch is also required and is typically preinstalled in Google Colab. GPU availability and library versions can affect execution time and reproducibility.

## Results and Observations

- **Inference steps:** The generated library remains broadly consistent across the tested settings. More steps generally increase runtime, while visual gains may diminish.
- **Guidance scale:** Low guidance produces less prompt-specific architecture; moderate values better preserve the intended glass library. Very high values can exaggerate contrast or alter details.
- **Sampling methods:** DDIM, Euler, Euler Ancestral, and DPM-Solver++ produce different structures and textures even with comparable settings.
- **Prompt specificity:** More descriptive prompts provide more control over the building, forest setting, and lighting.
- **CLIP similarity:** The comparison provides an additional numerical indication of text–image alignment, but should be interpreted alongside visual inspection.

The actual notebook outputs include image grids, a steps-versus-runtime plot, and CLIP similarity bar charts. **Runtime values are hardware-dependent.**

## Project Files

```text
.
├── README.md
├── stable_diffusion_colab_practical.ipynb   # Add/rename your notebook here
└── stable_diffusion_results/               # Optional generated outputs
```

## Conclusion

This practical demonstrates text-to-image generation using a pretrained latent diffusion model. The experiments show that inference steps, CFG guidance, sampling method, and prompt wording all influence the result. A moderate step count and guidance value can provide a useful balance between image quality, prompt alignment, and runtime. CLIP similarity supplements—but does not replace—visual evaluation.

## References

- [Hugging Face Diffusers documentation](https://huggingface.co/docs/diffusers/index)
- [Segmind Tiny-SD model](https://huggingface.co/segmind/tiny-sd)
- [CLIP ViT-B/32 model](https://huggingface.co/openai/clip-vit-base-patch32)
