# ComfyUI-MiniMax-H3-Turbo

Run [MiniMax-H3](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) —
joint **video + synchronized audio** — in **4 sampling steps** instead of ~20,
with the [MiniMax-H3 Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).

Two nodes that drop straight into the official H3 workflow (text-to-video and
image-to-video):

| node | what it does |
|---|---|
| **MiniMax-H3 Turbo LoRA** | `MODEL → MODEL`, applies the turbo LoRA |
| **MiniMax-H3 Turbo Sampler (4-step)** | `→ SAMPLER`, feeds `SamplerCustomAdvanced` |

> ⚠️ **Early preview.** The current LoRA is an early checkpoint — quality is
> well short of a finished model, but already a clear step up from the base at
> 4 steps. See the LoRA repo for details.

## Install

**Via ComfyUI-Manager** — search "MiniMax-H3 Turbo" and install.

**Or manually:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo
```
Then restart ComfyUI.

**Download the LoRA** from
[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
and put the `.safetensors` into `ComfyUI/models/loras/`.

You also need the base MiniMax-H3 model, VAEs and text encoder from the official
release — see the [MiniMax-H3 tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3).

## Use

Start from the **official MiniMax-H3 workflow** (t2v or i2v) and make two changes:

1. Insert **MiniMax-H3 Turbo LoRA** between the model loader and the sampler
   (`... → Load Diffusion Model → MiniMax-H3 Turbo LoRA → SamplerCustomAdvanced`),
   and pick the turbo `.safetensors`.
2. Replace the sampler feeding `SamplerCustomAdvanced` with **MiniMax-H3 Turbo
   Sampler (4-step)**, and set the scheduler node to **4 steps**
   (`BasicScheduler`, scheduler `simple`).

Everything else — the conditioning nodes, VAE decode, audio output — stays as in
the official workflow, so **both text-to-video and image-to-video** work
unchanged. A ready-made t2v workflow is in
[`example_workflows/`](example_workflows/minimax_h3_t2v_turbo.json) — drag it
into ComfyUI to see the wiring.

**Base model**: works with any MiniMax-H3 base — full (`bf16`, `int8_convrot`)
**and the pruned/curve variants** (`pruned_int8`, `pruned_fp8`). The node detects
a pruned base automatically and re-injects the LoRA's time-conditioning at run
time (a small `silu(t_emb)` grid ships with the node for this), so one LoRA file
covers every base.

## Why a custom sampler

MiniMax-H3 denoises the video and audio streams on two different flow schedules
(video shift 12, audio shift 3). ComfyUI's stock samplers step both streams on
one schedule, which is fine at ~20 steps but badly over-steps the audio at 4
steps — the audio comes out distorted or blown out. This sampler steps each
stream on its own schedule, so audio stays clean at 4 steps. If you load the
LoRA and use a stock sampler at 4 steps and the audio is broken, this is why.

## Notes

- **Steps**: 4 works, but this early checkpoint is under-trained, so the
  **comfort zone for sharpness is 6–8 steps**, not 4 (4 comes out softer). Any
  count **≥ 4** is valid and **more steps look better**. Scheduler stays `simple`.
- **Resolution / length**: width and height are multiples of 32 (short edge
  typically 768); frame count is at 24 fps and snaps to the model's 17·k+5 grid
  (124 ≈ 5 s). Validated range ~124–362 frames.
- **VRAM**: MiniMax-H3 is large (~33 B); an 80 GB GPU is comfortable.

## License

Apache-2.0.
