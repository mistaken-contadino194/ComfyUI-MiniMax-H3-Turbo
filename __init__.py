"""ComfyUI nodes for the MiniMax-H3 Turbo LoRA (4-step audio-video).

Drops into the stock MiniMax-H3 workflow (t2v and i2v):

  MiniMaxH3TurboLoRA    MODEL -> MODEL   applies the turbo LoRA
  MiniMaxH3TurboSampler       -> SAMPLER 4-step sampler for SamplerCustomAdvanced

The sampler steps the video and audio streams on their own flow schedules
(video shift 12, audio shift 3). A stock single-schedule sampler over-steps the
audio at 4 steps and the audio breaks; this one keeps each stream on its own
clock so 4 steps stays clean.
"""

import math
import os

import torch
import torch.nn.functional as F
from tqdm.auto import trange

import comfy.samplers
import comfy.lora
import comfy.weight_adapter
import comfy.utils
import comfy.patcher_extension
import folder_paths

SHIFT_V, SHIFT_A = 12.0, 3.0


def _time_shift_sigma(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _time_shift_slope(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return (to * (1.0 + (fr - 1.0) * base) ** 2) / (fr * (1.0 + (to - 1.0) * base) ** 2)


def _audio_sigma(sv):
    return _time_shift_sigma(sv, SHIFT_V, SHIFT_A)


def _audio_slope(sv):
    return _time_shift_slope(sv, SHIFT_V, SHIFT_A)


def _latent_shapes(model):
    """[video_shape, audio_shape] the sampler is packing over — video latent is
    flattened first, then audio, so we need the split point."""
    guider = getattr(model, "inner_model", model)
    conds = getattr(guider, "conds", None)
    if conds:
        for cond_list in conds.values():
            for c in (cond_list or []):
                mc = c.get("model_conds", {}) if isinstance(c, dict) else {}
                if "latent_shapes" in mc:
                    return mc["latent_shapes"].cond
    return None


@torch.no_grad()
def _turbo_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                   **kwargs):
    extra_args = {} if extra_args is None else extra_args
    shapes = _latent_shapes(model)
    if not shapes or len(shapes) < 2:
        raise RuntimeError(
            "MiniMaxH3TurboSampler expects the MiniMax-H3 video+audio latent "
            "(the EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo output).")
    v_numel = math.prod(shapes[0][1:])           # flat pack is [video | audio]
    a_numel = (x.shape[-1] - v_numel)
    print(f"[H3TURBO sampler] sigmas={[round(float(s),4) for s in sigmas]}  "
          f"x.shape={tuple(x.shape)} dtype={x.dtype}  v_numel={v_numel} a_numel={a_numel}  "
          f"shapes={shapes}", flush=True)
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):   # tqdm it/s bar, like stock
        sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        out = (x - denoised) / sigmas[i]
        xv, ov = x[..., :v_numel], out[..., :v_numel]
        xa, oa = x[..., v_numel:], out[..., v_numel:]
        xv = xv + (sv_n - sv) * ov               # video on its own sigma
        sl = _audio_slope(max(sv, 1e-6))
        xa = xa + (_audio_sigma(sv_n) - _audio_sigma(sv)) * (oa / sl)  # audio clock
        x = torch.cat([xv, xa], dim=-1)
        _rms = lambda t: float(t.float().pow(2).mean().sqrt())
        print(f"[H3TURBO step {i}] sv={sv:.4f}->{sv_n:.4f}  denoised_rms={_rms(denoised):.4f}  "
              f"video: x_rms={_rms(xv):.4f} v_rms={_rms(ov):.4f}  "
              f"audio: x_rms={_rms(xa):.4f} v_rms={_rms(oa):.4f} slope={sl:.4f}", flush=True)
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x,
                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})
    return x


# --- pruned / curve-mode base support -------------------------------------
# Pruned H3 checkpoints replace the time embedder + full-width adaln with a small
# 8-dim curve (adaln_t_table), so the LoRA's adaln update (which lives in the
# 2688-dim silu(t_emb) space) can't be applied as a weight patch. Instead we
# re-inject it at run time: a shared silu(t_emb) is interpolated from a bundled
# grid each forward, and each adaln projection adds B @ A @ silu(t_emb) to its
# output. The backbone (attn/mlp/refiner) is patched normally.

_EGRID = None


def _egrid():
    global _EGRID
    if _EGRID is None:
        p = os.path.join(os.path.dirname(__file__), "h3_silu_temb_grid.safetensors")
        _EGRID = comfy.utils.load_torch_file(p)["silu_t_emb_grid"]   # [1025, 2688]
    return _EGRID


def _unique_t(timestep, shift_v, shift_a, has_vis_cond):
    sv = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
    t_v = 1.0 - sv
    t_a = 1.0 - _time_shift_sigma(sv, shift_v, shift_a)
    s = {t_v, t_a}
    if has_vis_cond:
        s.add(max(t_v, 0.999))
    return sorted(s)


def _interp_egrid(unique_t, E, device, dtype):
    E = E.to(device)
    n = E.shape[0]
    rows = []
    for t in unique_t:
        pos = min(max(t, 0.0), 1.0) * (n - 1)
        i0 = min(int(math.floor(pos)), n - 2)
        rows.append(torch.lerp(E[i0].float(), E[i0 + 1].float(), pos - i0))
    return torch.stack(rows).to(dtype)                               # [M, 2688]


class _AdalnDelta(torch.nn.Module):
    """Curve-mode AdalnProj wrapper: adds B @ A @ silu(t_emb) to the projection
    output before the reference view/chunk."""

    def __init__(self, base, a, b, shared):
        super().__init__()
        self.base = base
        self.register_buffer("a", a, persistent=False)
        self.register_buffer("b", b, persistent=False)
        self.shared = shared

    def forward(self, t_emb):
        base = self.base
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        st = self.shared.get("silu_temb")
        if st is not None:
            # keep every operand on x's device/dtype — under VRAM-offload flags the
            # object-patch buffers can be stranded on CPU while x is on the GPU.
            a = self.a.to(x.device, x.dtype)
            b = self.b.to(x.device, x.dtype)
            st = st.to(x.device, x.dtype)
            x = x + (b @ (a @ st.T)).T                                # [M, out]
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)


class MiniMaxH3TurboSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "MiniMaxH3Turbo"
    DESCRIPTION = ("4-step sampler for the MiniMax-H3 Turbo LoRA. Feed into "
                   "SamplerCustomAdvanced and set the scheduler to 4 steps.")

    def get_sampler(self):
        return (comfy.samplers.KSAMPLER(_turbo_sampler),)


def _apply_bypass_lora(new_model, lora, modules, strength):
    """Apply the low-rank update at RUN TIME (output = base(x) + lora(x)) via
    ComfyUI's bypass injection, so it is never folded into the weights. The delta
    is tiny relative to the base weight — merging rounds it away in bf16 and
    requantizes it away in int8/fp8 — whereas bypass runs the base's own
    (possibly quantized) forward and adds the bf16 update in activation space,
    exactly like the standalone generate.py reference. The stock
    model_lora_keys_unet does not recognise the H3 lora naming, so build the key
    map directly (module -> diffusion_model.<module>.weight)."""
    key_map = {m: "diffusion_model.{}.weight".format(m) for m in modules}
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    manager = comfy.weight_adapter.BypassInjectionManager()
    sd_keys = set(new_model.model.state_dict().keys())
    n = 0
    for key, adapter in loaded.items():
        if (isinstance(adapter, comfy.weight_adapter.WeightAdapterBase)
                and key in sd_keys):
            manager.add_adapter(key, adapter, strength=strength)
            n += 1
    injections = manager.create_injections(new_model.model)
    if manager.get_hook_count() > 0:
        new_model.set_injections("bypass_lora", injections)
    return n


def _add_dbg_wrapper(new_model, dm, tag):
    """Observability: at diffusion-model forward time, log whether a lora'd
    module's forward has been taken over by the bypass hook (i.e. the lora is
    actually active this forward) plus the timestep and video/audio input rms.
    Only the first few calls are printed to avoid flooding."""
    st = {"n": 0}

    def wrap(executor, *args, **kwargs):
        if st["n"] < 6:
            st["n"] += 1
            try:
                m0 = dm.blocks[0].attn.qkv_proj
                owner = type(getattr(m0.forward, "__self__", None)).__name__
            except Exception as e:                       # noqa
                owner = "err:%s" % e
            ts = args[1] if len(args) > 1 else kwargs.get("timestep")
            xx = args[0] if args else kwargs.get("x")
            try:
                vr = float(xx[0].float().pow(2).mean().sqrt())
                ar = float(xx[1].float().pow(2).mean().sqrt())
                dt = str(xx[0].dtype)
            except Exception:
                vr = ar = -1.0
                dt = "?"
            tsv = float(ts.flatten()[0]) if ts is not None else -1
            print(f"[H3TURBO fwd {tag}] call#{st['n']}  qkv_proj.forward_owner={owner} "
                  f"(BypassForwardHook => lora ACTIVE; else => BASE ONLY!)  is_injected="
                  f"{getattr(new_model, 'is_injected', '?')}  timestep={tsv:.2f}  "
                  f"video_rms={vr:.4f} audio_rms={ar:.4f} dtype={dt}", flush=True)
        return executor(*args, **kwargs)

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo_dbg", wrap)


class MiniMaxH3TurboLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                                   "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_lora"
    CATEGORY = "MiniMaxH3Turbo"
    DESCRIPTION = "Apply the MiniMax-H3 Turbo LoRA to the H3 diffusion model."

    def apply_lora(self, model, lora_name, strength):
        path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        dm = model.model.diffusion_model
        pruned = getattr(dm, "use_adaln_curves", False)
        modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora})
        new_model = model.clone()

        if not pruned:
            n = _apply_bypass_lora(new_model, lora, modules, strength)
            injs = new_model.injections.get("bypass_lora", [])
            try:
                p0 = dm.blocks[0].attn.qkv_proj.weight
                wdt, wdev = str(p0.dtype), str(p0.device)
            except Exception:
                wdt, wdev = "?", "?"
            print(f"[MiniMaxH3TurboLoRA] full base: lora={lora_name} strength={strength} "
                  f"| {len(modules)} lora modules, {n} adapters bound, "
                  f"{len(injs)} bypass injections set | model={type(new_model.model).__name__} "
                  f"weight_dtype={wdt} weight_dev={wdev}", flush=True)
            _add_dbg_wrapper(new_model, dm, "full")
            return (new_model,)

        # pruned/curve base: run-time bypass for the backbone (same as the full
        # base), and a custom E-grid injection for the adaln update — the pruned
        # base has no full-width adaln to attach a standard adapter to.
        backbone = [m for m in modules if "adaln_proj" not in m]
        adaln = [m for m in modules if "adaln_proj" in m]
        n = _apply_bypass_lora(new_model, lora, backbone, strength)

        E = _egrid()
        shared = {"silu_temb": None}
        shift_v = float(getattr(dm, "sigma_shift_video", SHIFT_V))
        shift_a = float(getattr(dm, "sigma_shift_audio", SHIFT_A))

        def wrap(executor, *args, **kwargs):
            ts = args[1] if len(args) > 1 else kwargs.get("timestep")
            ctx = args[2] if len(args) > 2 else kwargs.get("context")
            payload = kwargs.get("minimax_payload") or {}
            has_vc = bool(payload.get("keyframes") or payload.get("refs"))
            us = _unique_t(ts, shift_v, shift_a, has_vc)
            shared["silu_temb"] = _interp_egrid(us, E, ctx.device, ctx.dtype)
            return executor(*args, **kwargs)

        new_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo", wrap)
        for name in adaln:                       # name = "....adaln_proj.linear"
            a = lora[name + ".lora_A.weight"]
            b = lora[name + ".lora_B.weight"] * strength
            key = "diffusion_model." + name.rsplit(".linear", 1)[0]
            new_model.add_object_patch(key, _AdalnDelta(
                new_model.get_model_object(key), a, b, shared))
        print(f"[MiniMaxH3TurboLoRA] pruned base: {n} backbone via bypass + "
              f"{len(adaln)} adaln injected at run time", flush=True)
        _add_dbg_wrapper(new_model, dm, "pruned")
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TurboLoRA": MiniMaxH3TurboLoRA,
    "MiniMaxH3TurboSampler": MiniMaxH3TurboSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TurboLoRA": "MiniMax-H3 Turbo LoRA",
    "MiniMaxH3TurboSampler": "MiniMax-H3 Turbo Sampler (4-step)",
}
