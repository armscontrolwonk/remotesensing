# SuperDove RGB Super-Resolution

Per-pair super-resolution RGB synthesis for Planet's 8-band SuperDove imagery.
Fuses spectrally adjacent band pairs into three sharper RGB channels by
exploiting the small sub-pixel offsets between bands that survive L1B
geometric correction.

## What this does

SuperDove acquires its eight bands on separate detector rows of the same focal
plane. The result is that each band sees the same ground point with a small
(fraction-of-a-pixel) spatial offset from its neighbors. This pipeline trains
one super-resolution network per RGB output channel, each taking two
spectrally adjacent SuperDove bands as input and producing one 2x
super-resolved channel:

| Output | Default pair          | Alternative          |
|--------|-----------------------|----------------------|
| Blue   | Coastal Blue + Blue   | —                    |
| Green  | Green I + Green       | —                    |
| Red    | Red + Yellow          | Red + Red Edge (`--red-edge`) |

`Red + Yellow` keeps vegetation chromatically natural. `Red + Red Edge` pulls
in vegetation reflectance — sharper-looking foliage, but trees and grass shift
toward magenta/orange.

## Architecture

- **Backbone:** RRDBNet (the ESRGAN / Real-ESRGAN generator), modified for
  2-channel input and 1-channel output, 2x upscale by default.
- **Training data:** Wald's protocol — real SuperDove scenes are degraded by
  Gaussian blur + 2x decimation to produce LR inputs; the HR target is the
  per-pixel average of the two source bands at native resolution. Random
  sub-pixel jitter is applied to one band of the LR input each sample so the
  network learns to use inter-band offsets at inference time.
- **Radiometric handling:** for Analytic Radiance (TOAR) products, per-band
  ``reflectanceCoefficient`` values are parsed from the Planet
  ``<stem>_metadata.xml`` sidecar and applied on read, so the network always
  sees data in a uniform TOA-reflectance [0, 1] frame regardless of
  acquisition geometry. Set ``product: sr`` in the config if you're using
  Surface Reflectance products instead — those use the constant 1/10000
  scaling and don't need an XML sidecar.
- **Loss:** L1 + LPIPS (perceptual). LPIPS is what buys the visual sharpness;
  pure L1 produces soft output.
- **Inference:** tile-based with Hann-window blending to suppress seams; an
  optional per-channel CDF match against the original downsampled RGB
  suppresses any color drift from the LPIPS term.

## Honest caveats

Wald's-protocol training only ever teaches the network to recover detail that
exists in the *original* SuperDove pixels — the ceiling is set by the input
resolution. The output looks crisper because (a) it actually exploits the
inter-band sub-pixel info and (b) LPIPS pushes toward perceptually plausible
texture, but it cannot invent ground-truth structure smaller than ~3m.

To break that ceiling you need a higher-resolution supervision target. The
practical option is **SkySat** (~0.5m): collocate SkySat acquisitions with the
matching SuperDove pair and train the network to recover the SkySat detail
from the SuperDove pair. The dataset class can be swapped for a paired
SkySat→SuperDove version without touching the model or training loop.

## Layout

```
src/superres/
  bands.py          # SuperDove band names + RGB pair definitions
  io.py             # rasterio I/O, scaled transforms
  registration.py   # sub-pixel phase correlation, coarse alignment
  data.py           # Wald's-protocol Dataset
  model.py          # RRDBNet (2→1 channel, 2x default)
  losses.py         # L1 + LPIPS bundle
  train.py          # single-pair training entry point
  infer.py          # 8-band scene → RGB GeoTIFF
configs/default.yaml
scripts/train_all.sh
tests/test_smoke.py
```

## Quick start

```bash
pip install -e .

# Put your 8-band SuperDove scenes under data/superdove/*.tif
# Each scene needs its <stem>_metadata.xml sidecar alongside it (Planet ships
# this by default). Edit configs/default.yaml if you have SR products instead.

# Train all three models (CPU-runnable for smoke tests, GPU for real)
bash scripts/train_all.sh configs/default.yaml

# Inference (TOAR is the default; pass --product sr for Surface Reflectance)
python -m superres.infer \
    --scene data/superdove/AOI_0001.tif \
    --blue checkpoints/blue/best.pt \
    --green checkpoints/green/best.pt \
    --red checkpoints/red/best.pt \
    --out out_rgb.tif
```

## Which Planet product to order

From Planet's order builder, pick **Analytic Radiance (TOAR) – 8 band**. The
Surface Reflectance 8-band variant goes through atmospheric correction and
additional resampling, both of which smear out the sub-pixel inter-band
offsets this pipeline depends on. TOAR is closer to raw — fewer processing
steps between the detector and the file — so band-to-band geometry is better
preserved. (If you have API access to the L1B Basic Scene product, that's
even more raw.)

## Tests

```bash
pip install -e .[dev]
pytest tests/ -v
```

The smoke tests synthesize fake 8-band rasters so they run without real Planet
data.

## Next steps to push quality further

- **GAN loss.** Add a PatchGAN discriminator + adversarial term once L1+LPIPS
  pre-training has converged. This is where Real-ESRGAN-style perceptual
  sharpness comes from.
- **SkySat supervision.** Swap `SuperDoveWaldDataset` for a paired dataset
  that uses collocated SkySat as HR truth. This is the only way to genuinely
  exceed the 3m input resolution.
- **Offset conditioning.** Pass the residual sub-pixel offset (returned by
  `coarse_align`) as a side input — concat as constant channels, or use FiLM
  conditioning — so the network can reason about offset magnitude explicitly.
- **Joint model.** Replace the three per-pair models with a single 6→3 channel
  network that sees all six bands at once. Lets the network share features
  across channels and runs faster at inference; harder to debug per channel.
