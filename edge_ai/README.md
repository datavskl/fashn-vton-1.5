# ShriVastra Edge AI — experimental baseline
Not production-ready. No trained checkpoint, licensed training dataset, or mobile application is included. This supervised RGB encoder-decoder is a research baseline, not a port of FASHN weights or a proven high-quality VTON system.

## Training on Lightning AI
From repository root in an isolated Python environment:
```sh
pip install -r edge_ai/requirements.txt
python -m pytest edge_ai/test_model.py
python -m edge_ai.train --train /private/train.jsonl --val /private/val.jsonl --output /private/student.pt
python -m edge_ai.export --weights /private/student.pt --output /private/student.onnx
```
Each JSONL row: {"person":"person.jpg","garment":"garment.jpg","target":"target.jpg","category":"tops"}. Paths are relative to manifest. Person is a source outfit, garment is the desired flat-lay item, target is the same consenting person wearing the desired item or a licensed teacher-generated result. Separate identities and garments between splits; reserve a blind test set. Do not train on customer photos without explicit authorization. Verify all dataset, teacher output and model redistribution rights. Never commit images or weights without review.

## Mobile tensor contract
Float32 NCHW [1,9,384,256]; EXIF-correct RGB images stretched to width 256 height 384, bilinear interpolation, [0,1]. Channels 0–2 person, 3–5 garment, 6–8 constant one-hot category maps (tops, bottoms, one-pieces). Output float32 [1,3,384,256] RGB [0,1]. Matching preprocessing across native platforms is essential.
Use ONNX Runtime Mobile CPU baseline; investigate Core ML and Android NNAPI or other supported acceleration providers, validate actual delegated operations; no GPU or NPU support is claimed. Consider quantization only after quality and performance measurements. Package signed, versioned weights with integrity verification and retain photos in private app storage. Offline mode must fail locally, never silently upload photographs.

## Release gates
1. Train with legitimate data and evaluate garment fidelity, identity retention, pose and failure cases; L1 alone may yield blurred output or copying. Add spatial conditioning, perceptual losses or distillation as needed.
2. Export parity test with trained checkpoint; verify output against Python reference on real inputs.
3. Build native React Native bridge, local image tensor preprocessing and result decoding. Benchmark actual iOS and Android phones in airplane mode.
4. Record cold/warm latency, peak memory, battery, thermal throttling, 10 repeated runs, crashes and provider fallback. Proposed target p95 <=15 seconds on selected devices, subject to measurement.
5. Compare against FASHN results with consented test data; publish supported device matrix only after acceptance.
6. Security/privacy review and signed model delivery.
Cloud backend remains unchanged; this branch does not require cloud inference at runtime once the mobile app and trained model exist.
