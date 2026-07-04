# Random Palletizing Submission

This package is intended to be extracted and executed from its root directory:

```bash
python main.py
```

The evaluator should provide input JSON files under `box_sequence/`. The program
writes outputs under `algorithm_results/`.

## Contents

- `main.py`: official execution entry point
- `buffer_manager.py`: sliding-window buffer helper
- `algorithm.py`: participant palletizing algorithm
- `visualize.py`: result visualization helper
- `config/algorithm_config.yaml`: input/output, pallet, and buffer settings
- `config/pct_config.yaml`: ONNX/PCT inference settings
- `src/models/pct_model.onnx`: trained policy exported to ONNX
- `src/pct/`: deterministic PCT packing and stability helper code

## Notes

The algorithm uses only the current box/current buffer passed by the execution
loop. It does not inspect future boxes outside the buffer. The submitted ZIP
does not include local result JSON files, screenshots, simulator outputs,
training logs, or PyTorch checkpoints.
