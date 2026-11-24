# System Dependency Graph

```mermaid
graph TD
    subgraph Configuration
        default_config["configs/default_config.yaml"] --> config["src/config.py"]
        config --> constants["src/constants.py"]
    end

    subgraph Core Modules
        constants --> inference["src/inference.py"]
        config --> inference
        constants --> visualization["src/visualization.py"]
        config --> corruptions["src/corruptions.py"]
        constants --> metrics["src/metrics.py"]
        constants --> dataset["src/dataset.py"]
    end

    subgraph Processing Pipelines
        inference --> video_processor["src/video_processor.py"]
        visualization --> video_processor
        visualization --> inference
    end

    subgraph Application & CLI
        inference --> app["app.py (Streamlit UI)"]
        video_processor --> app
        corruptions --> app
        visualization --> app
        metrics --> app

        inference --> eval_script["scripts/evaluate.py"]
        metrics --> eval_script
        dataset --> eval_script

        inference --> robust_script["scripts/robustness_test.py"]
        corruptions --> robust_script
        metrics --> robust_script
    end

    subgraph External Dependencies
        torch["PyTorch 2.15.0+cpu"] --> inference
        transformers["HuggingFace Transformers 4.57.6"] --> inference
        cv2["OpenCV 5.0.0"] --> video_processor
        cv2 --> corruptions
        cv2 --> visualization
        imageio_ffmpeg["imageio-ffmpeg 0.8.1"] --> video_processor
        streamlit["Streamlit 1.56.0"] --> app
    end
```

## Module Couplings & Invariants
1. `src/constants.py` is the single source of truth for the 19 Cityscapes evaluation classes and their RGB palette colors.
2. `src/config.py` provides lazy singleton access with support for environment overrides (`HF_HOME`, `TORCH_HOME`).
3. `src/inference.py` handles input dimension validation, model forward passes, and bilinear upsampling to input resolution.
4. `src/video_processor.py` depends on `imageio_ffmpeg` for FFmpeg binary location; imported lazily in `src/__init__.py`.
5. `src/metrics.py` implements pure NumPy confusion matrix accumulation with zero PyTorch dependency.
