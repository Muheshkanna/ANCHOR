"""Unified model loader for TorchScript and ONNX models.

Exposes a standard prediction interface and identifies whether white-box
access (direct tensor manipulation and forward hooks) is supported.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort


class LoadedModel:
    """Wrapper that abstracts TorchScript and ONNX inference pipelines.

    Attributes
    ----------
    model_path : Path
        Path to the loaded model.
    is_white_box : bool
        True if the model is TorchScript, allowing layer hooks and gradients.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.is_white_box = False
        self.torch_model: torch.jit.ScriptModule | None = None
        self.ort_session: ort.InferenceSession | None = None

        if self.model_path.suffix == ".pt":
            self.torch_model = torch.jit.load(str(self.model_path))
            self.torch_model.eval()
            self.is_white_box = True
        elif self.model_path.suffix == ".onnx":
            # Avoid excessive logging in ONNX runtime
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self.ort_session = ort.InferenceSession(str(self.model_path), opts)
            self.is_white_box = False
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run model inference.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (B, C, H, W) normalized according to model expectations.

        Returns
        -------
        np.ndarray
            Model logits output of shape (B, num_classes).
        """
        if self.is_white_box:
            assert self.torch_model is not None
            with torch.no_grad():
                # Ensure input is on correct device if model has been placed on GPU
                # (defaulting to CPU for now)
                x_tensor = torch.tensor(x, dtype=torch.float32)
                logits = self.torch_model(x_tensor)
                return logits.cpu().numpy()
        else:
            assert self.ort_session is not None
            input_name = self.ort_session.get_inputs()[0].name
            outputs = self.ort_session.run(None, {input_name: x.astype(np.float32)})
            return outputs[0]
