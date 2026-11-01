import torch
import transformers
import shap
import lime
import sklearn

print("=" * 45)
print("ENVIRONMENT CHECK")
print("=" * 45)

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Scikit-learn:", sklearn.__version__)
print("SHAP:", shap.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Training device: CPU")

print("\nAll environment checks completed!")