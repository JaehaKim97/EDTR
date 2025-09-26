import os, warnings
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
