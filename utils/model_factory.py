from model import (
    PointNetCpRegressor,          # Original
    MLPNoGlobalCpRegressor,       # Ablated version
    MLPWithParamsCpRegressor,     # MLP + 16 params
    PointNetWithParamsCpRegressor # PointNet + 16 params
)

MODEL_REGISTRY = {
    "pointnet": PointNetCpRegressor,
    "mlp_no_global": MLPNoGlobalCpRegressor,
    "mlp_with_params": MLPWithParamsCpRegressor,
    "pointnet_with_params": PointNetWithParamsCpRegressor,
}

# Models that require the 16 geometric parameters
MODELS_WITH_PARAMS = {"mlp_with_params", "pointnet_with_params"}

def get_model(model_name, input_dim=6, num_params=16, hidden=128):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    model_class = MODEL_REGISTRY[model_name]
    if model_name in MODELS_WITH_PARAMS:
        return model_class(input_dim=input_dim, num_params=num_params, hidden=hidden)
    else:
        return model_class(input_dim=input_dim, hidden=hidden)

