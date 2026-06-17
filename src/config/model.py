from copy import deepcopy


_VMGGA_MODEL_CONFIG = {
    "backbone_type": "ResNetFPN",
    "dinov3_name": "dinov3_vitb16",
    "resolution": (8, 2),
    "fine_window_size": 5,
    "fine_concat_coarse_feat": True,
    "resnetfpn": {"initial_dim": 128, "block_dims": [128, 196, 256]},
    "coarse": {
        "d_model": 256,
        "d_ffn": 256,
        "nhead": 8,
        "layer_names": ["self", "cross"] * 4,
        "attention": "linear",
        "temp_bug_fix": True,
    },
    "fine": {
        "d_model": 128,
        "d_ffn": 128,
        "nhead": 8,
        "layer_names": ["self", "cross"],
        "attention": "linear",
    },
    "match_coarse": {
        "thr": 2e-5,
        "border_rm": 2,
        "match_type": "dual_softmax",
        "dsmax_temperature": 0.1,
        "train_coarse_percent": 0.2,
        "train_pad_num_gt_min": 200,
        "sparse_spvs": True,
    },
}


def get_vmgga_model_config(match_threshold=None, fine_window_size=None):
    config = deepcopy(_VMGGA_MODEL_CONFIG)
    if match_threshold is not None:
        config["match_coarse"]["thr"] = float(match_threshold)
    if fine_window_size is not None:
        config["fine_window_size"] = int(fine_window_size)
    return config
