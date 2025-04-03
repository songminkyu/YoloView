op_map = {
    "abs": {
        "phi_name": "abs",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "accuracy": {
        "phi_name": "accuracy",
        "inputs": {
            "x": "Out",
            "indices": "Indices",
            "label": "Label"
        },
        "outputs": {
            "accuracy": "Accuracy",
            "correct": "Correct",
            "total": "Total"
        }
    },
    "acos": {
        "phi_name": "acos",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "acosh": {
        "phi_name": "acosh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "adadelta": {
        "phi_name": "adadelta_",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "avg_squared_grad": "AvgSquaredGrad",
            "avg_squared_update": "AvgSquaredUpdate",
            "learning_rate": "LearningRate",
            "master_param": "MasterParam"
        },
        "outputs": {
            "param_out": "ParamOut",
            "moment_out": "AvgSquaredGradOut",
            "inf_norm_out": "AvgSquaredUpdateOut",
            "master_param_out": "MasterParamOut"
        }
    },
    "adagrad": {
        "phi_name": "adagrad_",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "moment": "Moment",
            "learning_rate": "LearningRate",
            "master_param": "MasterParam"
        },
        "outputs": {
            "param_out": "ParamOut",
            "moment_out": "MomentOut",
            "master_param_out": "MasterParamOut"
        }
    },
    "adam": {
        "phi_name": "adam_",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "learning_rate": "LearningRate",
            "moment1": "Moment1",
            "moment2": "Moment2",
            "moment2_max": "Moment2Max",
            "beta1_pow": "Beta1Pow",
            "beta2_pow": "Beta2Pow",
            "master_param": "MasterParam",
            "skip_update": "SkipUpdate"
        },
        "outputs": {
            "param_out": "ParamOut",
            "moment1_out": "Moment1Out",
            "moment2_out": "Moment2Out",
            "moment2_max_out": "Moment2MaxOut",
            "beta1_pow_out": "Beta1PowOut",
            "beta2_pow_out": "Beta2PowOut",
            "master_param_out": "MasterParamOut"
        },
        "scalar": {
            "beta1": {
                "data_type": "float",
                "tensor_name": "Beta1Tensor"
            },
            "beta2": {
                "data_type": "float",
                "tensor_name": "Beta2Tensor"
            },
            "epsilon": {
                "data_type": "float",
                "tensor_name": "EpsilonTensor"
            }
        }
    },
    "adamax": {
        "phi_name": "adamax_",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "learning_rate": "LearningRate",
            "moment": "Moment",
            "inf_norm": "InfNorm",
            "beta1_pow": "Beta1Pow",
            "master_param": "MasterParam"
        },
        "outputs": {
            "param_out": "ParamOut",
            "moment_out": "MomentOut",
            "inf_norm_out": "InfNormOut",
            "master_param_out": "MasterParamOut"
        }
    },
    "adamw": {
        "phi_name": "adamw_",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "learning_rate": "LearningRate",
            "moment1": "Moment1",
            "moment2": "Moment2",
            "moment2_max": "Moment2Max",
            "beta1_pow": "Beta1Pow",
            "beta2_pow": "Beta2Pow",
            "master_param": "MasterParam",
            "skip_update": "SkipUpdate"
        },
        "outputs": {
            "param_out": "ParamOut",
            "moment1_out": "Moment1Out",
            "moment2_out": "Moment2Out",
            "moment2_max_out": "Moment2MaxOut",
            "beta1_pow_out": "Beta1PowOut",
            "beta2_pow_out": "Beta2PowOut",
            "master_param_out": "MasterParamOut"
        },
        "scalar": {
            "beta1": {
                "data_type": "float",
                "tensor_name": "Beta1Tensor"
            },
            "beta2": {
                "data_type": "float",
                "tensor_name": "Beta2Tensor"
            },
            "epsilon": {
                "data_type": "float",
                "tensor_name": "EpsilonTensor"
            }
        }
    },
    "elementwise_add": {
        "phi_name": "add",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "scale_x": "Scale_x",
            "scale_y": "Scale_y",
            "scale_out": "Scale_out"
        }
    },
    "sum": {
        "phi_name": "add_n",
        "inputs": {
            "inputs": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "add_position_encoding": {
        "phi_name": "add_position_encoding",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "addmm": {
        "phi_name": "addmm",
        "inputs": {
            "input": "Input",
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "alpha": "Alpha",
            "beta": "Beta"
        }
    },
    "affine_channel": {
        "phi_name": "affine_channel",
        "inputs": {
            "x": "X",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "affine_grid": {
        "phi_name": "affine_grid",
        "inputs": {
            "input": "Theta"
        },
        "outputs": {
            "output": "Output"
        },
        "int_array": {
            "output_shape": {
                "data_type": "int",
                "tensor_name": "OutputShape"
            }
        }
    },
    "reduce_all": {
        "phi_name": "all",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        }
    },
    "allclose": {
        "phi_name": "allclose",
        "inputs": {
            "x": "Input",
            "y": "Other"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "rtol": {
                "data_type": "std::string",
                "tensor_name": "Rtol"
            },
            "atol": {
                "data_type": "std::string",
                "tensor_name": "Atol"
            }
        }
    },
    "reduce_amax": {
        "phi_name": "amax",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        }
    },
    "reduce_amin": {
        "phi_name": "amin",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        }
    },
    "anchor_generator": {
        "phi_name": "anchor_generator",
        "inputs": {
            "input": "Input"
        },
        "outputs": {
            "anchors": "Anchors",
            "variances_out": "Variances"
        }
    },
    "angle": {
        "phi_name": "angle",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reduce_any": {
        "phi_name": "any",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        }
    },
    "range": {
        "phi_name": "arange",
        "inputs": {
            "start": "Start",
            "end": "End",
            "step": "Step"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "start": {
                "data_type": "double",
                "support_tensor": "True"
            },
            "end": {
                "data_type": "double",
                "support_tensor": "True"
            },
            "step": {
                "data_type": "double",
                "support_tensor": "True"
            }
        }
    },
    "arg_max": {
        "phi_name": "argmax",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "axis": {
                "data_type": "int64_t",
                "support_tensor": "True"
            }
        }
    },
    "arg_min": {
        "phi_name": "argmin",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "axis": {
                "data_type": "int64_t",
                "support_tensor": "True"
            }
        }
    },
    "argsort": {
        "phi_name": "argsort",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "indices": "Indices"
        }
    },
    "tensor_array_to_tensor": {
        "phi_name": "array_to_tensor",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "out_index": "OutIndex"
        }
    },
    "as_complex": {
        "phi_name": "as_complex",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "as_real": {
        "phi_name": "as_real",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "asin": {
        "phi_name": "asin",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "asinh": {
        "phi_name": "asinh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "assert": {
        "phi_name": "assert",
        "inputs": {
            "cond": "Cond",
            "data": "Data"
        }
    },
    "assign": {
        "phi_name": "assign",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "assign_pos": {
        "phi_name": "assign_pos",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "assign_value": {
        "phi_name": "assign_value",
        "outputs": {
            "out": "Out"
        }
    },
    "atan": {
        "phi_name": "atan",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "atan2": {
        "phi_name": "atan2",
        "inputs": {
            "x": "X1",
            "y": "X2"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "atanh": {
        "phi_name": "atanh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "attention_lstm": {
        "phi_name": "attention_lstm",
        "inputs": {
            "x": "X",
            "c0": "C0",
            "h0": "H0",
            "attention_weight": "AttentionWeight",
            "attention_bias": "AttentionBias",
            "attention_scalar": "AttentionScalar",
            "attention_scalar_bias": "AttentionScalarBias",
            "lstm_weight": "LSTMWeight",
            "lstm_bias": "LSTMBias"
        },
        "outputs": {
            "hidden": "Hidden",
            "cell": "Cell",
            "attentioned_x": "AttentionedX",
            "attention_fc_out": "AttentionFCOut",
            "lstm_x": "LSTMX",
            "lstm_out": "LSTMOUT"
        }
    },
    "auc": {
        "phi_name": "auc",
        "inputs": {
            "x": "Predict",
            "label": "Label",
            "stat_pos": "StatPos",
            "stat_neg": "StatNeg",
            "ins_tag_weight": "InsTagWeight"
        },
        "outputs": {
            "auc": "AUC",
            "stat_pos_out": "StatPosOut",
            "stat_neg_out": "StatNegOut"
        }
    },
    "baddbmm": {
        "phi_name": "baddbmm",
        "inputs": {
            "input": "Input",
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "alpha": "Alpha",
            "beta": "Beta"
        }
    },
    "barrier": {
        "phi_name": "barrier",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "batch_fc": {
        "phi_name": "batch_fc",
        "inputs": {
            "input": "Input",
            "w": "W",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "batch_norm": {
        "phi_name": "batch_norm",
        "inputs": {
            "x": "X",
            "mean": "Mean",
            "variance": "Variance",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Y",
            "mean_out": "MeanOut",
            "variance_out": "VarianceOut",
            "saved_mean": "SavedMean",
            "saved_variance": "SavedVariance",
            "reserve_space": "ReserveSpace"
        },
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "bce_loss": {
        "phi_name": "bce_loss",
        "inputs": {
            "input": "X",
            "label": "Label"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "beam_search_decode": {
        "phi_name": "beam_search_decode",
        "inputs": {
            "ids": "Ids",
            "scores": "Scores"
        },
        "outputs": {
            "sentence_ids": "SentenceIds",
            "sentence_scores": "SentenceScores"
        }
    },
    "bernoulli": {
        "phi_name": "bernoulli",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bicubic_interp_v2": {
        "phi_name": "bicubic_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        },
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "bilinear_tensor_product": {
        "phi_name": "bilinear",
        "inputs": {
            "x": "X",
            "y": "Y",
            "weight": "Weight",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bilinear_interp_v2": {
        "phi_name": "bilinear_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        },
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "bincount": {
        "phi_name": "bincount",
        "inputs": {
            "x": "X",
            "weights": "Weights"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "minlength": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "bipartite_match": {
        "phi_name": "bipartite_match",
        "inputs": {
            "dist_mat": "DistMat"
        },
        "outputs": {
            "col_to_row_match_indices": "ColToRowMatchIndices",
            "col_to_row_match_dist": "ColToRowMatchDist"
        }
    },
    "bitwise_and": {
        "phi_name": "bitwise_and",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bitwise_not": {
        "phi_name": "bitwise_not",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bitwise_or": {
        "phi_name": "bitwise_or",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bitwise_xor": {
        "phi_name": "bitwise_xor",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bmm": {
        "phi_name": "bmm",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bn_act_xpu": {
        "phi_name": "bn_act_xpu",
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "box_coder": {
        "phi_name": "box_coder",
        "inputs": {
            "prior_box": "PriorBox",
            "prior_box_var": "PriorBoxVar",
            "target_box": "TargetBox"
        },
        "outputs": {
            "output_box": "OutputBox"
        }
    },
    "broadcast_tensors": {
        "phi_name": "broadcast_tensors",
        "inputs": {
            "input": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "c_concat": {
        "phi_name": "c_concat",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "c_embedding": {
        "phi_name": "c_embedding",
        "inputs": {
            "weight": "W",
            "x": "Ids"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "c_softmax_with_cross_entropy": {
        "phi_name": "c_softmax_with_cross_entropy",
        "inputs": {
            "logits": "Logits",
            "label": "Label"
        },
        "outputs": {
            "softmax": "Softmax",
            "loss": "Loss"
        }
    },
    "c_split": {
        "phi_name": "c_split",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "cast": {
        "phi_name": "cast",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "ceil": {
        "phi_name": "ceil",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "celu": {
        "phi_name": "celu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "check_finite_and_unscale": {
        "phi_name": "check_finite_and_unscale_",
        "inputs": {
            "x": "X",
            "scale": "Scale"
        },
        "outputs": {
            "out": "Out",
            "found_infinite": "FoundInfinite"
        }
    },
    "cholesky": {
        "phi_name": "cholesky",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "cholesky_solve": {
        "phi_name": "cholesky_solve",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "class_center_sample": {
        "phi_name": "class_center_sample",
        "inputs": {
            "label": "Label"
        },
        "outputs": {
            "remapped_label": "RemappedLabel",
            "sampled_local_class_center": "SampledLocalClassCenter"
        }
    },
    "clip": {
        "phi_name": "clip",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "min": {
                "data_type": "float",
                "tensor_name": "Min"
            },
            "max": {
                "data_type": "float",
                "tensor_name": "Max"
            }
        }
    },
    "clip_by_norm": {
        "phi_name": "clip_by_norm",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "coalesce_tensor": {
        "phi_name": "coalesce_tensor",
        "inputs": {
            "input": "Input"
        },
        "outputs": {
            "output": "Output",
            "fused_output": "FusedOutput"
        },
        "attrs": {
            "size_of_dtype": "user_defined_size_of_dtype"
        }
    },
    "collect_fpn_proposals": {
        "phi_name": "collect_fpn_proposals",
        "inputs": {
            "multi_level_rois": "MultiLevelRois",
            "multi_level_scores": "MultiLevelScores",
            "multi_level_rois_num": "MultiLevelRoIsNum"
        },
        "outputs": {
            "fpn_rois": "FpnRois",
            "rois_num": "RoisNum"
        },
        "attrs": {
            "post_nms_topn": "post_nms_topN"
        }
    },
    "complex": {
        "phi_name": "complex",
        "inputs": {
            "real": "X",
            "imag": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "concat": {
        "phi_name": "concat",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "axis"
        },
        "scalar": {
            "axis": {
                "data_type": "int",
                "tensor_name": "AxisTensor"
            }
        }
    },
    "conditional_block": {
        "phi_name": "conditional_block"
    },
    "conj": {
        "phi_name": "conj",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "conv2d": {
        "phi_name": "conv2d",
        "inputs": {
            "input": "Input",
            "filter": "Filter"
        },
        "outputs": {
            "out": "Output"
        }
    },
    "conv2d_transpose": {
        "phi_name": "conv2d_transpose",
        "inputs": {
            "x": "Input",
            "filter": "Filter",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Output"
        },
        "int_array": {
            "output_size": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "conv2d_transpose_bias": {
        "phi_name": "conv2d_transpose_bias",
        "inputs": {
            "x": "Input",
            "filter": "Filter",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Output"
        },
        "int_array": {
            "output_size": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "conv3d": {
        "phi_name": "conv3d",
        "inputs": {
            "input": "Input",
            "filter": "Filter"
        },
        "outputs": {
            "out": "Output"
        }
    },
    "conv3d_transpose": {
        "phi_name": "conv3d_transpose",
        "inputs": {
            "x": "Input",
            "filter": "Filter"
        },
        "outputs": {
            "out": "Output"
        }
    },
    "correlation": {
        "phi_name": "correlation",
        "inputs": {
            "input1": "Input1",
            "input2": "Input2"
        },
        "outputs": {
            "out": "Output"
        }
    },
    "cos": {
        "phi_name": "cos",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "cosh": {
        "phi_name": "cosh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "crop_tensor": {
        "phi_name": "crop",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "shape": {
                "data_type": "int",
                "tensor_name": "Shape",
                "tensors_name": "ShapeTensor"
            },
            "offsets": {
                "data_type": "int",
                "tensor_name": "Offsets",
                "tensors_name": "OffsetsTensor"
            }
        }
    },
    "cross": {
        "phi_name": "cross",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim"
        }
    },
    "softmax_with_cross_entropy": {
        "phi_name": "cross_entropy_with_softmax",
        "inputs": {
            "input": "Logits",
            "label": "Label"
        },
        "outputs": {
            "softmax": "Softmax",
            "loss": "Loss"
        }
    },
    "cumprod": {
        "phi_name": "cumprod",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "dim": "dim"
        }
    },
    "cumsum": {
        "phi_name": "cumsum",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "cvm": {
        "phi_name": "cvm",
        "inputs": {
            "x": "X",
            "cvm": "CVM"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "data_norm": {
        "phi_name": "data_norm"
    },
    "decode_jpeg": {
        "phi_name": "decode_jpeg",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "deformable_conv": {
        "phi_name": "deformable_conv",
        "inputs": {
            "x": "Input",
            "offset": "Offset",
            "filter": "Filter",
            "mask": "Mask"
        },
        "outputs": {
            "out": "Output"
        }
    },
    "depthwise_conv2d": {
        "phi_name": "depthwise_conv2d",
        "inputs": {
            "input": "Input",
            "filter": "Filter"
        },
        "outputs": {
            "out": "Output"
        },
        "attrs": {
            "scale_in": "Scale_in",
            "scale_out": "Scale_out",
            "scale_in_eltwise": "Scale_in_eltwise",
            "scale_weights": "Scale_weights"
        }
    },
    "depthwise_conv2d_transpose": {
        "phi_name": "depthwise_conv2d_transpose",
        "inputs": {
            "x": "Input",
            "filter": "Filter",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Output"
        },
        "int_array": {
            "output_size": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "dequantize": {
        "phi_name": "dequantize",
        "inputs": {
            "input": "Input"
        },
        "outputs": {
            "output": "Output"
        },
        "attrs": {
            "scale": "Scale",
            "shift": "Shift"
        }
    },
    "dequantize_abs_max": {
        "phi_name": "dequantize_abs_max",
        "inputs": {
            "x": "X",
            "scale": "Scale"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dequantize_linear": {
        "phi_name": "dequantize_linear",
        "inputs": {
            "x": "X",
            "scale": "Scale",
            "zero_point": "ZeroPoint",
            "in_accum": "InAccum",
            "in_state": "InState"
        },
        "outputs": {
            "y": "Y",
            "out_scale": "OutScale",
            "out_accum": "OutAccum",
            "out_state": "OutState"
        }
    },
    "dequantize_log": {
        "phi_name": "dequantize_log",
        "inputs": {
            "x": "X",
            "dict": "Dict"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "determinant": {
        "phi_name": "det",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dgc_clip_by_norm": {
        "phi_name": "dgc_clip_by_norm",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dgc_momentum": {
        "phi_name": "dgc_momentum",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "velocity": "Velocity",
            "learning_rate": "LearningRate",
            "master_param": "MasterParam",
            "current_step_tensor": "current_step",
            "nranks_tensor": "nranks"
        },
        "outputs": {
            "param_out": "ParamOut",
            "velocity_out": "VelocityOut",
            "master_param_out": "MasterParamOut",
            "grad_out": "Grad_out"
        }
    },
    "diag_v2": {
        "phi_name": "diag",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "diag_embed": {
        "phi_name": "diag_embed",
        "inputs": {
            "input": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "diagonal": {
        "phi_name": "diagonal",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "digamma": {
        "phi_name": "digamma",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dirichlet": {
        "phi_name": "dirichlet",
        "inputs": {
            "alpha": "Alpha"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dist": {
        "phi_name": "dist",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "distributed_push_sparse": {
        "phi_name": "distributed_push_sparse",
        "inputs": {
            "ids": "Ids",
            "shows": "Shows",
            "clicks": "Clicks"
        },
        "outputs": {
            "output": "Outputs"
        }
    },
    "elementwise_div": {
        "phi_name": "divide",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dot": {
        "phi_name": "dot",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dropout": {
        "phi_name": "dropout",
        "inputs": {
            "x": "X",
            "seed_tensor": "Seed"
        },
        "outputs": {
            "out": "Out",
            "mask": "Mask"
        },
        "attrs": {
            "p": "dropout_prob",
            "is_test": "is_test",
            "mode": "dropout_implementation",
            "seed": "seed",
            "fix_seed": "fix_seed"
        },
        "scalar": {
            "p": {
                "support_tensor": "True"
            }
        }
    },
    "dropout_nd": {
        "phi_name": "dropout_nd"
    },
    "edit_distance": {
        "phi_name": "edit_distance",
        "inputs": {
            "hyps": "Hyps",
            "refs": "Refs",
            "hypslength": "HypsLength",
            "refslength": "RefsLength"
        },
        "outputs": {
            "sequencenum": "SequenceNum",
            "out": "Out"
        }
    },
    "eig": {
        "phi_name": "eig",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out_w": "Eigenvalues",
            "out_v": "Eigenvectors"
        }
    },
    "eigh": {
        "phi_name": "eigh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out_w": "Eigenvalues",
            "out_v": "Eigenvectors"
        }
    },
    "eigvals": {
        "phi_name": "eigvals",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "eigvalsh": {
        "phi_name": "eigvalsh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "eigenvalues": "Eigenvalues",
            "eigenvectors": "Eigenvectors"
        },
        "attrs": {
            "uplo": "UPLO"
        }
    },
    "einsum": {
        "phi_name": "einsum",
        "inputs": {
            "x": "Operands"
        },
        "outputs": {
            "out": "Out",
            "inner_cache": "InnerCache",
            "xshape": "XShape"
        }
    },
    "elementwise_pow": {
        "phi_name": "elementwise_pow",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elu": {
        "phi_name": "elu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lookup_table_v2": {
        "phi_name": "embedding",
        "inputs": {
            "x": "Ids",
            "weight": "W"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "sparse": "is_sparse"
        }
    },
    "empty": {
        "phi_name": "empty",
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "shape": {
                "data_type": "int64_t",
                "tensor_name": "ShapeTensor",
                "tensors_name": "ShapeTensorList"
            }
        }
    },
    "equal": {
        "phi_name": "equal",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "equal_all": {
        "phi_name": "equal_all",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "erf": {
        "phi_name": "erf",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "erfinv": {
        "phi_name": "erfinv",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "exp": {
        "phi_name": "exp",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "expand_v2": {
        "phi_name": "expand",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "shape": "shape"
        },
        "int_array": {
            "shape": {
                "data_type": "int",
                "tensor_name": "Shape",
                "tensors_name": "expand_shapes_tensor"
            }
        }
    },
    "expand_as_v2": {
        "phi_name": "expand_as",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "expm1": {
        "phi_name": "expm1",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "exponential": {
        "phi_name": "exponential_",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "lam": "lambda"
        }
    },
    "eye": {
        "phi_name": "eye",
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "num_rows": {
                "support_tensor": "True"
            },
            "num_columns": {
                "support_tensor": "True"
            }
        }
    },
    "fake_channel_wise_dequantize_max_abs": {
        "phi_name": "fake_channel_wise_dequantize_max_abs",
        "inputs": {
            "x": "X",
            "scales": "Scales"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fake_channel_wise_quantize_abs_max": {
        "phi_name": "fake_channel_wise_quantize_abs_max",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "out_scale": "OutScale"
        }
    },
    "fake_channel_wise_quantize_dequantize_abs_max": {
        "phi_name": "fake_channel_wise_quantize_dequantize_abs_max",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "out_scale": "OutScale"
        }
    },
    "fake_dequantize_max_abs": {
        "phi_name": "fake_dequantize_max_abs",
        "inputs": {
            "x": "X",
            "scale": "Scale"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fake_quantize_abs_max": {
        "phi_name": "fake_quantize_abs_max",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "out_scale": "OutScale"
        }
    },
    "fake_quantize_dequantize_abs_max": {
        "phi_name": "fake_quantize_dequantize_abs_max",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "out_scale": "OutScale"
        }
    },
    "fake_quantize_dequantize_moving_average_abs_max": {
        "phi_name": "fake_quantize_dequantize_moving_average_abs_max",
        "inputs": {
            "x": "X",
            "in_scale": "InScale",
            "in_accum": "InAccum",
            "in_state": "InState"
        },
        "outputs": {
            "out": "Out",
            "out_scale": "OutScale",
            "out_state": "OutState",
            "out_accum": "OutAccum"
        }
    },
    "fake_quantize_moving_average_abs_max": {
        "phi_name": "fake_quantize_moving_average_abs_max",
        "inputs": {
            "x": "X",
            "in_scale": "InScale",
            "in_accum": "InAccum",
            "in_state": "InState"
        },
        "outputs": {
            "out": "Out",
            "out_scale": "OutScale",
            "out_state": "OutState",
            "out_accum": "OutAccum"
        }
    },
    "fake_quantize_range_abs_max": {
        "phi_name": "fake_quantize_range_abs_max",
        "inputs": {
            "x": "X",
            "in_scale": "InScale",
            "iter": "Iter"
        },
        "outputs": {
            "out": "Out",
            "out_scale": "OutScale",
            "out_scales": "OutScales"
        }
    },
    "faster_tokenizer": {
        "phi_name": "faster_tokenizer",
        "inputs": {
            "vocab": "Vocab",
            "text": "Text",
            "text_pair": "TextPair"
        },
        "outputs": {
            "input_ids": "InputIds",
            "segment_ids": "SegmentIds"
        }
    },
    "fc": {
        "phi_name": "fc",
        "inputs": {
            "input": "Input",
            "w": "W",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "scale_in": "Scale_in",
            "scale_out": "Scale_out",
            "scale_weights": "Scale_weights"
        }
    },
    "feed": {
        "phi_name": "feed",
        "outputs": {
            "out": "Out"
        }
    },
    "fetch_barrier": {
        "phi_name": "fetch_barrier",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fft_c2c": {
        "phi_name": "fft_c2c",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fft_c2r": {
        "phi_name": "fft_c2r",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fft_r2c": {
        "phi_name": "fft_r2c",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fill_any": {
        "phi_name": "fill",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "value": {
                "data_type": "float",
                "support_tensor": "True"
            }
        }
    },
    "fill_diagonal": {
        "phi_name": "fill_diagonal",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fill_diagonal_tensor": {
        "phi_name": "fill_diagonal_tensor",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "flash_attn_unpadded": {
        "phi_name": "flash_attn_unpadded",
        "scalar": {
            "max_seqlen_q": {
                "data_type": "int64_t",
                "support_tensor": "True"
            },
            "max_seqlen_k": {
                "data_type": "int64_t",
                "support_tensor": "True"
            }
        }
    },
    "flash_attn_varlen_qkvpacked": {
        "phi_name": "flash_attn_varlen_qkvpacked",
        "scalar": {
            "max_seqlen_q": {
                "data_type": "int64_t",
                "support_tensor": "True"
            },
            "max_seqlen_k": {
                "data_type": "int64_t",
                "support_tensor": "True"
            }
        }
    },
    "flatten_contiguous_range": {
        "phi_name": "flatten",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "xshape": "XShape"
        },
        "attrs": {
            "start_axis": "start_axis",
            "stop_axis": "stop_axis"
        }
    },
    "flip": {
        "phi_name": "flip",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "floor": {
        "phi_name": "floor",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elementwise_floordiv": {
        "phi_name": "floor_divide",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elementwise_fmax": {
        "phi_name": "fmax",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elementwise_fmin": {
        "phi_name": "fmin",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fold": {
        "phi_name": "fold",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "frame": {
        "phi_name": "frame",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "frobenius_norm": {
        "phi_name": "frobenius_norm",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "fill_constant": {
        "phi_name": "full",
        "outputs": {
            "out": "Out"
        }
    },
    "fill_any_like": {
        "phi_name": "full_like",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "value": {
                "data_type": "float",
                "support_tensor": "True"
            }
        }
    },
    "full_with_tensor": {
        "phi_name": "full_with_tensor",
        "int_array": {
            "shape": {
                "data_type": "int64_t",
                "support_tensor": "True"
            }
        }
    },
    "fused_adam": {
        "phi_name": "fused_adam_",
        "inputs": {
            "params": "Params",
            "grads": "Grads",
            "learning_rate": "LearningRate",
            "moments1": "Moments1",
            "moments2": "Moments2",
            "moments2_max": "Moments2Max",
            "beta1_pows": "Beta1Pows",
            "beta2_pows": "Beta2Pows",
            "master_params": "MasterParams",
            "skip_update": "SkipUpdate"
        },
        "outputs": {
            "params_out": "ParamsOut",
            "moments1_out": "Moments1Out",
            "moments2_out": "Moments2Out",
            "moments2_max_out": "Moments2MaxOut",
            "beta1_pows_out": "Beta1PowsOut",
            "beta2_pows_out": "Beta2PowsOut",
            "master_params_out": "MasterParamsOut"
        }
    },
    "fused_attention": {
        "phi_name": "fused_attention",
        "inputs": {
            "x": "X",
            "ln_scale": "LnScale",
            "ln_bias": "LnBias",
            "qkv_weight": "QKVW",
            "qkv_bias": "QKVBias",
            "cache_kv": "CacheKV",
            "src_mask": "SrcMask",
            "out_linear_weight": "OutLinearW",
            "out_linear_bias": "OutLinearBias",
            "ln_scale_2": "Ln2Scale",
            "ln_bias_2": "Ln2Bias"
        },
        "outputs": {
            "ln_mean": "LnMean",
            "ln_var": "LnVariance",
            "ln_out": "LnOut",
            "qkv_out": "QKVOut",
            "qkv_bias_out": "QKVBiasOut",
            "transpose_out_2": "TransposeOut2",
            "qk_out": "QKOut",
            "qktv_out": "QKTVOut",
            "softmax_out": "SoftmaxOut",
            "attn_dropout_mask_out": "AttnDropoutMaskOut",
            "attn_dropout_out": "AttnDropoutOut",
            "src_mask_out": "SrcMaskOut",
            "fmha_out": "FMHAOut",
            "out_linear_out": "OutLinearOut",
            "dropout_mask_out": "DropoutMaskOut",
            "ln_mean_2": "Ln2Mean",
            "ln_var_2": "Ln2Variance",
            "bias_dropout_residual_out": "BiasDropoutResidualOut",
            "cache_kv_out": "CacheKVOut",
            "out": "Y"
        }
    },
    "fused_batch_norm_act": {
        "phi_name": "fused_batch_norm_act",
        "inputs": {
            "x": "X",
            "mean": "Mean",
            "variance": "Variance",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Y",
            "mean_out": "MeanOut",
            "variance_out": "VarianceOut",
            "saved_mean": "SavedMean",
            "saved_variance": "SavedVariance",
            "reserve_space": "ReserveSpace"
        }
    },
    "fused_bias_dropout_residual_layer_norm": {
        "phi_name": "fused_bias_dropout_residual_layer_norm",
        "inputs": {
            "x": "X",
            "residual": "Residual",
            "bias": "Bias",
            "ln_scale": "LnScale",
            "ln_bias": "LnBias"
        },
        "outputs": {
            "bias_dropout_residual_out": "BiasDropoutResidualOut",
            "dropout_mask_out": "DropoutMaskOut",
            "ln_mean": "LnMean",
            "ln_variance": "LnVariance",
            "y": "Y"
        }
    },
    "fused_bn_add_activation": {
        "phi_name": "fused_bn_add_activation_",
        "inputs": {
            "x": "X",
            "z": "Z",
            "mean": "Mean",
            "variance": "Variance",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Y",
            "mean_out": "MeanOut",
            "variance_out": "VarianceOut",
            "saved_mean": "SavedMean",
            "saved_variance": "SavedVariance",
            "reserve_space": "ReserveSpace"
        }
    },
    "fused_conv2d": {
        "phi_name": "fused_conv2d",
        "inputs": {
            "input": "Input",
            "filter": "Filter",
            "bias": "Bias",
            "residual_param": "ResidualData"
        },
        "outputs": {
            "output": "Output"
        },
        "attrs": {
            "scale_in": "Scale_in",
            "scale_out": "Scale_out",
            "scale_in_eltwise": "Scale_in_eltwise",
            "scale_weights": "Scale_weights"
        }
    },
    "fused_conv2d_add_act": {
        "phi_name": "fused_conv2d_add_act",
        "inputs": {
            "input": "Input",
            "filter": "Filter",
            "bias": "Bias",
            "residual_data": "ResidualData"
        },
        "outputs": {
            "output": "Output",
            "outputs": "Outputs"
        }
    },
    "fused_conv3d": {
        "phi_name": "fused_conv3d",
        "inputs": {
            "input": "Input",
            "filter": "Filter",
            "bias": "Bias",
            "residual_param": "ResidualData"
        },
        "outputs": {
            "output": "Output"
        },
        "attrs": {
            "scale_in": "Scale_in",
            "scale_out": "Scale_out",
            "scale_in_eltwise": "Scale_in_eltwise",
            "scale_weights": "Scale_weights"
        }
    },
    "fused_elementwise_add": {
        "phi_name": "fused_elementwise_add",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fused_elementwise_div": {
        "phi_name": "fused_elementwise_div",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fused_elementwise_mul": {
        "phi_name": "fused_elementwise_mul",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fused_elementwise_sub": {
        "phi_name": "fused_elementwise_sub",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fused_embedding_eltwise_layernorm": {
        "phi_name": "fused_embedding_eltwise_layernorm",
        "inputs": {
            "ids": "Ids",
            "embs": "Embs",
            "bias": "Bias",
            "scale": "Scale"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fused_embedding_fc_lstm": {
        "phi_name": "fused_embedding_fc_lstm",
        "inputs": {
            "ids": "Ids",
            "embeddings": "Embeddings",
            "weight_h": "WeightH",
            "bias": "Bias",
            "h0": "H0",
            "c0": "C0"
        },
        "outputs": {
            "hidden": "Hidden",
            "cell": "Cell",
            "xx": "XX",
            "batched_input": "BatchedInput",
            "batched_hidden": "BatchedHidden",
            "batched_cell": "BatchedCell",
            "reordered_h0": "ReorderedH0",
            "reordered_c0": "ReorderedC0"
        }
    },
    "fused_fc_elementwise_layernorm": {
        "phi_name": "fused_fc_elementwise_layernorm",
        "inputs": {
            "x": "X",
            "w": "W",
            "y": "Y",
            "bias0": "Bias0",
            "scale": "Scale",
            "bias1": "Bias1"
        },
        "outputs": {
            "out": "Out",
            "mean": "Mean",
            "variance": "Variance"
        }
    },
    "fused_feedforward": {
        "phi_name": "fused_feedforward",
        "inputs": {
            "x": "X",
            "dropout1_seed": "Dropout1Seed",
            "dropout2_seed": "Dropout2Seed",
            "linear1_weight": "Linear1Weight",
            "linear1_bias": "Linear1Bias",
            "linear2_weight": "Linear2Weight",
            "linear2_bias": "Linear2Bias",
            "ln1_scale": "Ln1Scale",
            "ln1_bias": "Ln1Bias",
            "ln2_scale": "Ln2Scale",
            "ln2_bias": "Ln2Bias"
        },
        "outputs": {
            "out": "Out",
            "dropout1_mask": "Dropout1Mask",
            "dropout2_mask": "Dropout2Mask",
            "ln1_mean": "Ln1Mean",
            "ln1_variance": "Ln1Variance",
            "ln2_mean": "Ln2Mean",
            "ln2_variance": "Ln2Variance",
            "linear1_out": "Linear1Out",
            "ln1_out": "Ln1Out",
            "dropout1_out": "Dropout1Out",
            "dropout2_out": "Dropout2Out"
        },
        "attrs": {
            "dropout1_seed_val": "dropout1_seed",
            "dropout2_seed_val": "dropout2_seed",
            "dropout1_prob": "dropout1_rate",
            "dropout2_prob": "dropout2_rate"
        }
    },
    "fused_gate_attention": {
        "phi_name": "fused_gate_attention",
        "inputs": {
            "query": "Query",
            "key": "Key",
            "query_weight": "QueryWeight",
            "key_weight": "KeyWeight",
            "value_weight": "ValueWeight",
            "qkv_weight": "QKVWeight",
            "nonbatched_bias": "NonbatchedBias",
            "src_mask": "SrcMask",
            "gate_weight": "GateWeight",
            "gate_bias": "GateBias",
            "out_linear_weight": "OutLinearWeight",
            "out_linear_bias": "OutLinearBias"
        },
        "outputs": {
            "query_transpose_out": "QueryTransposeOut",
            "key_transpose_out": "KeyTransposeOut",
            "value_transpose_out": "ValueTransposeOut",
            "qkv_transpose_out": "QKVTransposeOut",
            "softmax_out": "SoftmaxOut",
            "softmax_lse": "SoftmaxLse",
            "fmha_out": "FMHAOut",
            "gate_out": "GateOut",
            "out": "Out"
        }
    },
    "fused_gemm_epilogue": {
        "phi_name": "fused_gemm_epilogue",
        "inputs": {
            "x": "X",
            "y": "Y",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Out",
            "reserve_space": "ReserveSpace"
        }
    },
    "fused_gemm_epilogue_grad": {
        "phi_name": "fused_gemm_epilogue_grad",
        "inputs": {
            "x": "X",
            "y": "Y",
            "reserve_space": "ReserveSpace",
            "out_grad": "DOut"
        },
        "outputs": {
            "x_grad": "DX",
            "y_grad": "DY",
            "bias_grad": "DBias"
        }
    },
    "fused_multi_transformer_int8": {
        "phi_name": "fused_multi_transformer_int8",
        "inputs": {
            "x": "X",
            "ln_scale": "LnScale",
            "ln_bias": "LnBias",
            "qkv_w": "QKVW",
            "qkv_bias": "QKVBias",
            "cache_kv": "CacheKV",
            "time_step": "TimeStep",
            "src_mask": "SrcMask",
            "out_linear_w": "OutLinearW",
            "out_linear_bias": "OutLinearBias",
            "ffn_ln_scale": "FFNLnScale",
            "ffn_ln_bias": "FFNLnBias",
            "ffn1_weight": "FFN1Weight",
            "ffn1_bias": "FFN1Bias",
            "ffn2_weight": "FFN2Weight",
            "ffn2_bias": "FFN2Bias",
            "qkv_out_scale": "QKVOutScale",
            "out_linear_out_scale": "OutLinearOutScale",
            "ffn1_out_scale": "FFN1OutScale",
            "ffn2_out_scale": "FFN2OutScale"
        },
        "outputs": {
            "cache_kv_out": "CacheKVOut",
            "out": "Out"
        }
    },
    "fused_seqpool_cvm": {
        "phi_name": "fused_seqpool_cvm",
        "inputs": {
            "x": "X",
            "cvm": "CVM"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fused_transpose": {
        "phi_name": "fused_transpose",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fusion_gru": {
        "phi_name": "fusion_gru",
        "inputs": {
            "x": "X",
            "h0": "H0",
            "weight_x": "WeightX",
            "weight_h": "WeightH",
            "bias": "Bias"
        },
        "outputs": {
            "reordered_h0": "ReorderedH0",
            "xx": "XX",
            "batched_input": "BatchedInput",
            "batched_out": "BatchedOut",
            "hidden": "Hidden"
        },
        "attrs": {
            "scale_data": "Scale_data",
            "shift_data": "Shift_data",
            "scale_weights": "Scale_weights"
        }
    },
    "fusion_lstm": {
        "phi_name": "fusion_lstm",
        "inputs": {
            "x": "X",
            "h0": "H0",
            "weight_x": "WeightX",
            "weight_h": "WeightH",
            "bias": "Bias",
            "c0": "C0"
        },
        "outputs": {
            "out": "Out",
            "hidden": "Hidden",
            "cell": "Cell",
            "xx": "XX",
            "batched_input": "BatchedInput",
            "batched_hidden": "BatchedHidden",
            "batched_cell": "BatchedCell",
            "reordered_h0": "ReorderedH0",
            "reordered_c0": "ReorderedC0",
            "checked_cell": "CheckedCell"
        },
        "attrs": {
            "scale_data": "Scale_data",
            "shift_data": "Shift_data",
            "scale_weights": "Scale_weights"
        }
    },
    "fusion_repeated_fc_relu": {
        "phi_name": "fusion_repeated_fc_relu",
        "inputs": {
            "x": "X",
            "w": "W",
            "bias": "Bias"
        },
        "outputs": {
            "relu_out": "ReluOut",
            "out": "Out"
        }
    },
    "fusion_seqconv_eltadd_relu": {
        "phi_name": "fusion_seqconv_eltadd_relu",
        "inputs": {
            "x": "X",
            "filter": "Filter",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Out",
            "col_mat": "ColMat"
        },
        "attrs": {
            "context_length": "contextLength",
            "context_start": "contextStart",
            "context_stride": "contextStride"
        }
    },
    "fusion_seqpool_concat": {
        "phi_name": "fusion_seqpool_concat",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fusion_transpose_flatten_concat": {
        "phi_name": "fusion_transpose_flatten_concat",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "gather": {
        "phi_name": "gather",
        "inputs": {
            "x": "X",
            "index": "Index"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "axis": {
                "data_type": "int",
                "tensor_name": "Axis"
            }
        }
    },
    "gather_nd": {
        "phi_name": "gather_nd",
        "inputs": {
            "x": "X",
            "index": "Index"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "gather_tree": {
        "phi_name": "gather_tree",
        "inputs": {
            "ids": "Ids",
            "parents": "Parents"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "gaussian_random": {
        "phi_name": "gaussian",
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "shape": {
                "data_type": "int64_t",
                "tensor_name": "ShapeTensor",
                "tensors_name": "ShapeTensorList"
            }
        }
    },
    "gelu": {
        "phi_name": "gelu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "generate_proposals_v2": {
        "phi_name": "generate_proposals",
        "inputs": {
            "scores": "Scores",
            "bbox_deltas": "BboxDeltas",
            "im_shape": "ImShape",
            "anchors": "Anchors",
            "variances": "Variances"
        },
        "outputs": {
            "rpn_rois": "RpnRois",
            "rpn_roi_probs": "RpnRoiProbs",
            "rpn_rois_num": "RpnRoisNum"
        },
        "attrs": {
            "pre_nms_top_n": "pre_nms_topN",
            "post_nms_top_n": "post_nms_topN"
        }
    },
    "global_gather": {
        "phi_name": "global_gather",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "global_scatter": {
        "phi_name": "global_scatter",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "grad_add": {
        "phi_name": "grad_add",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "graph_khop_sampler": {
        "phi_name": "graph_khop_sampler",
        "inputs": {
            "row": "Row",
            "colptr": "Col_Ptr",
            "x": "X",
            "eids": "Eids"
        },
        "outputs": {
            "out_src": "Out_Src",
            "out_dst": "Out_Dst",
            "sample_index": "Sample_Index",
            "reindex_x": "Reindex_X",
            "out_eids": "Out_Eids"
        }
    },
    "graph_sample_neighbors": {
        "phi_name": "graph_sample_neighbors",
        "inputs": {
            "row": "Row",
            "colptr": "Col_Ptr",
            "x": "X",
            "eids": "Eids",
            "perm_buffer": "Perm_Buffer"
        },
        "outputs": {
            "out": "Out",
            "out_count": "Out_Count",
            "out_eids": "Out_Eids"
        }
    },
    "greater_equal": {
        "phi_name": "greater_equal",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "greater_than": {
        "phi_name": "greater_than",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "grid_sampler": {
        "phi_name": "grid_sample",
        "inputs": {
            "x": "X",
            "grid": "Grid"
        },
        "outputs": {
            "out": "Output"
        }
    },
    "group_norm": {
        "phi_name": "group_norm",
        "inputs": {
            "x": "X",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "y": "Y",
            "mean": "Mean",
            "variance": "Variance"
        },
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "gumbel_softmax": {
        "phi_name": "gumbel_softmax",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "hard_shrink": {
        "phi_name": "hardshrink",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "hard_sigmoid": {
        "phi_name": "hardsigmoid",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "hard_swish": {
        "phi_name": "hardswish",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "brelu": {
        "phi_name": "hardtanh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "hash": {
        "phi_name": "hash",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "runtime_shape": "ALL_KERNELS_MUST_COMPUTE_RUNTIME_SHAPE"
        }
    },
    "elementwise_heaviside": {
        "phi_name": "heaviside",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "hinge_loss": {
        "phi_name": "hinge_loss",
        "inputs": {
            "logits": "Logits",
            "labels": "Labels"
        },
        "outputs": {
            "loss": "Loss"
        }
    },
    "histogram": {
        "phi_name": "histogram",
        "inputs": {
            "input": "X",
            "weight": "Weight"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "hierarchical_sigmoid": {
        "phi_name": "hsigmoid_loss",
        "inputs": {
            "x": "X",
            "w": "W",
            "label": "Label",
            "bias": "Bias",
            "path": "PathTable",
            "code": "PathCode"
        },
        "outputs": {
            "out": "Out",
            "pre_out": "PreOut",
            "w_out": "W_Out"
        }
    },
    "huber_loss": {
        "phi_name": "huber_loss",
        "inputs": {
            "input": "X",
            "label": "Y"
        },
        "outputs": {
            "out": "Out",
            "residual": "Residual"
        }
    },
    "im2sequence": {
        "phi_name": "im2sequence",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "imag": {
        "phi_name": "imag",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "increment": {
        "phi_name": "increment",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "index_add": {
        "phi_name": "index_add",
        "inputs": {
            "x": "X",
            "index": "Index",
            "add_value": "AddValue"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "index_sample": {
        "phi_name": "index_sample",
        "inputs": {
            "x": "X",
            "index": "Index"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "index_select": {
        "phi_name": "index_select",
        "inputs": {
            "x": "X",
            "index": "Index"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim"
        }
    },
    "instance_norm": {
        "phi_name": "instance_norm",
        "inputs": {
            "x": "X",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "y": "Y",
            "saved_mean": "SavedMean",
            "saved_variance": "SavedVariance"
        }
    },
    "inverse": {
        "phi_name": "inverse",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Output"
        }
    },
    "is_empty": {
        "phi_name": "is_empty",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "isclose": {
        "phi_name": "isclose",
        "inputs": {
            "x": "Input",
            "y": "Other"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "rtol": {
                "data_type": "std::string",
                "tensor_name": "Rtol"
            },
            "atol": {
                "data_type": "std::string",
                "tensor_name": "Atol"
            }
        }
    },
    "isfinite_v2": {
        "phi_name": "isfinite",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "isinf_v2": {
        "phi_name": "isinf",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "isnan_v2": {
        "phi_name": "isnan",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "kldiv_loss": {
        "phi_name": "kldiv_loss",
        "inputs": {
            "x": "X",
            "label": "Target"
        },
        "outputs": {
            "out": "Loss"
        }
    },
    "kron": {
        "phi_name": "kron",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "kthvalue": {
        "phi_name": "kthvalue",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "indices": "Indices"
        }
    },
    "l1_norm": {
        "phi_name": "l1_norm",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "label_smooth": {
        "phi_name": "label_smooth",
        "inputs": {
            "label": "X",
            "prior_dist": "PriorDist"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lamb": {
        "phi_name": "lamb_",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "learning_rate": "LearningRate",
            "moment1": "Moment1",
            "moment2": "Moment2",
            "beta1_pow": "Beta1Pow",
            "beta2_pow": "Beta2Pow",
            "master_param": "MasterParam",
            "skip_update": "SkipUpdate"
        },
        "outputs": {
            "param_out": "ParamOut",
            "moment1_out": "Moment1Out",
            "moment2_out": "Moment2Out",
            "beta1_pow_out": "Beta1PowOut",
            "beta2_pow_out": "Beta2PowOut",
            "master_param_outs": "MasterParamOut"
        }
    },
    "layer_norm": {
        "phi_name": "layer_norm",
        "inputs": {
            "x": "X",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Y",
            "mean": "Mean",
            "variance": "Variance"
        }
    },
    "leaky_relu": {
        "phi_name": "leaky_relu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "negative_slope": "alpha"
        }
    },
    "bilinear_interp": {
        "phi_name": "legacy_bilinear_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        },
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "expand": {
        "phi_name": "legacy_expand",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "shape": "expand_times"
        },
        "int_array": {
            "shape": {
                "data_type": "int",
                "tensor_name": "ExpandTimes",
                "tensors_name": "expand_times_tensor"
            }
        }
    },
    "generate_proposals": {
        "phi_name": "legacy_generate_proposals",
        "inputs": {
            "scores": "Scores",
            "bbox_deltas": "BboxDeltas",
            "im_info": "ImInfo",
            "anchors": "Anchors",
            "variances": "Variances"
        },
        "outputs": {
            "rpn_rois": "RpnRois",
            "rpn_roi_probs": "RpnRoiProbs",
            "rpn_rois_num": "RpnRoisNum"
        },
        "attrs": {
            "pre_nms_top_n": "pre_nms_topN",
            "post_nms_top_n": "post_nms_topN"
        }
    },
    "matmul": {
        "phi_name": "legacy_matmul",
        "inputs": {
            "x": "X",
            "y": "Y",
            "out_grad": "DOut",
            "x_grad_grad": "DDX",
            "y_grad_grad": "DDY"
        },
        "outputs": {
            "out": "Out",
            "x_grad": "DX",
            "y_grad": "DY"
        },
        "attrs": {
            "transpose_x": "transpose_X",
            "transpose_y": "transpose_Y"
        }
    },
    "nearest_interp": {
        "phi_name": "legacy_nearest_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        },
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "reshape": {
        "phi_name": "legacy_reshape",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "xshape": "XShape"
        },
        "int_array": {
            "shape": {
                "data_type": "int",
                "tensor_name": "Shape",
                "tensors_name": "ShapeTensor"
            }
        }
    },
    "lerp": {
        "phi_name": "lerp",
        "inputs": {
            "x": "X",
            "y": "Y",
            "weight": "Weight"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "less_equal": {
        "phi_name": "less_equal",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "less_than": {
        "phi_name": "less_than",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lgamma": {
        "phi_name": "lgamma",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "linear_interp_v2": {
        "phi_name": "linear_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        },
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "linspace": {
        "phi_name": "linspace",
        "inputs": {
            "start": "Start",
            "stop": "Stop",
            "number": "Num"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lod_reset": {
        "phi_name": "lod_reset",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "log": {
        "phi_name": "log",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "log10": {
        "phi_name": "log10",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "log1p": {
        "phi_name": "log1p",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "log2": {
        "phi_name": "log2",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "log_loss": {
        "phi_name": "log_loss",
        "inputs": {
            "input": "Predicted",
            "label": "Labels"
        },
        "outputs": {
            "out": "Loss"
        }
    },
    "log_softmax": {
        "phi_name": "log_softmax",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "logcumsumexp": {
        "phi_name": "logcumsumexp",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "logical_and": {
        "phi_name": "logical_and",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "logical_not": {
        "phi_name": "logical_not",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "logical_or": {
        "phi_name": "logical_or",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "logical_xor": {
        "phi_name": "logical_xor",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "logit": {
        "phi_name": "logit",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "logsigmoid": {
        "phi_name": "logsigmoid",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "logsumexp": {
        "phi_name": "logsumexp",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lookup_table": {
        "phi_name": "lookup_table",
        "inputs": {
            "w": "W",
            "ids": "Ids"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lrn": {
        "phi_name": "lrn",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "mid_out": "MidOut"
        }
    },
    "lstsq": {
        "phi_name": "lstsq",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "solution": "Solution",
            "residuals": "Residuals",
            "rank": "Rank",
            "singular_values": "SingularValues"
        },
        "scalar": {
            "rcond": {
                "data_type": "float",
                "support_tensor": "True"
            }
        }
    },
    "lu_unpack": {
        "phi_name": "lu_unpack",
        "inputs": {
            "x": "X",
            "y": "Pivots"
        },
        "outputs": {
            "pmat": "Pmat",
            "l": "L",
            "u": "U"
        }
    },
    "margin_cross_entropy": {
        "phi_name": "margin_cross_entropy",
        "inputs": {
            "logits": "Logits",
            "label": "Label"
        },
        "outputs": {
            "softmax": "Softmax",
            "loss": "Loss"
        }
    },
    "masked_select": {
        "phi_name": "masked_select",
        "inputs": {
            "x": "X",
            "mask": "Mask"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "match_matrix_tensor": {
        "phi_name": "match_matrix_tensor",
        "inputs": {
            "x": "X",
            "y": "Y",
            "w": "W"
        },
        "outputs": {
            "out": "Out",
            "tmp": "Tmp"
        }
    },
    "matmul_v2": {
        "phi_name": "matmul",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "transpose_x": "trans_x",
            "transpose_y": "trans_y"
        }
    },
    "mul": {
        "phi_name": "matmul_with_flatten",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "matrix_nms": {
        "phi_name": "matrix_nms",
        "inputs": {
            "bboxes": "BBoxes",
            "scores": "Scores"
        },
        "outputs": {
            "out": "Out",
            "index": "Index",
            "roisnum": "RoisNum"
        }
    },
    "matrix_power": {
        "phi_name": "matrix_power",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "matrix_rank": {
        "phi_name": "matrix_rank",
        "inputs": {
            "x": "X",
            "tol_tensor": "TolTensor"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reduce_max": {
        "phi_name": "max",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "max_pool2d_with_index": {
        "phi_name": "max_pool2d_with_index",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "mask": "Mask"
        },
        "attrs": {
            "kernel_size": "ksize"
        }
    },
    "max_pool3d_with_index": {
        "phi_name": "max_pool3d_with_index",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "mask": "Mask"
        },
        "attrs": {
            "kernel_size": "ksize"
        }
    },
    "elementwise_max": {
        "phi_name": "maximum",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "maxout": {
        "phi_name": "maxout",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reduce_mean": {
        "phi_name": "mean",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "mean": {
        "phi_name": "mean_all",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "memory_efficient_attention": {
        "phi_name": "memory_efficient_attention",
        "scalar": {
            "max_seqlen_q": {
                "data_type": "int64_t",
                "support_tensor": "True"
            },
            "max_seqlen_k": {
                "data_type": "int64_t",
                "support_tensor": "True"
            }
        }
    },
    "merge_selected_rows": {
        "phi_name": "merge_selected_rows",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "merged_adam_": {
        "phi_name": "merged_adam_",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "learning_rate": "LearningRate",
            "moment1": "Moment1",
            "moment2": "Moment2",
            "moment2_max": "Moment2Max",
            "beta1_pow": "Beta1Pow",
            "beta2_pow": "Beta2Pow",
            "master_param": "MasterParam"
        },
        "outputs": {
            "param_out": "ParamOut",
            "moment1_out": "Moment1Out",
            "moment2_out": "Moment2Out",
            "moment2_max_out": "Moment2MaxOut",
            "beta1_pow_out": "Beta1PowOut",
            "beta2_pow_out": "Beta2PowOut",
            "master_param_out": "MasterParamOut"
        },
        "scalar": {
            "beta1": {
                "data_type": "float",
                "support_tensor": "True"
            },
            "beta2": {
                "data_type": "float",
                "support_tensor": "True"
            },
            "epsilon": {
                "data_type": "float",
                "support_tensor": "True"
            }
        }
    },
    "merged_momentum": {
        "phi_name": "merged_momentum_",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "velocity": "Velocity",
            "learning_rate": "LearningRate",
            "master_param": "MasterParam"
        },
        "outputs": {
            "param_out": "ParamOut",
            "velocity_out": "VelocityOut",
            "master_param_out": "MasterParamOut"
        }
    },
    "meshgrid": {
        "phi_name": "meshgrid",
        "inputs": {
            "inputs": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reduce_min": {
        "phi_name": "min",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "elementwise_min": {
        "phi_name": "minimum",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "mish": {
        "phi_name": "mish",
        "inputs": {
            "x": "X",
            "lambda": "threshold"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "mode": {
        "phi_name": "mode",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "indices": "Indices"
        }
    },
    "momentum": {
        "phi_name": "momentum_",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "velocity": "Velocity",
            "learning_rate": "LearningRate",
            "master_param": "MasterParam"
        },
        "outputs": {
            "param_out": "ParamOut",
            "velocity_out": "VelocityOut",
            "master_param_out": "MasterParamOut"
        }
    },
    "moving_average_abs_max_scale": {
        "phi_name": "moving_average_abs_max_scale",
        "inputs": {
            "x": "X",
            "in_accum": "InAccum",
            "in_state": "InState"
        },
        "outputs": {
            "out": "Out",
            "out_scale": "OutScale",
            "out_state": "OutState",
            "out_accum": "OutAccum"
        }
    },
    "multi_dot": {
        "phi_name": "multi_dot",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "multi_gru": {
        "phi_name": "multi_gru",
        "inputs": {
            "x": "X",
            "weight_x": "WeightX",
            "weight_h": "WeightH",
            "bias": "Bias",
            "scale_weights": "Scale_weights"
        },
        "outputs": {
            "hidden": "Hidden"
        },
        "attrs": {
            "scale_data": "Scale_data",
            "shift_data": "Shift_data"
        }
    },
    "multiclass_nms": {
        "phi_name": "multiclass_nms",
        "inputs": {
            "bboxes": "BBoxes",
            "scores": "Scores"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "multiclass_nms3": {
        "phi_name": "multiclass_nms3",
        "inputs": {
            "bboxes": "BBoxes",
            "scores": "Scores",
            "rois_num": "RoisNum"
        },
        "outputs": {
            "out": "Out",
            "index": "Index",
            "nms_rois_num": "NmsRoisNum"
        }
    },
    "multihead_matmul": {
        "phi_name": "multihead_matmul",
        "inputs": {
            "input": "Input",
            "w": "W",
            "bias": "Bias",
            "bias_qk": "BiasQK"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "transpose_q": "transpose_Q",
            "transpose_k": "transpose_K",
            "transpose_v": "transpose_V"
        }
    },
    "multinomial": {
        "phi_name": "multinomial",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "num_samples": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "multiplex": {
        "phi_name": "multiplex",
        "inputs": {
            "inputs": "X",
            "index": "Ids"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elementwise_mul": {
        "phi_name": "multiply",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "mv": {
        "phi_name": "mv",
        "inputs": {
            "x": "X",
            "vec": "Vec"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "nanmedian": {
        "phi_name": "nanmedian",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "medians": "MedianIndex"
        },
        "int_array": {
            "axis": {
                "data_type": "int"
            }
        }
    },
    "nearest_interp_v2": {
        "phi_name": "nearest_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        },
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "nll_loss": {
        "phi_name": "nll_loss",
        "inputs": {
            "input": "X",
            "label": "Label",
            "weight": "Weight"
        },
        "outputs": {
            "out": "Out",
            "total_weight": "Total_weight"
        }
    },
    "nms": {
        "phi_name": "nms",
        "inputs": {
            "x": "Boxes"
        },
        "outputs": {
            "out": "KeepBoxesIdxs"
        },
        "attrs": {
            "threshold": "iou_threshold"
        }
    },
    "where_index": {
        "phi_name": "nonzero",
        "inputs": {
            "condition": "Condition"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "norm": {
        "phi_name": "norm",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "norm": "Norm"
        }
    },
    "not_equal": {
        "phi_name": "not_equal",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "size": {
        "phi_name": "numel",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "size": "Out"
        }
    },
    "one_hot_v2": {
        "phi_name": "one_hot",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "depth": {
                "data_type": "int",
                "tensor_name": "depth_tensor"
            }
        }
    },
    "overlap_add": {
        "phi_name": "overlap_add",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "p_norm": {
        "phi_name": "p_norm",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "pad": {
        "phi_name": "pad",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "pad_value": {
                "data_type": "float",
                "support_tensor": "True"
            }
        }
    },
    "pad2d": {
        "phi_name": "pad2d"
    },
    "pad3d": {
        "phi_name": "pad3d",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "pad_value": "value"
        },
        "int_array": {
            "paddings": {
                "data_type": "int",
                "tensor_name": "Paddings"
            }
        }
    },
    "partial_allgather": {
        "phi_name": "partial_allgather",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "partial_concat": {
        "phi_name": "partial_concat",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "partial_recv": {
        "phi_name": "partial_recv",
        "outputs": {
            "out": "Out"
        }
    },
    "partial_sum": {
        "phi_name": "partial_sum",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "pixel_shuffle": {
        "phi_name": "pixel_shuffle",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "pixel_unshuffle": {
        "phi_name": "pixel_unshuffle",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "poisson": {
        "phi_name": "poisson",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "pool2d": {
        "phi_name": "pool2d",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "kernel_size": "ksize"
        },
        "int_array": {
            "kernel_size": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "pool3d": {
        "phi_name": "pool3d",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "kernel_size": "ksize"
        }
    },
    "pow": {
        "phi_name": "pow",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "y": "factor"
        },
        "scalar": {
            "y": {
                "data_type": "float",
                "tensor_name": "FactorTensor"
            }
        }
    },
    "prelu": {
        "phi_name": "prelu",
        "inputs": {
            "x": "X",
            "alpha": "Alpha"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "print": {
        "phi_name": "print",
        "inputs": {
            "in": "In"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "prior_box": {
        "phi_name": "prior_box",
        "inputs": {
            "input": "Input",
            "image": "Image"
        },
        "outputs": {
            "out": "Boxes",
            "var": "Variances"
        }
    },
    "reduce_prod": {
        "phi_name": "prod",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "psroi_pool": {
        "phi_name": "psroi_pool",
        "inputs": {
            "x": "X",
            "boxes": "ROIs",
            "boxes_num": "RoisNum"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "pull_box_sparse": {
        "phi_name": "pull_box_sparse",
        "inputs": {
            "w": "W",
            "ids": "Ids"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "sparse": "is_sparse"
        }
    },
    "pull_gpups_sparse": {
        "phi_name": "pull_gpups_sparse",
        "inputs": {
            "w": "W",
            "ids": "Ids"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "pull_sparse_v2": {
        "phi_name": "pull_sparse_v2",
        "inputs": {
            "ids": "Ids",
            "w": "W"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "push_box_sparse": {
        "phi_name": "push_box_sparse",
        "inputs": {
            "ids": "Ids"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "sparse": "is_sparse"
        }
    },
    "push_dense": {
        "phi_name": "push_dense",
        "inputs": {
            "ids": "Ids"
        },
        "attrs": {
            "table_id": "TableId",
            "scale_data_norm": "ScaleDataNorm",
            "input_names": "InputNames"
        }
    },
    "push_sparse_v2": {
        "phi_name": "push_sparse_v2",
        "inputs": {
            "x": "Ids",
            "W": "w"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "put_along_axis": {
        "phi_name": "put_along_axis",
        "inputs": {
            "arr": "Input",
            "indices": "Index",
            "values": "Value"
        },
        "outputs": {
            "out": "Result"
        },
        "attrs": {
            "axis": "Axis",
            "reduce": "Reduce",
            "include_self": "Include_self"
        }
    },
    "pylayer": {
        "phi_name": "pylayer"
    },
    "qr": {
        "phi_name": "qr",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "q": "Q",
            "r": "R"
        }
    },
    "quantize": {
        "phi_name": "quantize",
        "inputs": {
            "input": "Input"
        },
        "outputs": {
            "output": "Output"
        },
        "attrs": {
            "scale": "Scale",
            "shift": "Shift",
            "include_self": "Include_self"
        }
    },
    "quantize_linear": {
        "phi_name": "quantize_linear",
        "inputs": {
            "x": "X",
            "scale": "Scale",
            "zero_point": "ZeroPoint",
            "in_accum": "InAccum",
            "in_state": "InState"
        },
        "outputs": {
            "y": "Y",
            "out_scale": "OutScale",
            "out_accum": "OutAccum",
            "out_state": "OutState"
        }
    },
    "randint": {
        "phi_name": "randint",
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "shape": {
                "data_type": "int64_t",
                "tensor_name": "ShapeTensor",
                "tensors_name": "ShapeTensorList"
            }
        }
    },
    "randperm": {
        "phi_name": "randperm",
        "outputs": {
            "out": "Out"
        }
    },
    "real": {
        "phi_name": "real",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reciprocal": {
        "phi_name": "reciprocal",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "relu": {
        "phi_name": "relu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "relu6": {
        "phi_name": "relu6",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elementwise_mod": {
        "phi_name": "remainder",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "renorm": {
        "phi_name": "renorm",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "repeat_interleave": {
        "phi_name": "repeat_interleave",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "repeats": "Repeats",
            "axis": "dim"
        }
    },
    "repeat_interleave_with_tensor_index": {
        "phi_name": "repeat_interleave_with_tensor_index",
        "inputs": {
            "x": "X",
            "repeats": "RepeatTensor"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim"
        }
    },
    "requantize": {
        "phi_name": "requantize",
        "inputs": {
            "input": "Input"
        },
        "outputs": {
            "output": "Output"
        },
        "attrs": {
            "scale_in": "Scale_in",
            "scale_out": "Scale_out",
            "shift_in": "Shift_in",
            "shift_out": "Shift_out"
        }
    },
    "reshape2": {
        "phi_name": "reshape",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "xshape": "XShape"
        },
        "int_array": {
            "shape": {
                "data_type": "int",
                "tensor_name": "Shape",
                "tensors_name": "ShapeTensor"
            }
        }
    },
    "resnet_basic_block": {
        "phi_name": "resnet_basic_block",
        "inputs": {
            "x": "X",
            "filter1": "Filter1",
            "scale1": "Scale1",
            "bias1": "Bias1",
            "mean1": "Mean1",
            "var1": "Var1",
            "filter2": "Filter2",
            "scale2": "Scale2",
            "bias2": "Bias2",
            "mean2": "Mean2",
            "var2": "Var2",
            "filter3": "Filter3",
            "scale3": "Scale3",
            "bias3": "Bias3",
            "mean3": "Mean3",
            "var3": "Var3"
        },
        "outputs": {
            "out": "Y",
            "conv1": "Conv1",
            "saved_mean1": "SavedMean1",
            "saved_invstd1": "SavedInvstd1",
            "mean1_out": "Mean1Out",
            "var1_out": "Var1Out",
            "conv2": "Conv2",
            "conv2_input": "Conv2Input",
            "saved_mean2": "SavedMean2",
            "saved_invstd2": "SavedInvstd2",
            "mean2_out": "Mean2Out",
            "var2_out": "Var2Out",
            "conv3": "Conv3",
            "saved_mean3": "SavedMean3",
            "saved_invstd3": "SavedInvstd3",
            "mean3_out": "Mean3Out",
            "var3_out": "Var3Out",
            "max_input1": "MaxInput1",
            "max_filter1": "MaxFilter1",
            "max_input2": "MaxInput2",
            "max_filter2": "MaxFilter2",
            "max_input3": "MaxInput3",
            "max_filter3": "MaxFilter3"
        }
    },
    "resnet_unit": {
        "phi_name": "resnet_unit",
        "inputs": {
            "x": "X",
            "filter_x": "FilterX",
            "scale_x": "ScaleX",
            "bias_x": "BiasX",
            "mean_x": "MeanX",
            "var_x": "VarX",
            "z": "Z",
            "filter_z": "FilterZ",
            "scale_z": "ScaleZ",
            "bias_z": "BiasZ",
            "mean_z": "MeanZ",
            "var_z": "VarZ"
        },
        "outputs": {
            "out": "Y",
            "bit_mask": "BitMask",
            "conv_x": "ConvX",
            "saved_mean_x": "SavedMeanX",
            "saved_invstd_x": "SavedInvstdX",
            "running_mean_x": "RunningMeanX",
            "running_var_x": "RunningVarX",
            "conv_z": "ConvZ",
            "saved_mean_z": "SavedMeanZ",
            "saved_invstd_z": "SavedInvstdZ",
            "running_mean_z": "RunningMeanZ",
            "running_var_z": "RunningVarZ"
        }
    },
    "reverse": {
        "phi_name": "reverse",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "rmsprop": {
        "phi_name": "rmsprop_",
        "inputs": {
            "param": "Param",
            "mean_square": "MeanSquare",
            "mean_grad": "MeanGrad",
            "learning_rate": "LearningRate",
            "grad": "Grad",
            "moment": "Moment",
            "master_param": "MasterParam"
        },
        "outputs": {
            "param_out": "ParamOut",
            "moment_out": "MomentOut",
            "mean_square_out": "MeanSquareOut",
            "mean_grad_out": "MeanGradOut",
            "master_param_outs": "MasterParamOut"
        }
    },
    "rnn": {
        "phi_name": "rnn",
        "inputs": {
            "x": "Input",
            "pre_state": "PreState",
            "weight_list": "WeightList",
            "sequence_length": "SequenceLength"
        },
        "outputs": {
            "out": "Out",
            "dropout_state_out": "DropoutState",
            "state": "State",
            "reserve": "Reserve"
        }
    },
    "roi_align": {
        "phi_name": "roi_align",
        "inputs": {
            "x": "X",
            "boxes": "ROIs",
            "boxes_num": "RoisNum"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "roi_pool": {
        "phi_name": "roi_pool",
        "inputs": {
            "x": "X",
            "boxes": "ROIs",
            "boxes_num": "RoisNum"
        },
        "outputs": {
            "out": "Out",
            "arg_max": "Argmax"
        }
    },
    "roll": {
        "phi_name": "roll",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "shifts": {
                "data_type": "int64_t",
                "tensor_name": "ShiftsTensor"
            }
        }
    },
    "round": {
        "phi_name": "round",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "row_conv": {
        "phi_name": "row_conv",
        "inputs": {
            "x": "X",
            "filter": "Filter"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "rsqrt": {
        "phi_name": "rsqrt",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "save_combine": {
        "phi_name": "save_combine",
        "inputs": {
            "x": "X"
        }
    },
    "scale": {
        "phi_name": "scale",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "scale": {
                "data_type": "float",
                "tensor_name": "ScaleTensor"
            },
            "bias": {
                "data_type": "float",
                "support_tensor": "False"
            }
        }
    },
    "scatter": {
        "phi_name": "scatter",
        "inputs": {
            "x": "X",
            "index": "Ids",
            "updates": "Updates"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "scatter_nd_add": {
        "phi_name": "scatter_nd_add",
        "inputs": {
            "x": "X",
            "index": "Index",
            "updates": "Updates"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "searchsorted": {
        "phi_name": "searchsorted",
        "inputs": {
            "sorted_sequence": "SortedSequence",
            "values": "Values"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "seed": {
        "phi_name": "seed",
        "outputs": {
            "out": "Out"
        }
    },
    "segment_pool": {
        "phi_name": "segment_pool",
        "inputs": {
            "x": "X",
            "segment_ids": "SegmentIds"
        },
        "outputs": {
            "out": "Out",
            "summed_ids": "SummedIds"
        }
    },
    "self_dp_attention": {
        "phi_name": "self_dp_attention",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "selu": {
        "phi_name": "selu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "send_and_recv": {
        "phi_name": "send_and_recv",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "graph_send_recv": {
        "phi_name": "send_u_recv",
        "inputs": {
            "x": "X",
            "src_index": "Src_index",
            "dst_index": "Dst_index"
        },
        "outputs": {
            "out": "Out",
            "dst_count": "Dst_count"
        },
        "int_array": {
            "out_size": {
                "data_type": "int64_t",
                "tensor_name": "Out_size"
            }
        }
    },
    "graph_send_ue_recv": {
        "phi_name": "send_ue_recv",
        "inputs": {
            "x": "X",
            "y": "Y",
            "src_index": "Src_index",
            "dst_index": "Dst_index"
        },
        "outputs": {
            "out": "Out",
            "dst_count": "Dst_count"
        },
        "int_array": {
            "out_size": {
                "data_type": "int64_t",
                "tensor_name": "Out_size"
            }
        }
    },
    "graph_send_uv": {
        "phi_name": "send_uv"
    },
    "sequence_expand": {
        "phi_name": "sequence_expand",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sequence_mask": {
        "phi_name": "sequence_mask",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "y": "Y"
        },
        "attrs": {
            "max_len": "maxlen"
        },
        "scalar": {
            "max_len": {
                "data_type": "int",
                "tensor_name": "MaxLenTensor"
            }
        }
    },
    "sequence_softmax": {
        "phi_name": "sequence_softmax",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sgd": {
        "phi_name": "sgd_",
        "inputs": {
            "param": "Param",
            "learning_rate": "LearningRate",
            "grad": "Grad",
            "master_param": "MasterParam"
        },
        "outputs": {
            "param_out": "ParamOut",
            "master_param_out": "MasterParamOut"
        }
    },
    "shape": {
        "phi_name": "shape"
    },
    "shard_index": {
        "phi_name": "shard_index",
        "inputs": {
            "input": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "share_buffer": {
        "phi_name": "share_buffer",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "xout": "XOut"
        }
    },
    "share_data": {
        "phi_name": "share_data_",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "shuffle_batch": {
        "phi_name": "shuffle_batch",
        "inputs": {
            "x": "X",
            "seed": "Seed"
        },
        "outputs": {
            "out": "Out",
            "shuffle_idx": "ShuffleIdx",
            "seed_out": "SeedOut"
        }
    },
    "shuffle_channel": {
        "phi_name": "shuffle_channel",
        "inputs": {
            "x": "X",
            "group": "group"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sigmoid": {
        "phi_name": "sigmoid",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sign": {
        "phi_name": "sign",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "silu": {
        "phi_name": "silu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sin": {
        "phi_name": "sin",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sinh": {
        "phi_name": "sinh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "slice": {
        "phi_name": "slice",
        "inputs": {
            "input": "Input"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "starts": {
                "data_type": "int",
                "tensor_name": "StartsTensor",
                "tensors_name": "StartsTensorList"
            },
            "ends": {
                "data_type": "int",
                "tensor_name": "EndsTensor",
                "tensors_name": "EndsTensorList"
            }
        }
    },
    "slogdeterminant": {
        "phi_name": "slogdet",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "soft_relu": {
        "phi_name": "soft_relu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "softmax": {
        "phi_name": "softmax",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "softplus": {
        "phi_name": "softplus",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "softshrink": {
        "phi_name": "softshrink",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "threshold": "lambda"
        }
    },
    "softsign": {
        "phi_name": "softsign",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "solve": {
        "phi_name": "solve",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sparse_batch_norm": {
        "phi_name": "sparse_batch_norm",
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "sparse_reshape": {
        "phi_name": "sparse_reshape",
        "int_array": {
            "shape": {
                "data_type": "int64_t",
                "tensor_name": "ShapeTensor",
                "tensors_name": "ShapeTensorList"
            }
        }
    },
    "sparse_slice": {
        "phi_name": "sparse_slice",
        "int_array": {
            "starts": {
                "data_type": "int",
                "tensor_name": "StartsTensor",
                "tensors_name": "StartsTensorList"
            },
            "ends": {
                "data_type": "int",
                "tensor_name": "EndsTensor",
                "tensors_name": "EndsTensorList"
            }
        }
    },
    "sparse_sum": {
        "phi_name": "sparse_sum",
        "scalar": {
            "axis": {
                "data_type": "int",
                "tensor_name": "AxisTensor"
            }
        }
    },
    "sparse_sync_batch_norm": {
        "phi_name": "sparse_sync_batch_norm",
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "spectral_norm": {
        "phi_name": "spectral_norm",
        "inputs": {
            "weight": "Weight",
            "u": "U",
            "v": "V"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "split": {
        "phi_name": "split",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        },
        "int_array": {
            "sections": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "split_with_num": {
        "phi_name": "split_with_num",
        "scalar": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True",
                "tensor_name": "AxisTensor"
            }
        }
    },
    "sqrt": {
        "phi_name": "sqrt",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "square": {
        "phi_name": "square",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "squeeze2": {
        "phi_name": "squeeze",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "xshape": "XShape"
        },
        "attrs": {
            "axis": "axes"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "stack": {
        "phi_name": "stack",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "stanh": {
        "phi_name": "stanh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "straight_through_estimator_grad": {
        "phi_name": "straight_through_estimator_grad",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "strided_slice": {
        "phi_name": "strided_slice",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "starts": {
                "data_type": "int",
                "tensor_name": "StartsTensor",
                "tensors_name": "StartsTensorList"
            },
            "ends": {
                "data_type": "int",
                "tensor_name": "EndsTensor",
                "tensors_name": "EndsTensorList"
            },
            "strides": {
                "data_type": "int",
                "tensor_name": "StridesTensor",
                "tensors_name": "StridesTensorList"
            }
        }
    },
    "elementwise_sub": {
        "phi_name": "subtract",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reduce_sum": {
        "phi_name": "sum",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim",
            "dtype": "out_dtype"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "svd": {
        "phi_name": "svd",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "u": "U",
            "s": "S",
            "vh": "VH"
        }
    },
    "swish": {
        "phi_name": "swish",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sync_batch_norm": {
        "phi_name": "sync_batch_norm",
        "inputs": {
            "x": "X",
            "scale": "Scale",
            "bias": "Bias",
            "mean": "Mean",
            "variance": "Variance"
        },
        "outputs": {
            "out": "Y",
            "mean_out": "MeanOut",
            "variance_out": "VarianceOut",
            "saved_mean": "SavedMean",
            "saved_variance": "SavedVariance",
            "reserve_space": "ReserveSpace"
        },
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "take_along_axis": {
        "phi_name": "take_along_axis",
        "inputs": {
            "arr": "Input",
            "indices": "Index"
        },
        "outputs": {
            "out": "Result"
        },
        "attrs": {
            "axis": "Axis"
        }
    },
    "tan": {
        "phi_name": "tan",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "tanh": {
        "phi_name": "tanh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "tanh_shrink": {
        "phi_name": "tanh_shrink",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "tdm_child": {
        "phi_name": "tdm_child",
        "inputs": {
            "x": "X",
            "tree_info": "TreeInfo",
            "child_nums": "child_nums",
            "dtype": "dtype"
        },
        "outputs": {
            "child": "Child",
            "leaf_mask": "LeafMask"
        }
    },
    "tdm_sampler": {
        "phi_name": "tdm_sampler",
        "inputs": {
            "x": "X",
            "travel": "Travel",
            "layer": "Layer"
        },
        "outputs": {
            "out": "Out",
            "labels": "Labels",
            "mask": "Mask"
        }
    },
    "thresholded_relu": {
        "phi_name": "thresholded_relu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "tile": {
        "phi_name": "tile",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "repeat_times": {
                "data_type": "int",
                "tensor_name": "RepeatTimes",
                "tensors_name": "repeat_times_tensor"
            }
        }
    },
    "top_k_v2": {
        "phi_name": "topk",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "indices": "Indices"
        },
        "scalar": {
            "k": {
                "data_type": "int",
                "tensor_name": "K"
            }
        }
    },
    "top_k": {
        "phi_name": "topk_v1",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "indices": "Indices"
        },
        "scalar": {
            "k": {
                "data_type": "int",
                "tensor_name": "K"
            }
        }
    },
    "trace": {
        "phi_name": "trace",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "transpose2": {
        "phi_name": "transpose",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "perm": "axis"
        }
    },
    "triangular_solve": {
        "phi_name": "triangular_solve",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "tril_triu": {
        "phi_name": "tril_triu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "trilinear_interp_v2": {
        "phi_name": "trilinear_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        },
        "attrs": {
            "data_format": "data_layout"
        }
    },
    "trunc": {
        "phi_name": "trunc",
        "inputs": {
            "input": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "truncated_gaussian_random": {
        "phi_name": "truncated_gaussian_random",
        "outputs": {
            "out": "Out"
        }
    },
    "unbind": {
        "phi_name": "unbind",
        "inputs": {
            "input": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "unfold": {
        "phi_name": "unfold",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "uniform_random": {
        "phi_name": "uniform",
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "min": {
                "data_type": "float",
                "support_tensor": "True"
            },
            "max": {
                "data_type": "float",
                "support_tensor": "True"
            }
        },
        "int_array": {
            "shape": {
                "data_type": "int64_t",
                "tensor_name": "ShapeTensor",
                "tensors_name": "ShapeTensorList"
            }
        }
    },
    "uniform_random_inplace": {
        "phi_name": "uniform_inplace",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "unique": {
        "phi_name": "unique",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "indices": "Indices",
            "inverse": "Index",
            "counts": "Counts"
        }
    },
    "unique_consecutive": {
        "phi_name": "unique_consecutive",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "index": "Index",
            "counts": "Counts"
        }
    },
    "unpool": {
        "phi_name": "unpool",
        "inputs": {
            "x": "X",
            "indices": "Indices"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "padding": "paddings"
        },
        "int_array": {
            "output_size": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "unpool3d": {
        "phi_name": "unpool3d",
        "inputs": {
            "x": "X",
            "indices": "Indices"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "unsqueeze2": {
        "phi_name": "unsqueeze",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "xshape": "XShape"
        },
        "attrs": {
            "axis": "axes"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "tensor_name": "AxesTensor",
                "tensors_name": "AxesTensorList"
            }
        }
    },
    "unstack": {
        "phi_name": "unstack",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "update_loss_scaling": {
        "phi_name": "update_loss_scaling_",
        "inputs": {
            "x": "X",
            "found_infinite": "FoundInfinite",
            "prev_loss_scaling": "PrevLossScaling",
            "in_good_steps": "InGoodSteps",
            "in_bad_steps": "InBadSteps"
        },
        "outputs": {
            "out": "Out",
            "loss_scaling": "LossScaling",
            "out_good_steps": "OutGoodSteps",
            "out_bad_steps": "OutBadSteps"
        },
        "scalar": {
            "stop_update": {
                "data_type": "bool",
                "tensor_name": "StopUpdate"
            }
        }
    },
    "viterbi_decode": {
        "phi_name": "viterbi_decode",
        "inputs": {
            "potentials": "Input",
            "transition_params": "Transition",
            "lengths": "Length"
        },
        "outputs": {
            "scores": "Scores",
            "path": "Path"
        }
    },
    "warpctc": {
        "phi_name": "warpctc",
        "inputs": {
            "logits": "Logits",
            "label": "Label",
            "logits_length": "LogitsLength",
            "labels_length": "LabelLength"
        },
        "outputs": {
            "warpctcgrad": "WarpCTCGrad",
            "loss": "Loss"
        }
    },
    "where": {
        "phi_name": "where",
        "inputs": {
            "condition": "Condition",
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "while": {
        "phi_name": "while"
    },
    "yolo_box": {
        "phi_name": "yolo_box",
        "inputs": {
            "x": "X",
            "img_size": "ImgSize"
        },
        "outputs": {
            "boxes": "Boxes",
            "scores": "Scores"
        }
    },
    "yolo_box_head": {
        "phi_name": "yolo_box_head",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "yolo_box_post": {
        "phi_name": "yolo_box_post",
        "inputs": {
            "boxes0": "Boxes0",
            "boxes1": "Boxes1",
            "boxes2": "Boxes2",
            "image_shape": "ImageShape",
            "image_scale": "ImageScale"
        },
        "outputs": {
            "out": "Out",
            "nms_rois_num": "NmsRoisNum"
        }
    },
    "yolov3_loss": {
        "phi_name": "yolo_loss",
        "inputs": {
            "x": "X",
            "gt_box": "GTBox",
            "gt_label": "GTLabel",
            "gt_score": "GTScore"
        },
        "outputs": {
            "loss": "Loss",
            "objectness_mask": "ObjectnessMask",
            "gt_match_mask": "GTMatchMask"
        }
    },
    "box_clip": {
        "phi_name": "box_clip",
        "inputs": {
            "input": "Input",
            "im_info": "ImInfo"
        },
        "outputs": {
            "output": "Output"
        }
    },
    "c_allreduce_sum": {
        "phi_name": "c_allreduce_sum",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "c_identity": {
        "phi_name": "c_identity",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "c_scatter": {
        "phi_name": "c_scatter",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "channel_shuffle": {
        "phi_name": "channel_shuffle",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "chunk_eval": {
        "phi_name": "chunk_eval",
        "inputs": {
            "inference": "Inference",
            "label": "Label",
            "seq_length": "SeqLength"
        },
        "outputs": {
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-Score",
            "num_infer_chunks": "NumInferChunks",
            "num_label_chunks": "NumLabelChunks",
            "num_correct_chunks": "NumCorrectChunks"
        }
    },
    "comm_init_all": {
        "phi_name": "comm_init_all"
    },
    "crf_decoding": {
        "phi_name": "crf_decoding",
        "inputs": {
            "emission": "Emission",
            "transition": "Transition",
            "label": "Label",
            "length": "Length"
        },
        "outputs": {
            "viterbi_path": "ViterbiPath"
        }
    },
    "cross_entropy": {
        "phi_name": "cross_entropy",
        "inputs": {
            "x": "X",
            "label": "Label"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "cross_entropy2": {
        "phi_name": "cross_entropy2",
        "inputs": {
            "x": "X",
            "label": "Label"
        },
        "outputs": {
            "out": "Y",
            "x_shape": "XShape",
            "match_x": "MatchX"
        }
    },
    "ctc_align": {
        "phi_name": "ctc_align",
        "inputs": {
            "input": "Input",
            "input_length": "InputLength"
        },
        "outputs": {
            "output": "Output",
            "output_length": "OutputLength"
        }
    },
    "cudnn_lstm": {
        "phi_name": "cudnn_lstm",
        "inputs": {
            "x": "Input",
            "init_h": "InitH",
            "init_c": "InitC",
            "w": "W",
            "weight_list": "WeightList",
            "sequence_length": "SequenceLength"
        },
        "outputs": {
            "reserve": "Reserve",
            "state_out": "StateOut",
            "out": "Out",
            "last_h": "LastH",
            "last_c": "LastC"
        }
    },
    "decayed_adagrad": {
        "phi_name": "decayed_adagrad",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "moment": "Moment",
            "learning_rate": "LearningRate"
        },
        "outputs": {
            "param_out": "ParamOut",
            "moment_out": "MomentOut"
        }
    },
    "depend": {
        "phi_name": "depend",
        "inputs": {
            "x": "X",
            "dep": "Dep"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dgc": {
        "phi_name": "dgc",
        "inputs": {
            "u": "U",
            "v": "V",
            "grad": "Grad",
            "param": "Param"
        },
        "outputs": {
            "u_out": "U_out",
            "v_out": "V_out",
            "encode_grad": "EncodeGrad",
            "grad_out": "Grad_out",
            "gather_buff": "GatherBuff"
        }
    },
    "distribute_fpn_proposals": {
        "phi_name": "distribute_fpn_proposals",
        "inputs": {
            "fpn_rois": "FpnRois",
            "rois_num": "RoisNum"
        },
        "outputs": {
            "multi_fpn_rois": "MultiFpnRois",
            "multi_level_rois_num": "MultiLevelRoIsNum",
            "restore_index": "RestoreIndex"
        }
    },
    "distributed_fused_lamb_init": {
        "phi_name": "distributed_fused_lamb_init",
        "inputs": {
            "param": "Param",
            "grad": "Grad"
        },
        "outputs": {
            "fp32_fused_param": "FP32FusedParam",
            "fp32_fused_grad": "FP32FusedGrad",
            "fp16_fused_param": "FP16FusedParam",
            "fp16_fused_grad": "FP16FusedGrad",
            "moment1": "Moment1",
            "moment2": "Moment2",
            "beta1_pow": "Beta1Pow",
            "beta2_pow": "Beta2Pow",
            "fused_param_offsets": "FusedParamOffsets",
            "fp32_shard_fused_param_offsets": "FP32ShardFusedParamOffsets",
            "fp16_shard_fused_param_offsets": "FP16ShardFusedParamOffsets",
            "param_info": "ParamInfo",
            "param_order": "ParamOrder",
            "param_out": "ParamOut",
            "master_param_out": "MasterParamOut",
            "grad_out": "GradOut",
            "global_scale": "GlobalScale",
            "step": "Step"
        }
    },
    "distributed_lookup_table": {
        "phi_name": "distributed_lookup_table",
        "inputs": {
            "ids": "Ids",
            "w": "W"
        },
        "outputs": {
            "outputs": "Outputs"
        }
    },
    "dpsgd": {
        "phi_name": "dpsgd",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "learning_rate": "LearningRate"
        },
        "outputs": {
            "param_out": "ParamOut"
        }
    },
    "fetch_v2": {
        "phi_name": "fetch",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "flatten2": {
        "phi_name": "flatten2",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "x_shape": "XShape"
        }
    },
    "ftrl": {
        "phi_name": "ftrl",
        "inputs": {
            "param": "Param",
            "squared_accumulator": "SquaredAccumulator",
            "linear_accumulator": "LinearAccumulator",
            "grad": "Grad",
            "learning_rate": "LearningRate"
        },
        "outputs": {
            "param_out": "ParamOut",
            "squared_accum_out": "SquaredAccumOut",
            "linear_accum_out": "LinearAccumOut"
        }
    },
    "fill_constant_batch_size_like": {
        "phi_name": "full_batch_size_like",
        "inputs": {
            "input": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fused_elemwise_activation": {
        "phi_name": "fused_elemwise_activation",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out",
            "intermediate_out": "IntermediateOut"
        }
    },
    "fused_elemwise_add_activation": {
        "phi_name": "fused_elemwise_add_activation",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out",
            "intermediate_out": "IntermediateOut"
        }
    },
    "fused_matmul": {
        "phi_name": "fused_matmul",
        "inputs": {
            "x": "X",
            "y": "Y",
            "residual_data": "ResidualData"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "scale_x": "Scale_x",
            "scale_y": "Scale_y",
            "scale_out": "Scale_out",
            "scale_in_eltwise": "Scale_in_eltwise",
            "fused_reshape_x": "fused_reshape_X",
            "fused_transpose_x": "fused_transpose_X",
            "fused_reshape_y": "fused_reshape_Y",
            "fused_transpose_y": "fused_transpose_Y",
            "fused_reshape_out": "fused_reshape_Out",
            "fused_transpose_out": "fused_transpose_Out"
        }
    },
    "fused_softmax_mask": {
        "phi_name": "fused_softmax_mask",
        "inputs": {
            "x": "X",
            "mask": "Mask"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fused_softplus": {
        "phi_name": "fused_softplus",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fused_token_prune": {
        "phi_name": "fused_token_prune",
        "inputs": {
            "attn": "Attn",
            "x": "X",
            "mask": "Mask",
            "new_mask": "NewMask"
        },
        "outputs": {
            "slimmed_x": "SlimmedX",
            "cls_inds": "CLSInds"
        }
    },
    "fusion_group": {
        "phi_name": "fusion_group",
        "inputs": {
            "inputs": "Inputs"
        },
        "outputs": {
            "outs": "Outs"
        }
    },
    "fusion_seqpool_cvm_concat": {
        "phi_name": "fusion_seqpool_cvm_concat",
        "inputs": {
            "x": "X",
            "cvm": "CVM"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fusion_squared_mat_sub": {
        "phi_name": "fusion_squared_mat_sub",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "squared_x": "SquaredX",
            "squared_y": "SquaredY",
            "squared_xy": "SquaredXY",
            "out": "Out"
        }
    },
    "get_tensor_from_selected_rows": {
        "phi_name": "get_tensor_from_selected_rows",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "gru": {
        "phi_name": "gru",
        "inputs": {
            "input": "Input",
            "h0": "H0",
            "weight": "Weight",
            "bias": "Bias"
        },
        "outputs": {
            "batch_gate": "BatchGate",
            "batch_reset_hidden_prev": "BatchResetHiddenPrev",
            "batch_hidden": "BatchHidden",
            "hidden": "Hidden"
        }
    },
    "gru_unit": {
        "phi_name": "gru_unit",
        "inputs": {
            "input": "Input",
            "hidden_prev": "HiddenPrev",
            "weight": "Weight",
            "bias": "Bias"
        },
        "outputs": {
            "gate": "Gate",
            "reset_hidden_prev": "ResetHiddenPrev",
            "hidden": "Hidden"
        }
    },
    "identity_loss": {
        "phi_name": "identity_loss",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lars_momentum": {
        "phi_name": "lars_momentum_",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "velocity": "Velocity",
            "learning_rate": "LearningRate",
            "master_param": "MasterParam"
        },
        "outputs": {
            "param_out": "ParamOut",
            "velocity_out": "VelocityOut",
            "master_param_out": "MasterParamOut"
        }
    },
    "crop": {
        "phi_name": "legacy_crop",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "offsets": {
                "data_type": "int",
                "tensor_name": "Offsets"
            }
        }
    },
    "limit_by_capacity": {
        "phi_name": "limit_by_capacity",
        "outputs": {
            "out": "Out"
        }
    },
    "lod_array_length": {
        "phi_name": "lod_array_length",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "logspace": {
        "phi_name": "logspace",
        "inputs": {
            "start": "Start",
            "stop": "Stop",
            "num": "Num",
            "base": "Base"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lookup_table_dequant": {
        "phi_name": "lookup_table_dequant",
        "inputs": {
            "w": "W",
            "ids": "Ids"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lstm": {
        "phi_name": "lstm",
        "inputs": {
            "input": "Input",
            "h0": "H0",
            "c0": "C0",
            "weight": "Weight",
            "bias": "Bias"
        },
        "outputs": {
            "hidden": "Hidden",
            "cell": "Cell",
            "batch_gate": "BatchGate",
            "batch_cell_pre_act": "BatchCellPreAct"
        }
    },
    "lu": {
        "phi_name": "lu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "pivots": "Pivots",
            "infos": "Infos"
        },
        "attrs": {
            "pivot": "pivots"
        }
    },
    "memcpy": {
        "phi_name": "memcpy",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "memcpy_d2h": {
        "phi_name": "memcpy_d2h",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "mp_allreduce_sum": {
        "phi_name": "mp_allreduce_sum",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "nce": {
        "phi_name": "nce",
        "inputs": {
            "input": "Input",
            "label": "Label",
            "weight": "Weight",
            "bias": "Bias",
            "sample_weight": "SampleWeight",
            "custom_dist_probs": "CustomDistProbs",
            "custom_dist_alias": "CustomDistAlias",
            "custom_dist_alias_probs": "CustomDistAliasProbs"
        },
        "outputs": {
            "cost": "Cost",
            "sample_logits": "SampleLogits",
            "sample_labels": "SampleLabels"
        }
    },
    "nop": {
        "phi_name": "nop",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "number_count": {
        "phi_name": "number_count",
        "inputs": {
            "numbers": "numbers"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "partial_send": {
        "phi_name": "partial_send",
        "inputs": {
            "x": "X"
        }
    },
    "prune_gate_by_capacity": {
        "phi_name": "prune_gate_by_capacity",
        "inputs": {
            "gate_idx": "GateIdx",
            "expert_count": "ExpertCount"
        },
        "outputs": {
            "out_gate_idx": "NewGateIdx"
        }
    },
    "pyramid_hash": {
        "phi_name": "pyramid_hash",
        "inputs": {
            "x": "X",
            "w": "W",
            "white_list": "WhiteList",
            "black_list": "BlackList"
        },
        "outputs": {
            "out": "Out",
            "drop_pos": "DropPos",
            "x_temp_out": "X_Temp_Out"
        }
    },
    "random_routing": {
        "phi_name": "random_routing",
        "inputs": {
            "prob": "Prob",
            "topk_value": "TopK_Value",
            "topk_idx": "TopK_Idx"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "rank_attention": {
        "phi_name": "rank_attention",
        "inputs": {
            "x": "X",
            "rank_offset": "RankOffset",
            "rank_param": "RankParam"
        },
        "outputs": {
            "input_help": "InputHelp",
            "out": "Out",
            "ins_rank": "InsRank"
        },
        "attrs": {
            "max_rank": "MaxRank",
            "max_size": "MaxSize"
        }
    },
    "read_from_array": {
        "phi_name": "read_from_array",
        "inputs": {
            "array": "X",
            "i": "I"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "recv_v2": {
        "phi_name": "recv_v2",
        "outputs": {
            "out": "Out"
        }
    },
    "graph_reindex": {
        "phi_name": "reindex_graph",
        "inputs": {
            "x": "X",
            "neighbors": "Neighbors",
            "count": "Count",
            "hashtable_value": "HashTable_Value",
            "hashtable_index": "HashTable_Index"
        },
        "outputs": {
            "reindex_src": "Reindex_Src",
            "reindex_dst": "Reindex_Dst",
            "out_nodes": "Out_Nodes"
        }
    },
    "rrelu": {
        "phi_name": "rrelu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "noise": "Noise"
        }
    },
    "send_v2": {
        "phi_name": "send_v2",
        "inputs": {
            "x": "X"
        }
    },
    "sequence_conv": {
        "phi_name": "sequence_conv",
        "inputs": {
            "x": "X",
            "padding_data": "PaddingData",
            "filter": "Filter"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "padding_trainable": "paddingTrainable",
            "context_length": "contextLength",
            "context_start": "contextStart",
            "context_stride": "contextStride"
        }
    },
    "sequence_pool": {
        "phi_name": "sequence_pool",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "max_index": "MaxIndex"
        }
    },
    "set_value": {
        "phi_name": "set_value",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "starts": {
                "data_type": "int64_t",
                "tensors_name": "StartsTensorList"
            },
            "ends": {
                "data_type": "int64_t",
                "tensors_name": "EndsTensorList"
            },
            "steps": {
                "data_type": "int64_t",
                "tensors_name": "StepsTensorList"
            }
        }
    },
    "set_value_with_tensor": {
        "phi_name": "set_value_with_tensor",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "starts": {
                "data_type": "int64_t",
                "tensors_name": "StartsTensorList"
            },
            "ends": {
                "data_type": "int64_t",
                "tensors_name": "EndsTensorList"
            },
            "steps": {
                "data_type": "int64_t",
                "tensors_name": "StepsTensorList"
            }
        }
    },
    "sigmoid_cross_entropy_with_logits": {
        "phi_name": "sigmoid_cross_entropy_with_logits",
        "inputs": {
            "x": "X",
            "label": "Label"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "skip_layernorm": {
        "phi_name": "skip_layernorm",
        "inputs": {
            "x": "X",
            "y": "Y",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sparse_attention": {
        "phi_name": "sparse_attention",
        "inputs": {
            "q": "Q",
            "k": "K",
            "v": "V",
            "offset": "Offset",
            "columns": "Columns",
            "key_padding_mask": "KeyPaddingMask",
            "attn_mask": "AttnMask"
        },
        "outputs": {
            "out": "Out",
            "sparse_dot_sdd": "SparseDotSdd",
            "softmax": "Softmax"
        }
    },
    "sparse_momentum": {
        "phi_name": "sparse_momentum",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "velocity": "Velocity",
            "index": "Index",
            "axis": "Axis",
            "learning_rate": "LearningRate",
            "master_param": "MasterParam"
        },
        "outputs": {
            "param_out": "ParamOut",
            "velocity_out": "VelocityOut",
            "master_param_out": "MasterParamOut"
        },
        "attrs": {
            "axis": "axis"
        },
        "scalar": {
            "axis": {
                "data_type": "int",
                "tensor_name": "Axis"
            }
        }
    },
    "squared_l2_norm": {
        "phi_name": "squared_l2_norm",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "stft": {
        "phi_name": "stft",
        "inputs": {
            "x": "X",
            "window": "Window"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "c_sync_calc_stream": {
        "phi_name": "sync_calc_stream",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "c_sync_comm_stream": {
        "phi_name": "sync_comm_stream",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "temporal_shift": {
        "phi_name": "temporal_shift",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "transfer_layout": {
        "phi_name": "transfer_layout",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "uniform_random_batch_size_like": {
        "phi_name": "uniform_random_batch_size_like",
        "inputs": {
            "input": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "write_to_array": {
        "phi_name": "write_to_array",
        "inputs": {
            "x": "X",
            "i": "I"
        },
        "outputs": {
            "out": "Out"
        }
    }
}
op_info = {
    "abs": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "accuracy": {
        "args": "Tensor x, Tensor indices, Tensor label",
        "output": "Tensor(accuracy), Tensor(correct), Tensor(total)"
    },
    "accuracy_check": {
        "args": "Tensor x, Tensor y, str fn_name, double rtol=1e-5, double atol=1e-8,  bool equal_nan=false",
        "output": "Tensor(out)"
    },
    "acos": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "acosh": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "adadelta_": {
        "args": "Tensor param, Tensor grad, Tensor avg_squared_grad, Tensor avg_squared_update, Tensor learning_rate, Tensor master_param, float rho = 0.95f, float epsilon = 1.0e-6f, bool multi_precision = false",
        "output": "Tensor(param_out), Tensor(moment_out), Tensor(inf_norm_out), Tensor(master_param_out)"
    },
    "adagrad_": {
        "args": "Tensor param, Tensor grad, Tensor moment, Tensor learning_rate, Tensor master_param, float epsilon = 1.0e-6f, bool multi_precision = false",
        "output": "Tensor(param_out), Tensor(moment_out), Tensor(master_param_out)"
    },
    "adam_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment1, Tensor moment2, Tensor moment2_max, Tensor beta1_pow, Tensor beta2_pow, Tensor master_param, Tensor skip_update, Scalar beta1 = 0.9f, Scalar beta2 = 0.999f, Scalar epsilon = 1.0e-8f, bool lazy_mode = false, int64_t min_row_size_to_use_multithread = 1000, bool multi_precision = false, bool use_global_beta_pow = false, bool amsgrad = false",
        "output": "Tensor(param_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(moment2_max_out), Tensor(beta1_pow_out), Tensor(beta2_pow_out), Tensor(master_param_out)"
    },
    "adamax_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment, Tensor inf_norm, Tensor beta1_pow, Tensor master_param, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1.0e-8f, bool multi_precision = false",
        "output": "Tensor(param_out), Tensor(moment_out), Tensor(inf_norm_out), Tensor(master_param_out)"
    },
    "adamw_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment1, Tensor moment2, Tensor moment2_max, Tensor beta1_pow, Tensor beta2_pow, Tensor master_param, Tensor skip_update, Scalar beta1 = 0.9f, Scalar beta2 = 0.999f, Scalar epsilon = 1.0e-8f, float lr_ratio = 1.0f, float coeff = 0.01f, bool with_decay = false, bool lazy_mode = false, int64_t min_row_size_to_use_multithread = 1000, bool multi_precision = false, bool use_global_beta_pow = false, bool amsgrad = false",
        "output": "Tensor(param_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(moment2_max_out), Tensor(beta1_pow_out), Tensor(beta2_pow_out), Tensor(master_param_out)"
    },
    "add_position_encoding": {
        "args": "Tensor x, float alpha = 1.0f, float beta = 1.0f",
        "output": "Tensor (out)"
    },
    "addmm": {
        "args": "Tensor input, Tensor x, Tensor y, float beta=1.0, float alpha=1.0",
        "output": "Tensor(out)"
    },
    "affine_channel": {
        "args": "Tensor x, Tensor scale, Tensor bias, str data_layout = \"AnyLayout\"",
        "output": "Tensor (out)"
    },
    "affine_grid": {
        "args": "Tensor input, IntArray output_shape={}, bool align_corners=true",
        "output": "Tensor(output)"
    },
    "all": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "all_gather": {
        "args": "Tensor x, int ring_id = 0, int nranks=0",
        "output": "Tensor(out)"
    },
    "all_reduce": {
        "args": "Tensor x, int ring_id = 0, int reduce_type = 0",
        "output": "Tensor(out)"
    },
    "all_to_all": {
        "args": "Tensor x, int ring_id = 0",
        "output": "Tensor(out)"
    },
    "allclose": {
        "args": "Tensor x, Tensor y, Scalar(double) rtol=1e-5, Scalar(double) atol=1e-8, bool equal_nan=false",
        "output": "Tensor(out)"
    },
    "amax": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "amin": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "angle": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "any": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "apply_per_channel_scale": {
        "args": "Tensor x, Tensor scales",
        "output": "Tensor(out)"
    },
    "argmax": {
        "args": "Tensor x, Scalar(int64_t) axis, bool keepdims = false, bool flatten = false, DataType dtype = DataType::INT64",
        "output": "Tensor(out)"
    },
    "argmin": {
        "args": "Tensor x, Scalar(int64_t) axis, bool keepdims = false, bool flatten = false, DataType dtype = DataType::INT64",
        "output": "Tensor(out)"
    },
    "argsort": {
        "args": "Tensor x, int axis=-1, bool descending=false, bool stable=false",
        "output": "Tensor(out), Tensor(indices)"
    },
    "as_complex": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "as_real": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "as_strided": {
        "args": "Tensor input, int64_t[] dims = {}, int64_t[] stride = {}, int64_t offset = 0",
        "output": "Tensor"
    },
    "asgd_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor d, Tensor y, Tensor n, Tensor master_param, bool multi_precision=false",
        "output": "Tensor(param_out), Tensor(d_out), Tensor(y_out), Tensor(master_param_out)"
    },
    "asin": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "asinh": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "assign_out_": {
        "args": "Tensor x, Tensor output",
        "output": "Tensor(out)"
    },
    "assign_pos": {
        "args": "Tensor x, Tensor cum_count, Tensor eff_num_len",
        "output": "Tensor(out)"
    },
    "assign_value_": {
        "args": "Tensor output, int[] shape, DataType dtype, Scalar[] values, Place place = {}",
        "output": "Tensor(out)"
    },
    "atan": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "atan2": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "atanh": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "attention_lstm": {
        "args": "Tensor x, Tensor c0, Tensor h0, Tensor attention_weight, Tensor attention_bias, Tensor attention_scalar, Tensor attention_scalar_bias, Tensor lstm_weight, Tensor lstm_bias, str gate_activation = \"sigmoid\", str cell_activation = \"tanh\", str candidate_activation = \"tanh\"",
        "output": "Tensor (hidden), Tensor (cell), Tensor (attentioned_x), Tensor (attention_fc_out), Tensor (lstm_x), Tensor (lstm_out)"
    },
    "auc": {
        "args": "Tensor x, Tensor label, Tensor stat_pos, Tensor stat_neg, Tensor ins_tag_weight, str curve = \"ROC\", int num_thresholds = (2 << 12) - 1, int slide_steps = 1",
        "output": "Tensor(auc), Tensor(stat_pos_out), Tensor(stat_neg_out)"
    },
    "average_accumulates_": {
        "args": "Tensor param, Tensor in_sum_1, Tensor in_sum_2, Tensor in_sum_3, Tensor in_num_accumulates, Tensor in_old_num_accumulates, Tensor in_num_updates, float average_window = 0, int64_t max_average_window = INT64_MAX, int64_t min_average_window = 10000L",
        "output": "Tensor(out_sum_1), Tensor(out_sum_2), Tensor(out_sum_3), Tensor(out_num_accumulates), Tensor(out_old_num_accumulates), Tensor(out_num_updates)"
    },
    "baddbmm": {
        "args": "Tensor input, Tensor x, Tensor y, float beta=1.0, float alpha=1.0",
        "output": "Tensor(out)"
    },
    "barrier": {
        "args": "Tensor x, int ring_id=0",
        "output": "Tensor(out)"
    },
    "batch_fc": {
        "args": "Tensor input, Tensor w, Tensor bias",
        "output": "Tensor(out)"
    },
    "bce_loss": {
        "args": "Tensor input, Tensor label",
        "output": "Tensor"
    },
    "beam_search": {
        "args": "Tensor pre_ids, Tensor pre_scores, Tensor ids, Tensor scores, int level, int beam_size, int end_id, bool is_accumulated = true",
        "output": "Tensor (selected_ids), Tensor (selected_scores), Tensor (parent_idx)"
    },
    "bernoulli": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "bicubic_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_format=\"NCHW\", int out_d=0, int out_h=0, int out_w=0, float[] scale={}, str interp_method=\"bilinear\", bool align_corners=true, int align_mode=1",
        "output": "Tensor(output)"
    },
    "bilinear": {
        "args": "Tensor x, Tensor y, Tensor weight, Tensor bias",
        "output": "Tensor"
    },
    "bilinear_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_format=\"NCHW\", int out_d=0, int out_h=0, int out_w=0, float[] scale={}, str interp_method=\"bilinear\", bool align_corners=true, int align_mode=1",
        "output": "Tensor(output)"
    },
    "bincount": {
        "args": "Tensor x, Tensor weights, Scalar(int) minlength = 0",
        "output": "Tensor(out)"
    },
    "binomial": {
        "args": "Tensor count, Tensor prob",
        "output": "Tensor(out)"
    },
    "bipartite_match": {
        "args": "Tensor dist_mat, str match_type = \"bipartite\", float dist_threshold = 0.5",
        "output": "Tensor (col_to_row_match_indices), Tensor (col_to_row_match_dist)"
    },
    "bitwise_and": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "bitwise_left_shift": {
        "args": "Tensor x, Tensor y, bool is_arithmetic = true",
        "output": "Tensor(out)"
    },
    "bitwise_not": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "bitwise_or": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "bitwise_right_shift": {
        "args": "Tensor x, Tensor y, bool is_arithmetic = true",
        "output": "Tensor(out)"
    },
    "bitwise_xor": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "bmm": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "box_clip": {
        "args": "Tensor input, Tensor im_info",
        "output": "Tensor (output)"
    },
    "box_coder": {
        "args": "Tensor prior_box, Tensor prior_box_var, Tensor target_box, str code_type = \"encode_center_size\", bool box_normalized = true, int axis = 0, float[] variance = {}",
        "output": "Tensor(output_box)"
    },
    "broadcast": {
        "args": "Tensor x, int ring_id = 0, int root = 0",
        "output": "Tensor(out)"
    },
    "broadcast_tensors": {
        "args": "Tensor[] input",
        "output": "Tensor[]{input.size()}"
    },
    "c_allreduce_sum": {
        "args": "Tensor x, int ring_id, bool use_calc_stream, bool use_model_parallel",
        "output": "Tensor(out)"
    },
    "c_concat": {
        "args": "Tensor x, int rank, int nranks, int ring_id, bool use_calc_stream, bool use_model_parallel",
        "output": "Tensor(out)"
    },
    "c_identity": {
        "args": "Tensor x, int ring_id, bool use_calc_stream, bool use_model_parallel",
        "output": "Tensor(out)"
    },
    "c_scatter": {
        "args": "Tensor x, int ring_id = 0, int root = 0, int nranks = 0, bool use_calc_stream = false",
        "output": "Tensor(out)"
    },
    "c_softmax_with_cross_entropy": {
        "args": "Tensor logits, Tensor label,  int64_t ignore_index=-100, int ring_id=0, int rank=0, int nranks=0",
        "output": "Tensor(softmax), Tensor(loss)"
    },
    "c_split": {
        "args": "Tensor x, int rank = 0, int nranks = 1, int ring_id = 0, bool use_model_parallel = true",
        "output": "Tensor(out)"
    },
    "calc_reduced_attn_scores": {
        "args": "Tensor q, Tensor k, Tensor softmax_lse",
        "output": "Tensor(reduced_scores)"
    },
    "cast": {
        "args": "Tensor x, DataType dtype",
        "output": "Tensor(out)"
    },
    "ceil": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "celu": {
        "args": "Tensor x, float alpha = 1.0",
        "output": "Tensor(out)"
    },
    "channel_shuffle": {
        "args": "Tensor x, int groups, str data_format=\"NCHW\"",
        "output": "Tensor(out)"
    },
    "check_finite_and_unscale_": {
        "args": "Tensor[] x, Tensor scale",
        "output": "Tensor[](out){x.size()}, Tensor(found_infinite)"
    },
    "check_numerics": {
        "args": "Tensor tensor, str op_type = \"\", str var_name = \"\", int check_nan_inf_level = 0, int stack_height_limit = -1, str output_dir = \"\"",
        "output": "Tensor(stats), Tensor(values)"
    },
    "cholesky": {
        "args": "Tensor x, bool upper=false",
        "output": "Tensor"
    },
    "cholesky_solve": {
        "args": "Tensor x, Tensor y, bool upper=false",
        "output": "Tensor"
    },
    "class_center_sample": {
        "args": "Tensor label, int num_classes, int num_samples, int ring_id = 0, int rank = 0, int nranks = 1, bool fix_seed = false, int seed = 0",
        "output": "Tensor(remapped_label), Tensor(sampled_local_class_center)"
    },
    "clip": {
        "args": "Tensor x, Scalar(float) min, Scalar(float) max",
        "output": "Tensor(out)"
    },
    "clip_by_norm": {
        "args": "Tensor x, float max_norm",
        "output": "Tensor(out)"
    },
    "coalesce_tensor": {
        "args": "Tensor[] input, DataType dtype, bool copy_data = false, bool set_constant = false, bool persist_output = false, float constant = 0.0, bool use_align = true, int align_size = -1, int size_of_dtype = -1, int64_t[] concated_shapes = {}, int64_t[] concated_ranks = {}",
        "output": "Tensor[](output){input.size()}, Tensor(fused_output)"
    },
    "collect_fpn_proposals": {
        "args": "Tensor[] multi_level_rois, Tensor[] multi_level_scores, Tensor[] multi_level_rois_num, int post_nms_topn",
        "output": "Tensor (fpn_rois), Tensor (rois_num)"
    },
    "complex": {
        "args": "Tensor real, Tensor imag",
        "output": "Tensor"
    },
    "concat": {
        "args": "Tensor[] x, Scalar axis=0",
        "output": "Tensor"
    },
    "conj": {
        "args": "Tensor x",
        "output": "Tensor (out)"
    },
    "conv2d": {
        "args": "Tensor input, Tensor filter, int[] strides={1, 1}, int[] paddings={0, 0}, str padding_algorithm=\"EXPLICIT\", int[] dilations={1, 1}, int groups=1, str data_format=\"NCHW\"",
        "output": "Tensor"
    },
    "conv2d_transpose": {
        "args": "Tensor x, Tensor filter, int[] strides={1, 1}, int[] paddings={0, 0}, int[] output_padding={}, IntArray output_size={}, str padding_algorithm=\"EXPLICIT\", int groups=1, int[] dilations={1, 1}, str data_format=\"NCHW\"",
        "output": "Tensor(out)"
    },
    "conv2d_transpose_bias": {
        "args": "Tensor x, Tensor filter, Tensor bias, int[] strides={1, 1}, int[] paddings={0, 0}, int[] output_padding={}, IntArray output_size={}, str padding_algorithm=\"EXPLICIT\", int groups=1, int[] dilations={1, 1}, str data_format=\"NCHW\"",
        "output": "Tensor(out)"
    },
    "conv3d": {
        "args": "Tensor input, Tensor filter, int[] strides={1, 1, 1}, int[] paddings={0, 0, 0}, str padding_algorithm=\"EXPLICIT\", int groups=1, int[] dilations={1, 1, 1}, str data_format=\"NCDHW\"",
        "output": "Tensor"
    },
    "conv3d_transpose": {
        "args": "Tensor x, Tensor filter, int[] strides={1, 1, 1}, int[] paddings={0, 0, 0}, int[] output_padding={}, int[] output_size={}, str padding_algorithm=\"EXPLICIT\", int groups=1, int[] dilations={1, 1, 1}, str data_format=\"NCHW\"",
        "output": "Tensor(out)"
    },
    "copy_to": {
        "args": "Tensor x, Place place, bool blocking",
        "output": "Tensor(out)"
    },
    "copysign": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "correlation": {
        "args": "Tensor input1, Tensor input2, int pad_size, int kernel_size, int max_displacement, int stride1, int stride2, int corr_type_multiply=1",
        "output": "Tensor(out)"
    },
    "cos": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "cosh": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "crf_decoding": {
        "args": "Tensor emission, Tensor transition, Tensor label, Tensor length",
        "output": "Tensor (viterbi_path)"
    },
    "crop": {
        "args": "Tensor x, IntArray shape = {}, IntArray offsets = {}",
        "output": "Tensor(out)"
    },
    "cross": {
        "args": "Tensor x, Tensor y, int axis = 9",
        "output": "Tensor"
    },
    "cross_entropy_with_softmax": {
        "args": "Tensor input, Tensor label, bool soft_label=false, bool use_softmax=true, bool numeric_stable_mode=true, int ignore_index=-100, int axis=-1",
        "output": "Tensor(softmax), Tensor(loss)"
    },
    "ctc_align": {
        "args": "Tensor input, Tensor input_length, int blank = 0, bool merge_repeated = true, int padding_value = 0",
        "output": "Tensor (output), Tensor (output_length)"
    },
    "cudnn_lstm": {
        "args": "Tensor x, Tensor init_h, Tensor init_c, Tensor w, Tensor[] weight_list, Tensor sequence_length, float dropout_prob = 0.0, bool is_bidirec = false, int hidden_size = 100, int num_layers = 1, bool is_test = false, int seed = 0",
        "output": "Tensor (out), Tensor (last_h), Tensor (last_c), Tensor (reserve), Tensor (state_out)"
    },
    "cummax": {
        "args": "Tensor x, int axis=-1, DataType dtype = DataType::INT64",
        "output": "Tensor(out), Tensor(indices)"
    },
    "cummin": {
        "args": "Tensor x, int axis=-1, DataType dtype = DataType::INT64",
        "output": "Tensor(out), Tensor(indices)"
    },
    "cumprod": {
        "args": "Tensor x,  int dim, bool exclusive=false, bool reverse=false",
        "output": "Tensor(out)"
    },
    "cumsum": {
        "args": "Tensor x, Scalar axis=-1, bool flatten=false, bool exclusive=false, bool reverse=false",
        "output": "Tensor(out)"
    },
    "cvm": {
        "args": "Tensor x, Tensor cvm, bool use_cvm = true",
        "output": "Tensor (out)"
    },
    "data": {
        "args": "str name, IntArray shape, DataType dtype, Place place",
        "output": "Tensor(out)"
    },
    "decayed_adagrad": {
        "args": "Tensor param, Tensor grad, Tensor moment, Tensor learning_rate, float decay = 0.95f, float epsilon = 1.0e-6f",
        "output": "Tensor(param_out), Tensor(moment_out)"
    },
    "decode_jpeg": {
        "args": "Tensor x, str mode, Place place",
        "output": "Tensor(out)"
    },
    "deformable_conv": {
        "args": "Tensor x, Tensor offset, Tensor filter, Tensor mask, int[] strides, int[] paddings, int[] dilations, int deformable_groups, int groups, int im2col_step",
        "output": "Tensor(out)"
    },
    "depend": {
        "args": "Tensor x, Tensor[] dep",
        "output": "Tensor (out)"
    },
    "depthwise_conv2d": {
        "args": "Tensor input, Tensor filter, int[] strides={1, 1}, int[] paddings={0, 0}, str padding_algorithm=\"EXPLICIT\", int groups=1, int[] dilations={1, 1}, str data_format=\"NCHW\"",
        "output": "Tensor(out)"
    },
    "depthwise_conv2d_transpose": {
        "args": "Tensor x, Tensor filter, int[] strides={1, 1}, int[] paddings={0, 0}, int[] output_padding={}, IntArray output_size={}, str padding_algorithm=\"EXPLICIT\", int groups=1, int[] dilations={1, 1}, str data_format=\"NCHW\"",
        "output": "Tensor(out)"
    },
    "dequantize_abs_max": {
        "args": "Tensor x, Tensor scale, float max_range",
        "output": "Tensor(out)"
    },
    "dequantize_log": {
        "args": "Tensor x, Tensor dict",
        "output": "Tensor(out)"
    },
    "det": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "dgc": {
        "args": "Tensor u, Tensor v, Tensor grad, Tensor param, Tensor current_step, Tensor nranks, float m=0.9, bool use_nesterov=true, float[] sparsity={}, float rampup_begin_step=0.0, float rampup_step=0.0, float regular_coeff=0.0, int regular_type=0",
        "output": "Tensor(u_out), Tensor(v_out), Tensor(encode_grad), Tensor(grad_out), Tensor(k), Tensor(gather_buff)"
    },
    "dgc_clip_by_norm": {
        "args": "Tensor x, Tensor current_step, float max_norm, float rampup_begin_step = -1.0",
        "output": "Tensor(out)"
    },
    "dgc_momentum": {
        "args": "Tensor param, Tensor grad, Tensor velocity, Tensor learning_rate, Tensor master_param, Tensor current_step_tensor, Tensor nranks_tensor, float mu, bool use_nesterov = false, str regularization_method = \"\", float regularization_coeff = 0.0f, bool multi_precision = false, float rescale_grad = 1.0f, float rampup_begin_step = -1.0",
        "output": "Tensor (param_out), Tensor (velocity_out), Tensor (master_param_out), Tensor (grad_out)"
    },
    "diag": {
        "args": "Tensor x, int offset = 0, float padding_value = 0.0",
        "output": "Tensor"
    },
    "diag_embed": {
        "args": "Tensor input, int offset = 0, int dim1 = -2, int dim2 = -1",
        "output": "Tensor(out)"
    },
    "diagonal": {
        "args": "Tensor x, int offset = 0, int axis1 = 0, int axis2 = 1",
        "output": "Tensor"
    },
    "digamma": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "dirichlet": {
        "args": "Tensor alpha",
        "output": "Tensor(out)"
    },
    "disable_check_model_nan_inf": {
        "args": "Tensor x, int flag = 0",
        "output": "Tensor(out)"
    },
    "dist": {
        "args": "Tensor x, Tensor y, float p = 2.0",
        "output": "Tensor"
    },
    "dot": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "dpsgd": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, float clip = 10.0f, float batch_size = 16.0f, float sigma = 1.0f, int seed = 0",
        "output": "Tensor(param_out)"
    },
    "dropout": {
        "args": "Tensor x, Tensor seed_tensor, Scalar p, bool is_test, str mode, int seed, bool fix_seed",
        "output": "Tensor(out), Tensor(mask)"
    },
    "edit_distance": {
        "args": "Tensor hyps, Tensor refs, Tensor hypslength, Tensor refslength, bool normalized = false",
        "output": "Tensor(sequencenum), Tensor(out)"
    },
    "eig": {
        "args": "Tensor x",
        "output": "Tensor(out_w), Tensor(out_v)"
    },
    "eigh": {
        "args": "Tensor x, str UPLO = \"L\"",
        "output": "Tensor(out_w), Tensor(out_v)"
    },
    "eigvals": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "eigvalsh": {
        "args": "Tensor x, str uplo = \"L\", bool is_test = false",
        "output": "Tensor(eigenvalues), Tensor(eigenvectors)"
    },
    "elu": {
        "args": "Tensor x, float alpha = 1.0f",
        "output": "Tensor(out)"
    },
    "embedding_with_scaled_gradient": {
        "args": "Tensor x, Tensor weight, int64_t padding_idx=-1",
        "output": "Tensor"
    },
    "empty": {
        "args": "IntArray shape, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "empty_like": {
        "args": "Tensor x, DataType dtype = DataType::UNDEFINED, Place place = {}",
        "output": "Tensor(out)"
    },
    "enable_check_model_nan_inf": {
        "args": "Tensor x, int flag = 1",
        "output": "Tensor(out)"
    },
    "equal_all": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "erf": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "erfinv": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "exp": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "expand": {
        "args": "Tensor x, IntArray shape = {}",
        "output": "Tensor(out)"
    },
    "expand_as": {
        "args": "Tensor x, Tensor y, int[] target_shape = {}",
        "output": "Tensor(out)"
    },
    "expm1": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "exponential_": {
        "args": "Tensor x, float lam",
        "output": "Tensor(out)"
    },
    "eye": {
        "args": "Scalar num_rows, Scalar num_columns, DataType dtype=DataType::FLOAT32, Place place={}",
        "output": "Tensor(out)"
    },
    "fake_channel_wise_dequantize_max_abs": {
        "args": "Tensor x, Tensor[] scales, int[] quant_bits = {8}, int quant_axis = 0, int x_num_col_dims = 1",
        "output": "Tensor(out)"
    },
    "fake_channel_wise_quantize_abs_max": {
        "args": "Tensor x, int bit_length = 8, int round_type = 1, int quant_axis = 0, bool is_test = false",
        "output": "Tensor(out), Tensor(out_scale)"
    },
    "fake_channel_wise_quantize_dequantize_abs_max": {
        "args": "Tensor x, int bit_length = 8, int round_type = 1, int quant_axis = 0",
        "output": "Tensor(out), Tensor(out_scale)"
    },
    "fake_dequantize_max_abs": {
        "args": "Tensor x, Tensor scale, float max_range",
        "output": "Tensor(out)"
    },
    "fake_quantize_abs_max": {
        "args": "Tensor x, int bit_length = 8, int round_type = 1",
        "output": "Tensor(out), Tensor(out_scale)"
    },
    "fake_quantize_dequantize_abs_max": {
        "args": "Tensor x, int bit_length = 8, int round_type = 1",
        "output": "Tensor(out), Tensor(out_scale)"
    },
    "fake_quantize_dequantize_moving_average_abs_max": {
        "args": "Tensor x, Tensor in_scale, Tensor in_accum, Tensor in_state, float moving_rate = 0.9, int bit_length = 8, bool is_test = false, int round_type = 1",
        "output": "Tensor(out), Tensor(out_scale), Tensor(out_state), Tensor(out_accum)"
    },
    "fake_quantize_moving_average_abs_max": {
        "args": "Tensor x, Tensor in_scale, Tensor in_accum, Tensor in_state, float moving_rate = 0.9, int bit_length = 8, bool is_test = false, int round_type = 1",
        "output": "Tensor(out), Tensor(out_scale), Tensor(out_state), Tensor(out_accum)"
    },
    "fake_quantize_range_abs_max": {
        "args": "Tensor x, Tensor in_scale, Tensor iter, int window_size = 10000,  int bit_length = 8, bool is_test = false, int round_type = 1",
        "output": "Tensor(out), Tensor(out_scale), Tensor(out_scales)"
    },
    "fft_c2c": {
        "args": "Tensor x, int64_t[] axes, str normalization, bool forward",
        "output": "Tensor"
    },
    "fft_c2r": {
        "args": "Tensor x, int64_t[] axes, str normalization, bool forward, int64_t last_dim_size=0L",
        "output": "Tensor"
    },
    "fft_r2c": {
        "args": "Tensor x, int64_t[] axes, str normalization, bool forward, bool onesided",
        "output": "Tensor"
    },
    "fill": {
        "args": "Tensor x, Scalar(double) value=0",
        "output": "Tensor(out)"
    },
    "fill_diagonal": {
        "args": "Tensor x, float value=0, int offset=0, bool wrap=false",
        "output": "Tensor(out)"
    },
    "fill_diagonal_tensor": {
        "args": "Tensor x, Tensor y, int64_t offset = 0, int dim1 = 0, int dim2 = 1",
        "output": "Tensor(out)"
    },
    "flash_attn": {
        "args": "Tensor q, Tensor k, Tensor v, Tensor fixed_seed_offset, Tensor attn_mask, float dropout = 0.0, bool causal = false, bool return_softmax = false, bool is_test = false, str rng_name = \"\"",
        "output": "Tensor(out), Tensor(softmax), Tensor(softmax_lse), Tensor(seed_offset)"
    },
    "flash_attn_qkvpacked": {
        "args": "Tensor qkv, Tensor fixed_seed_offset, Tensor attn_mask, float dropout = 0.0, bool causal = false, bool return_softmax = false, bool is_test = false, str rng_name = \"\"",
        "output": "Tensor(out), Tensor(softmax), Tensor(softmax_lse), Tensor(seed_offset)"
    },
    "flash_attn_unpadded": {
        "args": "Tensor q, Tensor k, Tensor v, Tensor cu_seqlens_q,  Tensor cu_seqlens_k, Tensor fixed_seed_offset, Tensor attn_mask, Scalar max_seqlen_q, Scalar max_seqlen_k, float scale, float dropout = 0.0, bool causal = false, bool return_softmax = false, bool is_test = false, str rng_name = \"\"",
        "output": "Tensor(out), Tensor(softmax), Tensor(softmax_lse), Tensor(seed_offset)"
    },
    "flash_attn_varlen_qkvpacked": {
        "args": "Tensor qkv, Tensor cu_seqlens_q,  Tensor cu_seqlens_k, Tensor fixed_seed_offset, Tensor attn_mask, Scalar max_seqlen_q, Scalar max_seqlen_k, float scale, float dropout = 0.0, bool causal = false, bool return_softmax = false, bool is_test = false, str rng_name = \"\", bool varlen_padded = true",
        "output": "Tensor(out), Tensor(softmax), Tensor(softmax_lse), Tensor(seed_offset)"
    },
    "flashmask_attention": {
        "args": "Tensor q, Tensor k, Tensor v, Tensor startend_row_indices,  Tensor fixed_seed_offset, float dropout = 0.0, bool causal = false, bool return_softmax = false, bool is_test = false, str rng_name = \"\"",
        "output": "Tensor(out), Tensor(softmax), Tensor(softmax_lse), Tensor(seed_offset)"
    },
    "flatten": {
        "args": "Tensor x, int start_axis = 1, int stop_axis = 1",
        "output": "Tensor(out)"
    },
    "flip": {
        "args": "Tensor x, int[] axis",
        "output": "Tensor (out)"
    },
    "floor": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "fmax": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "fmin": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "fold": {
        "args": "Tensor x, int[] output_sizes, int[] kernel_sizes,  int[] strides, int[] paddings, int[] dilations",
        "output": "Tensor(out)"
    },
    "fractional_max_pool2d": {
        "args": "Tensor x, int[] output_size, int[] kernel_size = {0, 0}, float random_u = 0.0, bool return_mask = true",
        "output": "Tensor(out), Tensor(mask)"
    },
    "fractional_max_pool3d": {
        "args": "Tensor x, int[] output_size, int[] kernel_size = {0, 0, 0}, float random_u = 0.0, bool return_mask = true",
        "output": "Tensor(out), Tensor(mask)"
    },
    "frame": {
        "args": "Tensor x, int frame_length, int hop_length, int axis=-1",
        "output": "Tensor(out)"
    },
    "frobenius_norm": {
        "args": "Tensor x, IntArray axis,  bool keep_dim,  bool reduce_all",
        "output": "Tensor(out)"
    },
    "ftrl": {
        "args": "Tensor param, Tensor squared_accumulator, Tensor linear_accumulator, Tensor grad, Tensor learning_rate, float l1=0.0f, float l2=0.0f, float lr_power=-0.5f",
        "output": "Tensor(param_out), Tensor(squared_accum_out), Tensor(linear_accum_out)"
    },
    "full": {
        "args": "IntArray shape, Scalar(double) value, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "full_": {
        "args": "Tensor output, IntArray shape, Scalar(double) value, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "full_batch_size_like": {
        "args": "Tensor input, int[] shape, DataType dtype, Scalar(double) value, int input_dim_idx, int output_dim_idx, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "full_int_array": {
        "args": "int64_t[] value, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "full_like": {
        "args": "Tensor x, Scalar value, DataType dtype = DataType::UNDEFINED, Place place = {}",
        "output": "Tensor(out)"
    },
    "full_with_tensor": {
        "args": "Tensor value, IntArray shape, DataType dtype=DataType::FLOAT32",
        "output": "Tensor(out)"
    },
    "fused_batch_norm_act": {
        "args": "Tensor x, Tensor scale, Tensor bias, Tensor mean, Tensor variance, float momentum, float epsilon, str act_type",
        "output": "Tensor(out), Tensor(mean_out), Tensor(variance_out), Tensor(saved_mean), Tensor(saved_variance), Tensor(reserve_space)"
    },
    "fused_bn_add_activation": {
        "args": "Tensor x, Tensor z, Tensor scale, Tensor bias, Tensor mean, Tensor variance, float momentum, float epsilon, str act_type",
        "output": "Tensor(out), Tensor(mean_out), Tensor(variance_out), Tensor(saved_mean), Tensor(saved_variance), Tensor(reserve_space)"
    },
    "fused_softmax_mask": {
        "args": "Tensor x, Tensor mask",
        "output": "Tensor(out)"
    },
    "fused_softmax_mask_upper_triangle": {
        "args": "Tensor X",
        "output": "Tensor(Out)"
    },
    "gammaincc": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "gammaln": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "gather": {
        "args": "Tensor x, Tensor index, Scalar axis=0",
        "output": "Tensor(out)"
    },
    "gather_nd": {
        "args": "Tensor x, Tensor index",
        "output": "Tensor(out)"
    },
    "gather_tree": {
        "args": "Tensor ids, Tensor parents",
        "output": "Tensor(out)"
    },
    "gaussian": {
        "args": "IntArray shape, float mean, float std, int seed, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "gaussian_inplace": {
        "args": "Tensor x, float mean=0, float std=1.0, int seed=0",
        "output": "Tensor(out)"
    },
    "gelu": {
        "args": "Tensor x,  bool approximate = false",
        "output": "Tensor(out)"
    },
    "generate_proposals": {
        "args": "Tensor scores, Tensor bbox_deltas, Tensor im_shape, Tensor anchors, Tensor variances, int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size, float eta, bool pixel_offset=true",
        "output": "Tensor(rpn_rois), Tensor(rpn_roi_probs), Tensor(rpn_rois_num)"
    },
    "global_gather": {
        "args": "Tensor x, Tensor local_count, Tensor global_count, int ring_id = 0",
        "output": "Tensor(out)"
    },
    "global_scatter": {
        "args": "Tensor x, Tensor local_count, Tensor global_count, int ring_id = 0",
        "output": "Tensor(out)"
    },
    "graph_khop_sampler": {
        "args": "Tensor row, Tensor colptr, Tensor x, Tensor eids, int[] sample_sizes, bool return_eids",
        "output": "Tensor(out_src), Tensor(out_dst), Tensor(sample_index), Tensor(reindex_x), Tensor(out_eids)"
    },
    "graph_sample_neighbors": {
        "args": "Tensor row, Tensor colptr, Tensor x, Tensor eids, Tensor perm_buffer, int sample_size, bool return_eids, bool flag_perm_buffer",
        "output": "Tensor(out), Tensor(out_count), Tensor(out_eids)"
    },
    "grid_sample": {
        "args": "Tensor x, Tensor grid, str mode = \"bilinear\", str padding_mode = \"zeros\", bool align_corners = true",
        "output": "Tensor(out)"
    },
    "group_norm": {
        "args": "Tensor x, Tensor scale, Tensor bias, float epsilon = 1e-5, int groups = -1, str data_format = \"NCHW\"",
        "output": "Tensor(y), Tensor(mean), Tensor(variance)"
    },
    "gru": {
        "args": "Tensor input, Tensor h0, Tensor weight, Tensor bias, str activation = \"tanh\", str gate_activation = \"sigmoid\", bool is_reverse = false, bool origin_mode = false, bool is_test=false",
        "output": "Tensor (batch_gate), Tensor (batch_reset_hidden_prev), Tensor (batch_hidden), Tensor (hidden)"
    },
    "gru_unit": {
        "args": "Tensor input, Tensor hidden_prev, Tensor weight, Tensor bias, int activation = 2, int gate_activation = 1, bool origin_mode = false",
        "output": "Tensor (gate), Tensor (reset_hidden_prev), Tensor (hidden)"
    },
    "gumbel_softmax": {
        "args": "Tensor x, float temperature = 1.0, bool hard = false, int axis = -1",
        "output": "Tensor"
    },
    "hardshrink": {
        "args": "Tensor x, float threshold = 0.5",
        "output": "Tensor (out)"
    },
    "hardsigmoid": {
        "args": "Tensor x, float slope = 0.2, float offset = 0.5",
        "output": "Tensor (out)"
    },
    "hardtanh": {
        "args": "Tensor x, float t_min=0, float t_max=24",
        "output": "Tensor(out)"
    },
    "heaviside": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "hinge_loss": {
        "args": "Tensor logits, Tensor labels",
        "output": "Tensor (loss)"
    },
    "histogram": {
        "args": "Tensor input, Tensor weight, int64_t bins = 100, float min = 0.0, float max = 0.0, bool density = false",
        "output": "Tensor(out)"
    },
    "hsigmoid_loss": {
        "args": "Tensor x, Tensor label, Tensor w, Tensor bias, Tensor path, Tensor code, int num_classes, bool is_sparse",
        "output": "Tensor(out), Tensor(pre_out), Tensor(w_out)"
    },
    "huber_loss": {
        "args": "Tensor input, Tensor label, float delta",
        "output": "Tensor(out), Tensor(residual)"
    },
    "i0": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "i0e": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "i1": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "i1e": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "identity_loss": {
        "args": "Tensor x, int reduction = 1",
        "output": "Tensor(out)"
    },
    "im2sequence": {
        "args": "Tensor x, Tensor y, int[] kernels, int[] strides = {1, 1}, int[] paddings = {0, 0, 0, 0}, int[] out_stride = {1, 1}",
        "output": "Tensor (out)"
    },
    "imag": {
        "args": "Tensor x",
        "output": "Tensor (out)"
    },
    "increment": {
        "args": "Tensor x, float value = 1.0",
        "output": "Tensor(out)"
    },
    "index_add": {
        "args": "Tensor x, Tensor index,  Tensor add_value, int axis = 0",
        "output": "Tensor(out)"
    },
    "index_put": {
        "args": "Tensor x, Tensor[] indices, Tensor value, bool accumulate=false",
        "output": "Tensor(out)"
    },
    "index_sample": {
        "args": "Tensor x, Tensor index",
        "output": "Tensor"
    },
    "index_select": {
        "args": "Tensor x, Tensor index, int axis = 0",
        "output": "Tensor(out)"
    },
    "index_select_strided": {
        "args": "Tensor x, int64_t index, int axis = 0",
        "output": "Tensor(out)"
    },
    "instance_norm": {
        "args": "Tensor x, Tensor scale, Tensor bias, float epsilon=1e-5",
        "output": "Tensor(y), Tensor(saved_mean), Tensor(saved_variance)"
    },
    "inverse": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "is_empty": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "isclose": {
        "args": "Tensor x, Tensor y, Scalar(double) rtol=1e-5, Scalar(double) atol=1e-8,  bool equal_nan=false",
        "output": "Tensor(out)"
    },
    "isfinite": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "isinf": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "isnan": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "kldiv_loss": {
        "args": "Tensor x, Tensor label, str reduction = \"mean\", bool log_target = false",
        "output": "Tensor(out)"
    },
    "kron": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "kthvalue": {
        "args": "Tensor x, int k = 1, int axis = -1, bool keepdim = false",
        "output": "Tensor(out), Tensor(indices)"
    },
    "l1_norm": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "label_smooth": {
        "args": "Tensor label, Tensor prior_dist, float epsilon = 0.0f",
        "output": "Tensor (out)"
    },
    "lamb_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment1, Tensor moment2, Tensor beta1_pow, Tensor beta2_pow, Tensor master_param, Tensor skip_update, float weight_decay, float beta1=0.9, float beta2=0.999, float epsilon=1.0e-6f, bool always_adapt=false, bool multi_precision=false",
        "output": "Tensor(param_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(beta1_pow_out), Tensor(beta2_pow_out), Tensor(master_param_outs)"
    },
    "layer_norm": {
        "args": "Tensor x, Tensor scale, Tensor bias, float epsilon = 1e-5, int begin_norm_axis = 1",
        "output": "Tensor(out), Tensor(mean), Tensor(variance)"
    },
    "leaky_relu": {
        "args": "Tensor x, float negative_slope = 0.02f",
        "output": "Tensor(out)"
    },
    "lerp": {
        "args": "Tensor x, Tensor y, Tensor weight",
        "output": "Tensor(out)"
    },
    "lgamma": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "limit_by_capacity": {
        "args": "Tensor expert_count, Tensor capacity, int n_worker",
        "output": "Tensor(out)"
    },
    "linear_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_format=\"NCHW\", int out_d=0, int out_h=0, int out_w=0, float[] scale={}, str interp_method=\"bilinear\", bool align_corners=true, int align_mode=1",
        "output": "Tensor(output)"
    },
    "linspace": {
        "args": "Tensor start, Tensor stop, Tensor number, DataType dtype, Place place",
        "output": "Tensor(out)"
    },
    "llm_int8_linear": {
        "args": "Tensor x, Tensor weight, Tensor bias, Tensor weight_scale, float threshold=6.0",
        "output": "Tensor(out)"
    },
    "log": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "log10": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "log1p": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "log2": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "log_loss": {
        "args": "Tensor input, Tensor label, float epsilon",
        "output": "Tensor"
    },
    "log_softmax": {
        "args": "Tensor x, int axis = -1",
        "output": "Tensor(out)"
    },
    "logcumsumexp": {
        "args": "Tensor x, int axis=-1, bool flatten=false, bool exclusive=false, bool reverse=false",
        "output": "Tensor(out)"
    },
    "logical_and": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "logical_not": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "logical_or": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "logical_xor": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "logit": {
        "args": "Tensor x, float eps = 1e-6f",
        "output": "Tensor(out)"
    },
    "logsigmoid": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "logspace": {
        "args": "Tensor start, Tensor stop, Tensor num, Tensor base, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "logsumexp": {
        "args": "Tensor x, int[] axis={0},  bool keepdim=false,  bool reduce_all=false",
        "output": "Tensor(out)"
    },
    "lookup_table_dequant": {
        "args": "Tensor w, Tensor ids, int64_t padding_idx = -1",
        "output": "Tensor (out)"
    },
    "lp_pool2d": {
        "args": "Tensor x, IntArray kernel_size, int[] strides = {1,1}, int[] paddings = {0,0}, bool ceil_mode = false, bool exclusive = true, str data_format = \"NCHW\", str pooling_type = \"\", bool global_pooling = false, bool adaptive = false, str padding_algorithm = \"EXPLICIT\", float norm_type = 0.0f",
        "output": "Tensor(out)"
    },
    "lstm": {
        "args": "Tensor input, Tensor h0, Tensor c0, Tensor weight, Tensor bias, bool use_peepholes = true, bool is_reverse = false, bool is_test = false, str gate_activation = \"sigmoid\", str cell_activation = \"tanh\", str candidate_activation = \"tanh\"",
        "output": "Tensor (hidden), Tensor (cell), Tensor (batch_gate), Tensor (batch_cell_pre_act)"
    },
    "lstsq": {
        "args": "Tensor x, Tensor y, Scalar rcond=0.0f, str driver=\"gels\"",
        "output": "Tensor(solution), Tensor(residuals), Tensor(rank), Tensor(singular_values)"
    },
    "lu": {
        "args": "Tensor x, bool pivot = true",
        "output": "Tensor(out), Tensor(pivots), Tensor(infos)"
    },
    "lu_solve": {
        "args": "Tensor b, Tensor lu, Tensor pivots, str trans",
        "output": "Tensor(out)"
    },
    "lu_unpack": {
        "args": "Tensor x, Tensor y, bool unpack_ludata = true, bool unpack_pivots = true",
        "output": "Tensor(pmat), Tensor(l), Tensor(u)"
    },
    "margin_cross_entropy": {
        "args": "Tensor logits, Tensor label, bool return_softmax = false, int ring_id = 0, int rank = 0, int nranks = 1, float margin1 = 1.0f, float margin2 = 0.5f, float margin3 = 0.0f, float scale = 64.0f",
        "output": "Tensor(softmax), Tensor(loss)"
    },
    "masked_multihead_attention_": {
        "args": "Tensor x, Tensor cache_kv, Tensor bias, Tensor src_mask, Tensor cum_offsets, Tensor sequence_lengths, Tensor rotary_tensor, Tensor beam_cache_offset, Tensor qkv_out_scale, Tensor out_shift, Tensor out_smooth, int seq_len, int rotary_emb_dims, bool use_neox_rotary_style=false, str compute_dtype = \"default\", float out_scale=-1, int quant_round_type=1, float quant_max_bound=127.0, float quant_min_bound=-127.0",
        "output": "Tensor(out), Tensor(cache_kv_out), Tensor(beam_cache_offset_out)"
    },
    "masked_select": {
        "args": "Tensor x, Tensor mask",
        "output": "Tensor (out)"
    },
    "match_matrix_tensor": {
        "args": "Tensor x, Tensor y, Tensor w, int dim_t = 1",
        "output": "Tensor (out), Tensor (tmp)"
    },
    "matrix_nms": {
        "args": "Tensor bboxes, Tensor scores, float score_threshold, int nms_top_k, int keep_top_k, float post_threshold=0., bool use_gaussian = false, float gaussian_sigma = 2., int background_label = 0, bool normalized = true",
        "output": "Tensor(out), Tensor(index), Tensor(roisnum)"
    },
    "matrix_power": {
        "args": "Tensor x, int n",
        "output": "Tensor"
    },
    "matrix_rank": {
        "args": "Tensor x, float tol, bool use_default_tol=true, bool hermitian=false",
        "output": "Tensor(out)"
    },
    "matrix_rank_atol_rtol": {
        "args": "Tensor x, Tensor atol, Tensor rtol, bool hermitian=false",
        "output": "Tensor(out)"
    },
    "matrix_rank_tol": {
        "args": "Tensor x, Tensor atol_tensor, bool use_default_tol=true, bool hermitian=false",
        "output": "Tensor(out)"
    },
    "max": {
        "args": "Tensor x, IntArray axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "max_pool2d_with_index": {
        "args": "Tensor x, int[] kernel_size, int[] strides= {1, 1}, int[] paddings = {0, 0}, bool global_pooling = false, bool adaptive = false, bool ceil_mode = false",
        "output": "Tensor(out), Tensor(mask)"
    },
    "max_pool3d_with_index": {
        "args": "Tensor x, int[] kernel_size, int[] strides = {1, 1, 1}, int[] paddings = {0, 0, 0}, bool global_pooling = false, bool adaptive = false, bool ceil_mode = false",
        "output": "Tensor(out), Tensor(mask)"
    },
    "maxout": {
        "args": "Tensor x, int groups, int axis = 1",
        "output": "Tensor(out)"
    },
    "mean": {
        "args": "Tensor x, IntArray axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "mean_all": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "memcpy_d2h": {
        "args": "Tensor x, int dst_place_type",
        "output": "Tensor"
    },
    "memcpy_h2d": {
        "args": "Tensor x, int dst_place_type",
        "output": "Tensor"
    },
    "memory_efficient_attention": {
        "args": "Tensor query, Tensor key, Tensor value, Tensor bias, Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor causal_diagonal, Tensor seqlen_k, Scalar max_seqlen_q, Scalar max_seqlen_k, bool causal, double dropout_p, float scale, bool is_test",
        "output": "Tensor(output), Tensor(logsumexp), Tensor(seed_and_offset)"
    },
    "merge_selected_rows": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "merged_adam_": {
        "args": "Tensor[] param, Tensor[] grad, Tensor[] learning_rate, Tensor[] moment1, Tensor[] moment2, Tensor[] moment2_max, Tensor[] beta1_pow, Tensor[] beta2_pow, Tensor[] master_param, Scalar beta1 = 0.9f, Scalar beta2 = 0.999f, Scalar epsilon = 1.0e-8f, bool multi_precision = false, bool use_global_beta_pow = false, bool amsgrad = false",
        "output": "Tensor[](param_out){param.size()}, Tensor[](moment1_out){param.size()}, Tensor[](moment2_out){param.size()}, Tensor[](moment2_max_out){param.size()}, Tensor[](beta1_pow_out){param.size()}, Tensor[](beta2_pow_out){param.size()}, Tensor[](master_param_out){param.size()}"
    },
    "merged_momentum_": {
        "args": "Tensor[] param, Tensor[] grad, Tensor[] velocity, Tensor[] learning_rate, Tensor[] master_param, float mu, bool use_nesterov = false, str[] regularization_method = {}, float[] regularization_coeff = {}, bool multi_precision = false, float rescale_grad = 1.0f",
        "output": "Tensor[](param_out){param.size()}, Tensor[](velocity_out){param.size()}, Tensor[](master_param_out){param.size()}"
    },
    "meshgrid": {
        "args": "Tensor[] inputs",
        "output": "Tensor[](out){inputs.size()}"
    },
    "mish": {
        "args": "Tensor x, float lambda",
        "output": "Tensor"
    },
    "mode": {
        "args": "Tensor x,  int axis = -1,  bool keepdim = false",
        "output": "Tensor(out), Tensor(indices)"
    },
    "moe_dispatch": {
        "args": "Tensor X, Tensor gating_output, int moe_topk = 2, bool group_moe = false, bool topk_only_mode = false",
        "output": "Tensor(permute_input), Tensor(token_nums_per_expert), Tensor(permute_indices_per_token), Tensor(expert_scales_float), Tensor(top_k_indices)"
    },
    "moe_ffn": {
        "args": "Tensor permute_input, Tensor token_nums_per_expert, Tensor ffn1_weight, Tensor ffn2_weight, Tensor ffn1_bias, Tensor ffn1_scale, Tensor ffn2_scale, str quant_method",
        "output": "Tensor(ffn_out)"
    },
    "moe_reduce": {
        "args": "Tensor ffn_out, Tensor expert_scales_float, Tensor permute_indices_per_token, Tensor top_k_indices, Tensor ffn2_bias, bool norm_topk_prob = false, float routed_scaling_factor = 1.0",
        "output": "Tensor(output)"
    },
    "momentum_": {
        "args": "Tensor param, Tensor grad, Tensor velocity, Tensor learning_rate, Tensor master_param, float mu, bool use_nesterov = false, str regularization_method = \"\", float regularization_coeff = 0.0f, bool multi_precision = false, float rescale_grad = 1.0f",
        "output": "Tensor(param_out), Tensor(velocity_out), Tensor(master_param_out)"
    },
    "mp_allreduce_sum": {
        "args": "Tensor x, int ring_id = 0",
        "output": "Tensor(out)"
    },
    "multi_dot": {
        "args": "Tensor[] x",
        "output": "Tensor"
    },
    "multiclass_nms3": {
        "args": "Tensor bboxes, Tensor scores, Tensor rois_num, float score_threshold, int nms_top_k, int keep_top_k, float nms_threshold=0.3, bool normalized=true, float nms_eta=1.0, int background_label=0",
        "output": "Tensor(out), Tensor(index), Tensor(nms_rois_num)"
    },
    "multinomial": {
        "args": "Tensor x, Scalar(int) num_samples = 1, bool replacement = false",
        "output": "Tensor(out)"
    },
    "multiplex": {
        "args": "Tensor[] inputs, Tensor index",
        "output": "Tensor"
    },
    "mv": {
        "args": "Tensor x, Tensor vec",
        "output": "Tensor"
    },
    "nadam_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor momentum_decay_pow, Tensor beta2_pow, Tensor mu_product, Tensor moment1, Tensor moment2, Tensor master_param, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1.0e-8f, float momentum_decay = 0.004f, bool multi_precision = false",
        "output": "Tensor(param_out), Tensor(momentum_decay_pow_out), Tensor(beta2_pow_out), Tensor(mu_product_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(master_param_out)"
    },
    "nanmedian": {
        "args": "Tensor x, IntArray axis = {}, bool keepdim = true, str mode=\"avg\"",
        "output": "Tensor(out), Tensor(medians)"
    },
    "nearest_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_format=\"NCHW\", int out_d=0, int out_h=0, int out_w=0, float[] scale={}, str interp_method=\"bilinear\", bool align_corners=true, int align_mode=1",
        "output": "Tensor(output)"
    },
    "nextafter": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "nll_loss": {
        "args": "Tensor input, Tensor label, Tensor weight, int64_t ignore_index = -100, str reduction = \"mean\"",
        "output": "Tensor(out), Tensor(total_weight)"
    },
    "nms": {
        "args": "Tensor x, float threshold = 1.0f",
        "output": "Tensor(out)"
    },
    "nonzero": {
        "args": "Tensor condition",
        "output": "Tensor(out)"
    },
    "norm": {
        "args": "Tensor x, int axis, float epsilon, bool is_test",
        "output": "Tensor(out), Tensor(norm)"
    },
    "npu_identity": {
        "args": "Tensor x, int format = -1",
        "output": "Tensor"
    },
    "numel": {
        "args": "Tensor x",
        "output": "Tensor(size)"
    },
    "one_hot": {
        "args": "Tensor x, Scalar(int) num_classes",
        "output": "Tensor(out)"
    },
    "ones": {
        "args": "IntArray shape, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "ones_like": {
        "args": "Tensor x, DataType dtype=DataType::UNDEFINED, Place place={}",
        "output": "Tensor(out)"
    },
    "overlap_add": {
        "args": "Tensor x, int hop_length, int axis=-1",
        "output": "Tensor"
    },
    "p_norm": {
        "args": "Tensor x,  float porder=2,  int axis=-1,  float epsilon=1.0e-12f,  bool keepdim=false,  bool asvector=false",
        "output": "Tensor(out)"
    },
    "pad": {
        "args": "Tensor x, int[] paddings, Scalar pad_value",
        "output": "Tensor"
    },
    "pad3d": {
        "args": "Tensor x, IntArray paddings, str mode = \"constant\", float pad_value = 0.0, str data_format = \"NCDHW\"",
        "output": "Tensor(out)"
    },
    "partial_allgather": {
        "args": "Tensor x, int nranks, int rank, int ring_id = 0",
        "output": "Tensor(out)"
    },
    "partial_concat": {
        "args": "Tensor[] x, int start_index = 0, int length = -1",
        "output": "Tensor(out)"
    },
    "partial_sum": {
        "args": "Tensor[] x, int start_index = 0, int length = -1",
        "output": "Tensor(out)"
    },
    "pixel_shuffle": {
        "args": "Tensor x, int upscale_factor=1, str data_format=\"NCHW\"",
        "output": "Tensor"
    },
    "pixel_unshuffle": {
        "args": "Tensor x, int downscale_factor=1, str data_format=\"NCHW\"",
        "output": "Tensor"
    },
    "poisson": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "polygamma": {
        "args": "Tensor x, int n",
        "output": "Tensor(out)"
    },
    "pool2d": {
        "args": "Tensor x, IntArray kernel_size, int[] strides, int[] paddings, bool ceil_mode, bool exclusive, str data_format, str pooling_type, bool global_pooling, bool adaptive, str padding_algorithm",
        "output": "Tensor(out)"
    },
    "pool3d": {
        "args": "Tensor x, int[] kernel_size, int[] strides, int[] paddings, bool ceil_mode, bool exclusive, str data_format, str pooling_type, bool global_pooling, bool adaptive, str padding_algorithm",
        "output": "Tensor(out)"
    },
    "pow": {
        "args": "Tensor x, Scalar y=1.0f",
        "output": "Tensor(out)"
    },
    "prelu": {
        "args": "Tensor x, Tensor alpha, str data_format=\"NCHW\", str mode=\"all\"",
        "output": "Tensor(out)"
    },
    "prior_box": {
        "args": "Tensor input, Tensor image, float[] min_sizes, float[] max_sizes = {}, float[] aspect_ratios = {}, float[] variances = {}, bool flip=true, bool clip=true, float step_w=0.0, float step_h=0.0, float offset=0.5, bool min_max_aspect_ratios_order=false",
        "output": "Tensor(out), Tensor(var)"
    },
    "prod": {
        "args": "Tensor x, IntArray axis, bool keepdim, bool reduce_all",
        "output": "Tensor"
    },
    "prune_gate_by_capacity": {
        "args": "Tensor gate_idx, Tensor expert_count, int64_t n_expert=0, int64_t n_worker=0",
        "output": "Tensor(out_gate_idx)"
    },
    "psroi_pool": {
        "args": "Tensor x, Tensor boxes, Tensor boxes_num, int pooled_height=1, int pooled_width=1, int output_channels=1, float spatial_scale=1.0",
        "output": "Tensor"
    },
    "put_along_axis": {
        "args": "Tensor arr, Tensor indices, Tensor values, int axis, str reduce = \"assign\", bool include_self = true",
        "output": "Tensor(out)"
    },
    "pyramid_hash": {
        "args": "Tensor x, Tensor w, Tensor white_list, Tensor black_list, int num_emb = 0, int space_len = 0, int pyramid_layer = 2, int rand_len = 0, float drop_out_percent = 0, int is_training = 0, bool use_filter = true, int white_list_len = 0, int black_list_len = 0, int seed = 0, float lr = 0.0, str distribute_update_vars = \"\"",
        "output": "Tensor (out), Tensor (drop_pos), Tensor (x_temp_out)"
    },
    "qr": {
        "args": "Tensor x, str mode = \"reduced\"",
        "output": "Tensor(q), Tensor(r)"
    },
    "radam_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor beta1_pow, Tensor beta2_pow, Tensor rho, Tensor moment1, Tensor moment2, Tensor master_param, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1.0e-8f, bool multi_precision = false",
        "output": "Tensor(param_out), Tensor(beta1_pow_out), Tensor(beta2_pow_out), Tensor(rho_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(master_param_out)"
    },
    "randint": {
        "args": "int low, int high, IntArray shape, DataType dtype=DataType::INT64, Place place={}",
        "output": "Tensor(out)"
    },
    "random_routing": {
        "args": "Tensor prob, Tensor topk_value, Tensor topk_idx",
        "output": "Tensor(out)"
    },
    "randperm": {
        "args": "int n, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "rank_attention": {
        "args": "Tensor x, Tensor rank_offset, Tensor rank_param, int max_rank = 3, int max_size = 0",
        "output": "Tensor(input_help), Tensor(out), Tensor(ins_rank)"
    },
    "read_file": {
        "args": "str filename = \"\", DataType dtype=DataType::UINT8, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "real": {
        "args": "Tensor x",
        "output": "Tensor (out)"
    },
    "reciprocal": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "reduce": {
        "args": "Tensor x, int ring_id = 0, int root_id = 0, int reduce_type = 0",
        "output": "Tensor(out)"
    },
    "reduce_as": {
        "args": "Tensor x, Tensor target",
        "output": "Tensor(out)"
    },
    "reduce_scatter": {
        "args": "Tensor x, int ring_id = 0, int nranks = 1",
        "output": "Tensor(out)"
    },
    "reindex_graph": {
        "args": "Tensor x, Tensor neighbors, Tensor count, Tensor hashtable_value, Tensor hashtable_index",
        "output": "Tensor(reindex_src), Tensor(reindex_dst), Tensor(out_nodes)"
    },
    "relu": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "relu6": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "renorm": {
        "args": "Tensor x, float p, int axis, float max_norm",
        "output": "Tensor(out)"
    },
    "repeat_interleave": {
        "args": "Tensor x, int repeats, int axis",
        "output": "Tensor(out)"
    },
    "repeat_interleave_with_tensor_index": {
        "args": "Tensor x, Tensor repeats, int axis",
        "output": "Tensor(out)"
    },
    "reshape": {
        "args": "Tensor x, IntArray shape",
        "output": "Tensor(out)"
    },
    "reverse": {
        "args": "Tensor x, IntArray axis",
        "output": "Tensor"
    },
    "rms_norm": {
        "args": "Tensor x, Tensor bias, Tensor residual, Tensor norm_weight, Tensor norm_bias, float epsilon, int begin_norm_axis, float quant_scale, int quant_round_type, float quant_max_bound, float quant_min_bound",
        "output": "Tensor(out), Tensor(residual_out), Tensor(inv_var)"
    },
    "rmsprop_": {
        "args": "Tensor param, Tensor mean_square, Tensor grad, Tensor moment, Tensor learning_rate, Tensor mean_grad, Tensor master_param, float epsilon = 1.0e-10f, float decay = 0.9f, float momentum = 0.0f, bool centered = false, bool multi_precision = false",
        "output": "Tensor(param_out), Tensor(moment_out), Tensor(mean_square_out), Tensor(mean_grad_out), Tensor(master_param_outs)"
    },
    "rnn": {
        "args": "Tensor x, Tensor[] pre_state, Tensor[] weight_list, Tensor sequence_length, Tensor dropout_state_in, float dropout_prob=0.0, bool is_bidirec=false, int input_size=10, int hidden_size=100, int num_layers=1, str mode=\"RNN_TANH\", int seed=0, bool is_test=false",
        "output": "Tensor(out), Tensor(dropout_state_out), Tensor[](state){pre_state.size()}, Tensor(reserve)"
    },
    "roi_align": {
        "args": "Tensor x, Tensor boxes, Tensor boxes_num, int pooled_height=1, int pooled_width=1, float spatial_scale=1.0, int sampling_ratio=-1, bool aligned=false",
        "output": "Tensor"
    },
    "roi_pool": {
        "args": "Tensor x, Tensor boxes, Tensor boxes_num, int pooled_height=1, int pooled_width=1, float spatial_scale=1.0",
        "output": "Tensor(out), Tensor(arg_max)"
    },
    "roll": {
        "args": "Tensor x, IntArray shifts={}, int64_t[] axis={}",
        "output": "Tensor(out)"
    },
    "round": {
        "args": "Tensor x, int decimals = 0 ",
        "output": "Tensor(out)"
    },
    "rprop_": {
        "args": "Tensor param, Tensor grad, Tensor prev, Tensor learning_rate, Tensor master_param, Tensor learning_rate_range, Tensor etas, bool multi_precision=false",
        "output": "Tensor(param_out), Tensor(prev_out), Tensor(learning_rate_out), Tensor(master_param_out)"
    },
    "rrelu": {
        "args": "Tensor x, float lower=1.0f/8, float upper=1.0f/3, bool is_test=false",
        "output": "Tensor(out), Tensor(noise)"
    },
    "rsqrt": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "scale": {
        "args": "Tensor x, Scalar scale=1.0, Scalar bias=0.0, bool bias_after_scale=true",
        "output": "Tensor(out)"
    },
    "scatter": {
        "args": "Tensor x, Tensor index, Tensor updates, bool overwrite=true",
        "output": "Tensor(out)"
    },
    "scatter_nd_add": {
        "args": "Tensor x, Tensor index, Tensor updates",
        "output": "Tensor"
    },
    "searchsorted": {
        "args": "Tensor sorted_sequence, Tensor values, bool out_int32 = false, bool right = false",
        "output": "Tensor(out)"
    },
    "segment_pool": {
        "args": "Tensor x, Tensor segment_ids, str pooltype=\"SUM\"",
        "output": "Tensor(out), Tensor(summed_ids)"
    },
    "selu": {
        "args": "Tensor x, float scale=1.0507009873554804934193349852946, float alpha=1.6732632423543772848170429916717",
        "output": "Tensor"
    },
    "send_u_recv": {
        "args": "Tensor x, Tensor src_index, Tensor dst_index, str reduce_op = \"SUM\", IntArray out_size = {0}",
        "output": "Tensor(out), Tensor(dst_count)"
    },
    "send_ue_recv": {
        "args": "Tensor x, Tensor y, Tensor src_index, Tensor dst_index, str message_op=\"ADD\", str reduce_op=\"SUM\", IntArray out_size={0}",
        "output": "Tensor(out), Tensor(dst_count)"
    },
    "send_uv": {
        "args": "Tensor x, Tensor y, Tensor src_index, Tensor dst_index, str message_op = \"ADD\"",
        "output": "Tensor(out)"
    },
    "sequence_conv": {
        "args": "Tensor x, Tensor padding_data, Tensor filter, int context_length, bool padding_trainable = false, int context_start = 0, int context_stride = 1",
        "output": "Tensor (out)"
    },
    "sequence_mask": {
        "args": "Tensor x, Scalar(int) max_len, DataType out_dtype",
        "output": "Tensor(y)"
    },
    "sequence_pool": {
        "args": "Tensor x, bool is_test=false, str pooltype = \"AVERAGE\", float pad_value = 0.0",
        "output": "Tensor (out), Tensor (max_index)"
    },
    "set": {
        "args": "Tensor x, Tensor source, int64_t[] dims = {}, int64_t[] stride = {}, int64_t offset = 0",
        "output": "Tensor (out)"
    },
    "set_value_with_tensor": {
        "args": "Tensor x, Tensor values, IntArray starts, IntArray ends, IntArray steps, int64_t[] axes, int64_t[] decrease_axes, int64_t[] none_axes",
        "output": "Tensor(out)"
    },
    "sgd_": {
        "args": "Tensor param, Tensor learning_rate, Tensor grad, Tensor master_param, bool multi_precision=false",
        "output": "Tensor(param_out), Tensor(master_param_out)"
    },
    "shape": {
        "args": "Tensor input",
        "output": "Tensor(out)"
    },
    "shape64": {
        "args": "Tensor input",
        "output": "Tensor(out)"
    },
    "shard_index": {
        "args": "Tensor input, int index_num, int nshards, int shard_id, int ignore_value=-1",
        "output": "Tensor(out)"
    },
    "share_data": {
        "args": "Tensor x",
        "output": "Tensor (out)"
    },
    "shuffle_batch": {
        "args": "Tensor x, Tensor seed, int startup_seed=0",
        "output": "Tensor(out), Tensor(shuffle_idx), Tensor(seed_out)"
    },
    "shuffle_channel": {
        "args": "Tensor x, int group = 1",
        "output": "Tensor(out)"
    },
    "sigmoid": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "sigmoid_cross_entropy_with_logits": {
        "args": "Tensor x, Tensor label, Tensor pos_weight, bool normalize=false, int ignore_index=-100",
        "output": "Tensor"
    },
    "sign": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "silu": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "sin": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "sinh": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "slice": {
        "args": "Tensor input, int64_t[] axes, IntArray starts, IntArray ends, int64_t[] infer_flags, int64_t[] decrease_axis",
        "output": "Tensor"
    },
    "slogdet": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "softplus": {
        "args": "Tensor x, float beta = 1.0, float threshold = 20.0f",
        "output": "Tensor"
    },
    "softshrink": {
        "args": "Tensor x, float threshold = 0.5",
        "output": "Tensor"
    },
    "softsign": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "solve": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "sparse_attention": {
        "args": "Tensor q, Tensor k, Tensor v, Tensor offset, Tensor columns, Tensor key_padding_mask, Tensor attn_mask",
        "output": "Tensor (out), Tensor (sparse_dot_sdd), Tensor (softmax)"
    },
    "spectral_norm": {
        "args": "Tensor weight, Tensor u, Tensor v, int dim = 0, int power_iters = 1, float eps = 1e-12f",
        "output": "Tensor"
    },
    "split": {
        "args": "Tensor x, IntArray sections, Scalar(int) axis",
        "output": "Tensor[]{sections.size()}"
    },
    "split_with_num": {
        "args": "Tensor x, int num, Scalar(int) axis",
        "output": "Tensor[]{num}"
    },
    "sqrt": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "square": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "squared_l2_norm": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "squeeze": {
        "args": "Tensor x, IntArray axis={}",
        "output": "Tensor(out)"
    },
    "stack": {
        "args": "Tensor[] x, int axis = 0",
        "output": "Tensor (out)"
    },
    "standard_gamma": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "stanh": {
        "args": "Tensor x, float scale_a=0.67f, float scale_b=1.7159f",
        "output": "Tensor(out)"
    },
    "stft": {
        "args": "Tensor x, Tensor window, int n_fft, int hop_length, bool normalized, bool onesided",
        "output": "Tensor (out)"
    },
    "strided_slice": {
        "args": "Tensor x, int[] axes, IntArray starts, IntArray ends, IntArray strides",
        "output": "Tensor"
    },
    "sum": {
        "args": "Tensor x, IntArray axis={}, DataType dtype=DataType::UNDEFINED, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "svd": {
        "args": "Tensor x, bool full_matrices = false",
        "output": "Tensor(u), Tensor(s), Tensor(vh)"
    },
    "svdvals": {
        "args": "Tensor x",
        "output": "Tensor(s)"
    },
    "swiglu": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "swish": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "sync_batch_norm_": {
        "args": "Tensor x, Tensor mean, Tensor variance, Tensor scale, Tensor bias, bool is_test, float momentum, float epsilon, str data_format, bool use_global_stats, bool trainable_statistics",
        "output": "Tensor(out), Tensor(mean_out), Tensor(variance_out), Tensor(saved_mean), Tensor(saved_variance), Tensor(reserve_space)"
    },
    "sync_calc_stream": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "take_along_axis": {
        "args": "Tensor arr, Tensor indices, int axis",
        "output": "Tensor"
    },
    "tan": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "tanh": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "tanh_shrink": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "tdm_child": {
        "args": "Tensor x, Tensor tree_info, int child_nums, DataType dtype = DataType::INT32",
        "output": "Tensor (child), Tensor (leaf_mask)"
    },
    "tdm_sampler": {
        "args": "Tensor x, Tensor travel, Tensor layer, bool output_positive=true, int[] neg_samples_num_list={}, int[] layer_offset={}, int seed = 0, int dtype=2",
        "output": "Tensor(out), Tensor(labels), Tensor(mask)"
    },
    "temporal_shift": {
        "args": "Tensor x, int seg_num, float shift_ratio = 0.25f, str data_format = \"NCHW\"",
        "output": "Tensor(out)"
    },
    "thresholded_relu": {
        "args": "Tensor x, float threshold = 1.0, float value = 0.0",
        "output": "Tensor(out)"
    },
    "top_p_sampling": {
        "args": "Tensor x, Tensor ps, Tensor threshold, Tensor topp_seed, int seed=-1, int k=0, str mode=\"truncate\"",
        "output": "Tensor (out), Tensor(ids), Tensor(topk_scores), Tensor(topk_ids)"
    },
    "topk": {
        "args": "Tensor x, Scalar(int) k = 1, int axis = -1, bool largest = true, bool sorted = true",
        "output": "Tensor(out), Tensor(indices)"
    },
    "trace": {
        "args": "Tensor x, int offset = 0, int axis1 = 0, int axis2 = 1",
        "output": "Tensor"
    },
    "trans_layout": {
        "args": "Tensor x, int[] perm",
        "output": "Tensor"
    },
    "transpose": {
        "args": "Tensor x, int[] perm",
        "output": "Tensor(out)"
    },
    "triangular_solve": {
        "args": "Tensor x, Tensor y, bool upper=true, bool transpose=false, bool unitriangular=false",
        "output": "Tensor"
    },
    "tril": {
        "args": "Tensor x, int diagonal",
        "output": "Tensor(out)"
    },
    "tril_indices": {
        "args": "int rows, int cols, int offset, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "trilinear_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_format=\"NCHW\", int out_d=0, int out_h=0, int out_w=0, float[] scale={}, str interp_method=\"bilinear\", bool align_corners=true, int align_mode=1",
        "output": "Tensor(output)"
    },
    "triu": {
        "args": "Tensor x, int diagonal",
        "output": "Tensor(out)"
    },
    "triu_indices": {
        "args": "int row, int col, int offset, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "trunc": {
        "args": "Tensor input",
        "output": "Tensor(out)"
    },
    "truncated_gaussian_random": {
        "args": "int[] shape, float mean, float std, int seed, float a, float b, DataType dtype=DataType::FLOAT32, Place place={}",
        "output": "Tensor(out)"
    },
    "unbind": {
        "args": "Tensor input, int axis = 0",
        "output": "Tensor[] {axis<0 ? input.dims()[input.dims().size()+axis]:input.dims()[axis]}"
    },
    "unfold": {
        "args": "Tensor x, int[] kernel_sizes, int[] strides, int[] paddings, int[] dilations",
        "output": "Tensor(out)"
    },
    "uniform": {
        "args": "IntArray shape,  DataType dtype,  Scalar min,  Scalar max,  int seed, Place place={}",
        "output": "Tensor(out)"
    },
    "uniform_inplace": {
        "args": "Tensor x, float min = -1.0, float max = 1.0, int seed = 0, int diag_num = 0, int diag_step = 0, float diag_val = 1.0",
        "output": "Tensor(out)"
    },
    "uniform_random_batch_size_like": {
        "args": "Tensor input, int[] shape, int input_dim_idx = 0, int output_dim_idx = 0, float min=-1.0f, float max=1.0f, int seed=0, int diag_num=0, int diag_step=0, float diag_val=1.0f, DataType dtype=DataType::FLOAT32",
        "output": "Tensor (out)"
    },
    "unique_consecutive": {
        "args": "Tensor x, bool return_inverse = false, bool return_counts = false, int[] axis = {}, DataType dtype = DataType::FLOAT32",
        "output": "Tensor(out), Tensor(index), Tensor(counts)"
    },
    "unpool": {
        "args": "Tensor x, Tensor indices, int[] ksize, int[] strides, int[] padding, IntArray output_size, str data_format",
        "output": "Tensor(out)"
    },
    "unpool3d": {
        "args": "Tensor x, Tensor indices, int[] ksize, int[] strides={1,1,1}, int[] paddings={0,0,0}, int[] output_size={0,0,0}, str data_format=\"NCDHW\"",
        "output": "Tensor(out)"
    },
    "unsqueeze": {
        "args": "Tensor x, IntArray axis = {}",
        "output": "Tensor(out)"
    },
    "unstack": {
        "args": "Tensor x, int axis=0, int num=0",
        "output": "Tensor[](out){num}"
    },
    "update_loss_scaling_": {
        "args": "Tensor[] x, Tensor found_infinite, Tensor prev_loss_scaling, Tensor in_good_steps, Tensor in_bad_steps, int incr_every_n_steps, int decr_every_n_nan_or_inf, float incr_ratio, float decr_ratio, Scalar stop_update=false",
        "output": "Tensor[](out){x.size()}, Tensor(loss_scaling), Tensor(out_good_steps), Tensor(out_bad_steps)"
    },
    "variance": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "view_dtype": {
        "args": "Tensor input, DataType dtype",
        "output": "Tensor(out)"
    },
    "view_slice": {
        "args": "Tensor input, int64_t begin_idx, int64_t end_idx",
        "output": "Tensor"
    },
    "viterbi_decode": {
        "args": "Tensor potentials, Tensor transition_params, Tensor lengths, bool include_bos_eos_tag = true",
        "output": "Tensor(scores), Tensor(path)"
    },
    "warpctc": {
        "args": "Tensor logits, Tensor label, Tensor logits_length, Tensor labels_length, int blank = 0, bool norm_by_times = false",
        "output": "Tensor(loss), Tensor(warpctcgrad)"
    },
    "warprnnt": {
        "args": "Tensor input, Tensor label, Tensor input_lengths, Tensor label_lengths, int blank = 0, float fastemit_lambda = 0.0",
        "output": "Tensor(loss), Tensor(warprnntgrad)"
    },
    "weight_dequantize": {
        "args": "Tensor x, Tensor scale, str algo = \"weight_only_int8\", DataType out_dtype = DataType::FLOAT16, int group_size = -1",
        "output": "Tensor(out)"
    },
    "weight_only_linear": {
        "args": "Tensor x, Tensor weight, Tensor bias, Tensor weight_scale, str weight_dtype, int arch = 80, int group_size = -1",
        "output": "Tensor(out)"
    },
    "weight_quantize": {
        "args": "Tensor x, str algo = \"weight_only_int8\", int arch = 80, int group_size = -1",
        "output": "Tensor(out), Tensor(scale)"
    },
    "weighted_sample_neighbors": {
        "args": "Tensor row, Tensor colptr, Tensor edge_weight, Tensor input_nodes, Tensor eids, int sample_size, bool return_eids",
        "output": "Tensor(out_neighbors), Tensor(out_count), Tensor(out_eids)"
    },
    "where": {
        "args": "Tensor condition, Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "yolo_box": {
        "args": "Tensor x, Tensor img_size, int[] anchors={}, int class_num = 1, float conf_thresh = 0.01, int downsample_ratio = 32, bool clip_bbox = true, float scale_x_y=1.0, bool iou_aware=false, float iou_aware_factor=0.5",
        "output": "Tensor(boxes), Tensor(scores)"
    },
    "yolo_box_head": {
        "args": "Tensor x, int[] anchors, int class_num",
        "output": "Tensor(out)"
    },
    "yolo_box_post": {
        "args": "Tensor boxes0, Tensor boxes1, Tensor boxes2, Tensor image_shape, Tensor image_scale, int[] anchors0, int[] anchors1, int[] anchors2, int class_num, float conf_thresh, int downsample_ratio0, int downsample_ratio1, int downsample_ratio2, bool clip_bbox, float scale_x_y, float nms_threshold",
        "output": "Tensor(out), Tensor(nms_rois_num)"
    },
    "yolo_loss": {
        "args": "Tensor x, Tensor gt_box, Tensor gt_label, Tensor gt_score, int[] anchors={}, int[] anchor_mask={}, int class_num =1 , float ignore_thresh=0.7, int downsample_ratio=32, bool use_label_smooth=true, float scale_x_y=1.0",
        "output": "Tensor(loss), Tensor(objectness_mask), Tensor(gt_match_mask)"
    },
    "zeros": {
        "args": "IntArray shape, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "zeros_like": {
        "args": "Tensor x, DataType dtype=DataType::UNDEFINED, Place place = {}",
        "output": "Tensor(out)"
    },
    "chunk_eval": {
        "args": "Tensor inference, Tensor label, Tensor seq_length, int num_chunk_types, str chunk_scheme = \"IOB\", int[] excluded_chunk_types = {}",
        "output": "Tensor (precision), Tensor (recall), Tensor (f1_score), Tensor (num_infer_chunks), Tensor (num_label_chunks), Tensor (num_correct_chunks)"
    },
    "number_count": {
        "args": "Tensor numbers, int upper_range",
        "output": "Tensor(out)"
    },
    "add": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "add_n": {
        "args": "Tensor[] inputs",
        "output": "Tensor"
    },
    "arange": {
        "args": "Tensor start, Tensor end, Tensor step, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "assign": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "batch_norm": {
        "args": "Tensor x, Tensor mean, Tensor variance, Tensor scale, Tensor bias, bool is_test, float momentum, float epsilon, str data_format, bool use_global_stats, bool trainable_statistics",
        "output": "Tensor(out), Tensor(mean_out), Tensor(variance_out), Tensor(saved_mean), Tensor(saved_variance), Tensor(reserve_space)"
    },
    "c_embedding": {
        "args": "Tensor weight, Tensor x, int64_t start_index=0, int64_t vocab_size=-1",
        "output": "Tensor(out)"
    },
    "distribute_fpn_proposals": {
        "args": "Tensor fpn_rois, Tensor rois_num, int min_level, int max_level, int refer_level, int refer_scale, bool pixel_offset",
        "output": "Tensor[](multi_fpn_rois){max_level - min_level + 1}, Tensor[](multi_level_rois_num){max_level - min_level + 1}, Tensor(restore_index)"
    },
    "divide": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "einsum": {
        "args": "Tensor[] x, str equation",
        "output": "Tensor(out), Tensor[](inner_cache){x.size()}, Tensor[](xshape){x.size()}"
    },
    "elementwise_pow": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "embedding": {
        "args": "Tensor x, Tensor weight, int64_t padding_idx=-1, bool sparse=false",
        "output": "Tensor"
    },
    "embedding_grad_dense": {
        "args": "Tensor x, Tensor weight, Tensor out_grad, int64_t padding_idx=-1, bool sparse=false",
        "output": "Tensor(weight_grad)"
    },
    "equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "floor_divide": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "fused_adam_": {
        "args": "Tensor[] params, Tensor[] grads, Tensor learning_rate, Tensor[] moments1, Tensor[] moments2, Tensor[] moments2_max, Tensor[] beta1_pows, Tensor[] beta2_pows, Tensor[] master_params, Tensor skip_update, Scalar beta1, Scalar beta2, Scalar epsilon, int chunk_size, float weight_decay, bool use_adamw, bool multi_precision, bool use_global_beta_pow, bool amsgrad = false",
        "output": "Tensor[](params_out){params.size()}, Tensor[](moments1_out){params.size()}, Tensor[](moments2_out){params.size()}, Tensor[](moments2_max_out){params.size()}, Tensor[](beta1_pows_out){params.size()}, Tensor[](beta2_pows_out){params.size()}, Tensor[](master_params_out){params.size()}"
    },
    "fused_gemm_epilogue": {
        "args": "Tensor x, Tensor y, Tensor bias, bool trans_x, bool trans_y, str activation",
        "output": "Tensor(out), Tensor(reserve_space)"
    },
    "greater_equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "greater_than": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "hardswish": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "less_equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "less_than": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "matmul": {
        "args": "Tensor x, Tensor y, bool transpose_x = false, bool transpose_y = false",
        "output": "Tensor"
    },
    "maximum": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "min": {
        "args": "Tensor x, IntArray axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "minimum": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "multiply": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "not_equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "remainder": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor (out)"
    },
    "set_value": {
        "args": "Tensor x, IntArray starts, IntArray ends, IntArray steps, int64_t[] axes, int64_t[] decrease_axes, int64_t[] none_axes, int64_t[] shape, Scalar[] values",
        "output": "Tensor(out)"
    },
    "softmax": {
        "args": "Tensor x, int axis",
        "output": "Tensor(out)"
    },
    "subtract": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "sync_comm_stream": {
        "args": "Tensor[] x, int ring_id = 0",
        "output": "Tensor[](out){x.size()}"
    },
    "tensor_unfold": {
        "args": "Tensor input, int64_t axis, int64_t size, int64_t step",
        "output": "Tensor"
    },
    "tile": {
        "args": "Tensor x, IntArray repeat_times = {}",
        "output": "Tensor(out)"
    },
    "unique": {
        "args": "Tensor x, bool return_index, bool return_inverse, bool return_counts, int[] axis, DataType dtype=DataType::INT64",
        "output": "Tensor(out), Tensor(indices), Tensor(inverse), Tensor(counts)"
    },
    "view_shape": {
        "args": "Tensor input, int64_t[] dims = {}",
        "output": "Tensor(out)"
    }
}
