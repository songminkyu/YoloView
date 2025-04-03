from __future__ import annotations
import paddle
__all__ = ['abs', 'abs_', 'abs_grad', 'accuracy', 'accuracy_check', 'acos', 'acos_', 'acos_grad', 'acos_grad_', 'acosh', 'acosh_', 'acosh_grad', 'acosh_grad_', 'adadelta_', 'adagrad_', 'adam_', 'adamax_', 'adamw_', 'add', 'add_', 'add_grad', 'add_grad_', 'add_group_norm_silu', 'add_n', 'add_n_array', 'add_position_encoding', 'add_position_encoding_grad', 'addmm', 'addmm_', 'addmm_grad', 'affine_channel', 'affine_channel_', 'affine_channel_grad', 'affine_channel_grad_', 'affine_grid', 'affine_grid_grad', 'all', 'all_gather', 'all_reduce', 'all_reduce_', 'all_to_all', 'allclose', 'amax', 'amax_grad', 'amin', 'amin_grad', 'anchor_generator', 'angle', 'angle_grad', 'any', 'apply_per_channel_scale', 'arange', 'argmax', 'argmin', 'argsort', 'argsort_grad', 'array_length', 'array_pop', 'array_read', 'array_to_tensor', 'array_write_', 'as_complex', 'as_real', 'as_strided', 'as_strided_grad', 'asgd_', 'asin', 'asin_', 'asin_grad', 'asin_grad_', 'asinh', 'asinh_', 'asinh_grad', 'asinh_grad_', 'assign', 'assign_', 'assign_out_', 'assign_out__grad', 'assign_out__grad_', 'assign_pos', 'assign_value', 'assign_value_', 'atan', 'atan2', 'atan2_grad', 'atan_', 'atan_grad', 'atan_grad_', 'atanh', 'atanh_', 'atanh_grad', 'atanh_grad_', 'attention_lstm', 'auc', 'average_accumulates_', 'baddbmm', 'baddbmm_', 'baddbmm_grad', 'batch_norm', 'batch_norm_', 'batch_norm_grad', 'bce_loss', 'bce_loss_', 'bce_loss_grad', 'bce_loss_grad_', 'beam_search', 'beam_search_decode', 'bernoulli', 'bicubic_interp', 'bicubic_interp_grad', 'bilinear', 'bilinear_grad', 'bilinear_interp', 'bilinear_interp_grad', 'bincount', 'binomial', 'bipartite_match', 'bitwise_and', 'bitwise_and_', 'bitwise_left_shift', 'bitwise_left_shift_', 'bitwise_not', 'bitwise_not_', 'bitwise_or', 'bitwise_or_', 'bitwise_right_shift', 'bitwise_right_shift_', 'bitwise_xor', 'bitwise_xor_', 'blha_get_max_len', 'block_multihead_attention_', 'block_multihead_attention_xpu_', 'bmm', 'bmm_grad', 'box_clip', 'box_coder', 'broadcast', 'broadcast_', 'broadcast_tensors', 'broadcast_tensors_grad', 'builtin_combine', 'builtin_split', 'c_allreduce_sum', 'c_allreduce_sum_', 'c_concat', 'c_embedding', 'c_embedding_grad', 'c_identity', 'c_identity_', 'c_softmax_with_cross_entropy', 'c_softmax_with_cross_entropy_grad', 'c_softmax_with_multi_label_cross_entropy', 'c_softmax_with_multi_label_cross_entropy_grad', 'calc_reduced_attn_scores', 'cast', 'cast_', 'ceil', 'ceil_', 'ceil_grad', 'ceil_grad_', 'celu', 'celu_grad', 'celu_grad_', 'channel_shuffle', 'channel_shuffle_grad', 'check_finite_and_unscale_', 'check_numerics', 'cholesky', 'cholesky_grad', 'cholesky_solve', 'cholesky_solve_grad', 'chunk_eval', 'class_center_sample', 'clip', 'clip_', 'clip_by_norm', 'clip_grad', 'clip_grad_', 'coalesce_tensor', 'coalesce_tensor_', 'collect_fpn_proposals', 'complex', 'complex_grad', 'concat', 'concat_grad', 'conj', 'conv2d', 'conv2d_transpose', 'conv2d_transpose_bias', 'conv2d_transpose_grad', 'conv3d', 'conv3d_grad', 'conv3d_transpose', 'conv3d_transpose_grad', 'copysign', 'copysign_', 'copysign_grad', 'copysign_grad_', 'correlation', 'correlation_grad', 'cos', 'cos_', 'cos_grad', 'cos_grad_', 'cosh', 'cosh_', 'cosh_grad', 'cosh_grad_', 'create_array', 'create_array_like', 'crf_decoding', 'crop', 'crop_grad', 'cross', 'cross_entropy_with_softmax', 'cross_entropy_with_softmax_', 'cross_entropy_with_softmax_grad', 'cross_entropy_with_softmax_grad_', 'cross_grad', 'ctc_align', 'cudnn_lstm', 'cudnn_lstm_grad', 'cummax', 'cummax_grad', 'cummin', 'cummin_grad', 'cumprod', 'cumprod_', 'cumprod_grad', 'cumsum', 'cumsum_', 'cumsum_grad', 'cvm', 'cvm_grad', 'data', 'decode_jpeg', 'deformable_conv', 'deformable_conv_grad', 'depend', 'depthwise_conv2d', 'depthwise_conv2d_grad', 'depthwise_conv2d_transpose', 'depthwise_conv2d_transpose_grad', 'dequantize_abs_max', 'dequantize_linear', 'dequantize_linear_', 'dequantize_log', 'det', 'det_grad', 'dgc_clip_by_norm', 'diag', 'diag_embed', 'diag_grad', 'diagonal', 'diagonal_grad', 'digamma', 'digamma_', 'digamma_grad', 'dirichlet', 'disable_check_model_nan_inf', 'disable_check_model_nan_inf_grad', 'dist', 'dist_grad', 'distribute_fpn_proposals', 'distributed_fused_lamb_init', 'distributed_fused_lamb_init_', 'divide', 'divide_', 'divide_grad', 'dot', 'dot_grad', 'dropout', 'dropout_grad', 'edit_distance', 'eig', 'eig_grad', 'eigh', 'eigh_grad', 'eigvals', 'eigvalsh', 'eigvalsh_grad', 'einsum', 'einsum_grad', 'elementwise_pow', 'elementwise_pow_grad', 'elu', 'elu_', 'elu_grad', 'elu_grad_', 'embedding', 'embedding_grad', 'embedding_with_scaled_gradient', 'embedding_with_scaled_gradient_grad', 'empty', 'empty_like', 'enable_check_model_nan_inf', 'enable_check_model_nan_inf_grad', 'equal', 'equal_', 'equal_all', 'erf', 'erf_', 'erf_grad', 'erfinv', 'erfinv_', 'erfinv_grad', 'exp', 'exp_', 'exp_grad', 'exp_grad_', 'expand', 'expand_as', 'expand_as_grad', 'expand_grad', 'expm1', 'expm1_', 'expm1_grad', 'expm1_grad_', 'exponential_', 'eye', 'fake_channel_wise_dequantize_max_abs', 'fake_channel_wise_quantize_abs_max', 'fake_channel_wise_quantize_dequantize_abs_max', 'fake_channel_wise_quantize_dequantize_abs_max_grad', 'fake_dequantize_max_abs', 'fake_quantize_abs_max', 'fake_quantize_dequantize_abs_max', 'fake_quantize_dequantize_abs_max_grad', 'fake_quantize_dequantize_moving_average_abs_max', 'fake_quantize_dequantize_moving_average_abs_max_', 'fake_quantize_dequantize_moving_average_abs_max_grad', 'fake_quantize_moving_average_abs_max', 'fake_quantize_moving_average_abs_max_', 'fake_quantize_range_abs_max', 'fake_quantize_range_abs_max_', 'fc', 'fetch', 'fft_c2c', 'fft_c2c_grad', 'fft_c2r', 'fft_c2r_grad', 'fft_r2c', 'fft_r2c_grad', 'fill', 'fill_', 'fill_diagonal', 'fill_diagonal_', 'fill_diagonal_grad', 'fill_diagonal_tensor', 'fill_diagonal_tensor_', 'fill_diagonal_tensor_grad', 'fill_diagonal_tensor_grad_', 'fill_grad', 'fill_grad_', 'flash_attn', 'flash_attn_grad', 'flash_attn_qkvpacked', 'flash_attn_qkvpacked_grad', 'flash_attn_unpadded', 'flash_attn_unpadded_grad', 'flash_attn_varlen_qkvpacked', 'flash_attn_varlen_qkvpacked_grad', 'flashmask_attention', 'flashmask_attention_grad', 'flatten', 'flatten_', 'flatten_grad', 'flatten_grad_', 'flip', 'floor', 'floor_', 'floor_divide', 'floor_divide_', 'floor_grad', 'floor_grad_', 'fmax', 'fmax_grad', 'fmin', 'fmin_grad', 'fold', 'fold_grad', 'fp8_fp8_half_gemm_fused', 'fractional_max_pool2d', 'fractional_max_pool2d_grad', 'fractional_max_pool3d', 'fractional_max_pool3d_grad', 'frame', 'frame_grad', 'frobenius_norm', 'frobenius_norm_grad', 'full', 'full_', 'full_batch_size_like', 'full_int_array', 'full_like', 'full_with_tensor', 'fused_attention', 'fused_attention_grad', 'fused_batch_norm_act', 'fused_batch_norm_act_grad', 'fused_bias_act', 'fused_bias_dropout_residual_layer_norm', 'fused_bias_dropout_residual_layer_norm_grad', 'fused_bias_residual_layernorm', 'fused_bn_add_activation', 'fused_bn_add_activation_grad', 'fused_conv2d_add_act', 'fused_dropout_add', 'fused_dropout_add_grad', 'fused_embedding_eltwise_layernorm', 'fused_fc_elementwise_layernorm', 'fused_feedforward', 'fused_feedforward_grad', 'fused_gemm_epilogue', 'fused_linear_param_grad_add', 'fused_moe', 'fused_multi_transformer_', 'fused_rotary_position_embedding', 'fused_rotary_position_embedding_grad', 'fused_softmax_mask', 'fused_softmax_mask_grad', 'fused_softmax_mask_upper_triangle', 'fused_softmax_mask_upper_triangle_grad', 'fusion_gru', 'fusion_repeated_fc_relu', 'fusion_seqconv_eltadd_relu', 'fusion_seqpool_concat', 'fusion_squared_mat_sub', 'fusion_transpose_flatten_concat', 'gammaincc', 'gammaincc_', 'gammaincc_grad', 'gammaln', 'gammaln_', 'gammaln_grad', 'gather', 'gather_grad', 'gather_nd', 'gather_nd_grad', 'gather_tree', 'gaussian', 'gaussian_inplace', 'gaussian_inplace_', 'gaussian_inplace_grad', 'gaussian_inplace_grad_', 'gelu', 'gelu_grad', 'generate_proposals', 'get_tensor_from_selected_rows', 'graph_khop_sampler', 'graph_sample_neighbors', 'greater_equal', 'greater_equal_', 'greater_than', 'greater_than_', 'grid_sample', 'grid_sample_grad', 'group_norm', 'group_norm_grad', 'group_norm_grad_', 'gru', 'gru_grad', 'gru_unit', 'gru_unit_grad', 'gumbel_softmax', 'gumbel_softmax_grad', 'hardshrink', 'hardshrink_grad', 'hardshrink_grad_', 'hardsigmoid', 'hardsigmoid_grad', 'hardsigmoid_grad_', 'hardswish', 'hardswish_grad', 'hardswish_grad_', 'hardtanh', 'hardtanh_', 'hardtanh_grad', 'hardtanh_grad_', 'hash', 'heaviside', 'heaviside_grad', 'hinge_loss', 'hinge_loss_grad', 'histogram', 'hsigmoid_loss', 'hsigmoid_loss_grad', 'huber_loss', 'huber_loss_grad', 'i0', 'i0_', 'i0_grad', 'i0e', 'i0e_grad', 'i1', 'i1_grad', 'i1e', 'i1e_grad', 'identity_loss', 'identity_loss_', 'identity_loss_grad', 'identity_loss_grad_', 'im2sequence', 'im2sequence_grad', 'imag', 'imag_grad', 'increment', 'increment_', 'index_add', 'index_add_', 'index_add_grad', 'index_add_grad_', 'index_put', 'index_put_', 'index_put_grad', 'index_sample', 'index_sample_grad', 'index_select', 'index_select_grad', 'index_select_strided', 'index_select_strided_grad', 'instance_norm', 'instance_norm_grad', 'inverse', 'inverse_grad', 'is_empty', 'isclose', 'isfinite', 'isinf', 'isnan', 'kldiv_loss', 'kldiv_loss_grad', 'kron', 'kron_grad', 'kthvalue', 'kthvalue_grad', 'l1_norm', 'l1_norm_', 'l1_norm_grad', 'label_smooth', 'label_smooth_grad', 'lamb_', 'lars_momentum_', 'layer_norm', 'layer_norm_grad', 'leaky_relu', 'leaky_relu_', 'leaky_relu_grad', 'leaky_relu_grad_', 'lerp', 'lerp_', 'lerp_grad', 'less_equal', 'less_equal_', 'less_than', 'less_than_', 'lgamma', 'lgamma_', 'lgamma_grad', 'limit_by_capacity', 'linear_interp', 'linear_interp_grad', 'linspace', 'llm_int8_linear', 'lod_reset_grad_', 'log', 'log10', 'log10_', 'log10_grad', 'log10_grad_', 'log1p', 'log1p_', 'log1p_grad', 'log1p_grad_', 'log2', 'log2_', 'log2_grad', 'log2_grad_', 'log_', 'log_grad', 'log_grad_', 'log_loss', 'log_loss_grad', 'log_softmax', 'log_softmax_grad', 'logcumsumexp', 'logcumsumexp_grad', 'logical_and', 'logical_and_', 'logical_not', 'logical_not_', 'logical_or', 'logical_or_', 'logical_xor', 'logical_xor_', 'logit', 'logit_', 'logit_grad', 'logsigmoid', 'logsigmoid_grad', 'logsigmoid_grad_', 'logspace', 'logsumexp', 'logsumexp_grad', 'lookup_table_dequant', 'lp_pool2d', 'lp_pool2d_grad', 'lstm', 'lstm_grad', 'lstsq', 'lu', 'lu_', 'lu_grad', 'lu_grad_', 'lu_solve', 'lu_solve_grad', 'lu_unpack', 'lu_unpack_grad', 'margin_cross_entropy', 'margin_cross_entropy_grad', 'margin_cross_entropy_grad_', 'masked_multihead_attention_', 'masked_select', 'masked_select_grad', 'matmul', 'matmul_grad', 'matmul_with_flatten', 'matmul_with_flatten_grad', 'matrix_nms', 'matrix_power', 'matrix_power_grad', 'matrix_rank', 'matrix_rank_atol_rtol', 'matrix_rank_tol', 'max', 'max_grad', 'max_pool2d_with_index', 'max_pool2d_with_index_grad', 'max_pool3d_with_index', 'max_pool3d_with_index_grad', 'maximum', 'maximum_grad', 'maxout', 'maxout_grad', 'mean', 'mean_all', 'mean_all_grad', 'mean_grad', 'memcpy', 'memcpy_d2h', 'memcpy_h2d', 'memory_efficient_attention', 'memory_efficient_attention_grad', 'merge_selected_rows', 'merged_adam_', 'merged_momentum_', 'meshgrid', 'meshgrid_grad', 'min', 'min_grad', 'minimum', 'minimum_grad', 'mish', 'mish_grad', 'mish_grad_', 'mode', 'mode_grad', 'moe_dispatch', 'moe_ffn', 'moe_reduce', 'momentum_', 'moving_average_abs_max_scale', 'moving_average_abs_max_scale_', 'mp_allreduce_sum', 'mp_allreduce_sum_', 'multi_dot', 'multi_dot_grad', 'multiclass_nms3', 'multihead_matmul', 'multinomial', 'multiplex', 'multiplex_grad', 'multiply', 'multiply_', 'mv', 'mv_grad', 'nadam_', 'nanmedian', 'nanmedian_grad', 'nearest_interp', 'nearest_interp_grad', 'nextafter', 'nll_loss', 'nll_loss_grad', 'nms', 'nonzero', 'nop', 'nop_', 'norm', 'norm_grad', 'not_equal', 'not_equal_', 'npu_identity', 'number_count', 'numel', 'one_hot', 'onednn_to_paddle_layout', 'overlap_add', 'overlap_add_grad', 'p_norm', 'p_norm_grad', 'pad', 'pad3d', 'pad3d_grad', 'pad_grad', 'parameter', 'pixel_shuffle', 'pixel_shuffle_grad', 'pixel_unshuffle', 'pixel_unshuffle_grad', 'poisson', 'poisson_grad', 'polygamma', 'polygamma_', 'polygamma_grad', 'pool2d', 'pool2d_grad', 'pool3d', 'pool3d_grad', 'pow', 'pow_', 'pow_grad', 'pow_grad_', 'prelu', 'prelu_grad', 'print', 'prior_box', 'prod', 'prod_grad', 'prune_gate_by_capacity', 'psroi_pool', 'psroi_pool_grad', 'put_along_axis', 'put_along_axis_', 'put_along_axis_grad', 'pyramid_hash', 'pyramid_hash_grad', 'qkv_unpack_mha', 'qr', 'qr_grad', 'quantize_linear', 'quantize_linear_', 'radam_', 'randint', 'random_routing_', 'randperm', 'rank_attention', 'rank_attention_grad', 'read_file', 'real', 'real_grad', 'reciprocal', 'reciprocal_', 'reciprocal_grad', 'reciprocal_grad_', 'recv_v2', 'reduce', 'reduce_', 'reduce_as', 'reduce_as_grad', 'reduce_scatter', 'reindex_graph', 'relu', 'relu6', 'relu6_grad', 'relu6_grad_', 'relu_', 'relu_grad', 'relu_grad_', 'remainder', 'remainder_', 'remainder_grad', 'renorm', 'renorm_', 'renorm_grad', 'repeat_interleave', 'repeat_interleave_grad', 'repeat_interleave_with_tensor_index', 'repeat_interleave_with_tensor_index_grad', 'reshape', 'reshape_', 'reshape_grad', 'reshape_grad_', 'resnet_basic_block', 'resnet_basic_block_grad', 'resnet_unit', 'resnet_unit_grad', 'reverse', 'rms_norm', 'rms_norm_grad', 'rmsprop_', 'rnn', 'rnn_grad', 'roi_align', 'roi_align_grad', 'roi_pool', 'roi_pool_grad', 'roll', 'roll_grad', 'round', 'round_', 'round_grad', 'round_grad_', 'rprop_', 'rrelu', 'rrelu_grad', 'rsqrt', 'rsqrt_', 'rsqrt_grad', 'rsqrt_grad_', 'scale', 'scale_', 'scatter', 'scatter_', 'scatter_grad', 'scatter_nd_add', 'scatter_nd_add_grad', 'searchsorted', 'segment_pool', 'segment_pool_grad', 'self_dp_attention', 'selu', 'selu_grad', 'send_u_recv', 'send_u_recv_grad', 'send_ue_recv', 'send_ue_recv_grad', 'send_uv', 'send_uv_grad', 'send_v2', 'sequence_conv', 'sequence_conv_grad', 'sequence_expand', 'sequence_expand_grad', 'sequence_mask', 'sequence_pool', 'sequence_pool_grad', 'sequence_softmax', 'sequence_softmax_grad', 'set', 'set_', 'set_parameter', 'set_persistable_value', 'set_value', 'set_value_', 'set_value_grad', 'set_value_with_tensor', 'set_value_with_tensor_', 'set_value_with_tensor_grad', 'sgd_', 'shape', 'shape64', 'shard_index', 'share_data', 'share_data_', 'share_var', 'shuffle_batch', 'shuffle_batch_grad', 'shuffle_channel', 'shuffle_channel_grad', 'sigmoid', 'sigmoid_', 'sigmoid_cross_entropy_with_logits', 'sigmoid_cross_entropy_with_logits_', 'sigmoid_cross_entropy_with_logits_grad', 'sigmoid_cross_entropy_with_logits_grad_', 'sigmoid_grad', 'sigmoid_grad_', 'sign', 'silu', 'silu_grad', 'silu_grad_', 'sin', 'sin_', 'sin_grad', 'sin_grad_', 'sinh', 'sinh_', 'sinh_grad', 'sinh_grad_', 'skip_layernorm', 'slice', 'slice_array', 'slice_array_dense', 'slice_grad', 'slogdet', 'slogdet_grad', 'softmax', 'softmax_', 'softmax_grad', 'softplus', 'softplus_grad', 'softplus_grad_', 'softshrink', 'softshrink_grad', 'softshrink_grad_', 'softsign', 'softsign_grad', 'softsign_grad_', 'solve', 'solve_grad', 'sparse_abs', 'sparse_abs_grad', 'sparse_acos', 'sparse_acos_grad', 'sparse_acosh', 'sparse_acosh_grad', 'sparse_add', 'sparse_add_grad', 'sparse_addmm', 'sparse_addmm_grad', 'sparse_asin', 'sparse_asin_grad', 'sparse_asinh', 'sparse_asinh_grad', 'sparse_atan', 'sparse_atan_grad', 'sparse_atanh', 'sparse_atanh_grad', 'sparse_attention', 'sparse_attention_grad', 'sparse_batch_norm_', 'sparse_batch_norm_grad', 'sparse_cast', 'sparse_cast_grad', 'sparse_coalesce', 'sparse_conv3d', 'sparse_conv3d_grad', 'sparse_conv3d_implicit_gemm', 'sparse_divide', 'sparse_divide_grad', 'sparse_divide_scalar', 'sparse_expm1', 'sparse_expm1_grad', 'sparse_full_like', 'sparse_fused_attention', 'sparse_fused_attention_grad', 'sparse_indices', 'sparse_isnan', 'sparse_leaky_relu', 'sparse_leaky_relu_grad', 'sparse_log1p', 'sparse_log1p_grad', 'sparse_mask_as', 'sparse_mask_as_grad', 'sparse_masked_matmul', 'sparse_masked_matmul_grad', 'sparse_matmul', 'sparse_matmul_grad', 'sparse_maxpool', 'sparse_maxpool_grad', 'sparse_multiply', 'sparse_mv', 'sparse_mv_grad', 'sparse_pow', 'sparse_pow_grad', 'sparse_relu', 'sparse_relu6', 'sparse_relu6_grad', 'sparse_relu_grad', 'sparse_reshape', 'sparse_reshape_grad', 'sparse_scale', 'sparse_sin', 'sparse_sin_grad', 'sparse_sinh', 'sparse_sinh_grad', 'sparse_slice', 'sparse_slice_grad', 'sparse_softmax', 'sparse_softmax_grad', 'sparse_sparse_coo_tensor', 'sparse_sparse_coo_tensor_grad', 'sparse_sqrt', 'sparse_sqrt_grad', 'sparse_square', 'sparse_square_grad', 'sparse_subtract', 'sparse_subtract_grad', 'sparse_sum', 'sparse_sum_grad', 'sparse_sync_batch_norm_', 'sparse_sync_batch_norm_grad', 'sparse_tan', 'sparse_tan_grad', 'sparse_tanh', 'sparse_tanh_grad', 'sparse_to_dense', 'sparse_to_dense_grad', 'sparse_to_sparse_coo', 'sparse_to_sparse_coo_grad', 'sparse_to_sparse_csr', 'sparse_transpose', 'sparse_transpose_grad', 'sparse_values', 'sparse_values_grad', 'spectral_norm', 'spectral_norm_grad', 'split', 'split_grad', 'split_with_num', 'sqrt', 'sqrt_', 'sqrt_grad', 'sqrt_grad_', 'square', 'square_', 'square_grad', 'square_grad_', 'squared_l2_norm', 'squared_l2_norm_grad', 'squeeze', 'squeeze_', 'squeeze_excitation_block', 'squeeze_grad', 'squeeze_grad_', 'stack', 'stack_grad', 'standard_gamma', 'stanh', 'stanh_grad', 'stft', 'stft_grad', 'strided_slice', 'strided_slice_grad', 'subtract', 'subtract_', 'subtract_grad', 'subtract_grad_', 'sum', 'sum_grad', 'svd', 'svd_grad', 'svdvals', 'svdvals_grad', 'swiglu', 'swiglu_grad', 'swish', 'swish_grad', 'swish_grad_', 'sync_batch_norm_', 'sync_batch_norm_grad', 'sync_calc_stream', 'sync_calc_stream_', 'take_along_axis', 'take_along_axis_grad', 'tan', 'tan_', 'tan_grad', 'tan_grad_', 'tanh', 'tanh_', 'tanh_grad', 'tanh_grad_', 'tanh_shrink', 'tanh_shrink_grad', 'tanh_shrink_grad_', 'tdm_child', 'tdm_sampler', 'temporal_shift', 'temporal_shift_grad', 'tensorrt_engine', 'thresholded_relu', 'thresholded_relu_', 'thresholded_relu_grad', 'thresholded_relu_grad_', 'tile', 'tile_grad', 'top_p_sampling', 'topk', 'topk_grad', 'trace', 'trace_grad', 'trans_layout', 'trans_layout_grad', 'transpose', 'transpose_', 'transpose_grad', 'triangular_solve', 'triangular_solve_grad', 'tril', 'tril_', 'tril_grad', 'tril_indices', 'trilinear_interp', 'trilinear_interp_grad', 'triu', 'triu_', 'triu_grad', 'triu_indices', 'trunc', 'trunc_', 'trunc_grad', 'truncated_gaussian_random', 'unbind', 'unfold', 'unfold_grad', 'uniform', 'uniform_inplace', 'uniform_inplace_', 'uniform_inplace_grad', 'uniform_inplace_grad_', 'uniform_random_batch_size_like', 'unique', 'unique_consecutive', 'unpool', 'unpool3d', 'unpool3d_grad', 'unpool_grad', 'unsqueeze', 'unsqueeze_', 'unsqueeze_grad', 'unsqueeze_grad_', 'unstack', 'unstack_grad', 'update_loss_scaling_', 'update_parameter', 'variable_length_memory_efficient_attention', 'variance', 'view_dtype', 'view_dtype_grad', 'view_slice', 'viterbi_decode', 'warpctc', 'warpctc_grad', 'warprnnt', 'warprnnt_grad', 'weight_dequantize', 'weight_only_linear', 'weight_only_linear_grad', 'weight_quantize', 'weighted_sample_neighbors', 'where', 'where_', 'where_grad', 'yolo_box', 'yolo_box_head', 'yolo_box_post', 'yolo_loss', 'yolo_loss_grad']
def _run_custom_op(*args, **kwargs):
    """
    C++ interface function for run_custom_op.
    """
def abs(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for abs.
    """
def abs_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for abs_.
    """
def abs_grad(*args, **kwargs):
    """
    C++ interface function for abs_grad.
    """
def accuracy(x: paddle.Tensor, indices: paddle.Tensor, label: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for accuracy.
    """
def accuracy_check(x: paddle.Tensor, y: paddle.Tensor, fn_name: str, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> paddle.Tensor:
    """
    C++ interface function for accuracy_check.
    """
def acos(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for acos.
    """
def acos_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for acos_.
    """
def acos_grad(*args, **kwargs):
    """
    C++ interface function for acos_grad.
    """
def acos_grad_(*args, **kwargs):
    """
    C++ interface function for acos_grad_.
    """
def acosh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for acosh.
    """
def acosh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for acosh_.
    """
def acosh_grad(*args, **kwargs):
    """
    C++ interface function for acosh_grad.
    """
def acosh_grad_(*args, **kwargs):
    """
    C++ interface function for acosh_grad_.
    """
def adadelta_(param: paddle.Tensor, grad: paddle.Tensor, avg_squared_grad: paddle.Tensor, avg_squared_update: paddle.Tensor, learning_rate: paddle.Tensor, master_param: paddle.Tensor, rho: float = 0.95, epsilon: float = 1.0e-6, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for adadelta_.
    """
def adagrad_(param: paddle.Tensor, grad: paddle.Tensor, moment: paddle.Tensor, learning_rate: paddle.Tensor, master_param: paddle.Tensor, epsilon: float = 1.0e-6, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for adagrad_.
    """
def adam_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, moment1: paddle.Tensor, moment2: paddle.Tensor, moment2_max: paddle.Tensor, beta1_pow: paddle.Tensor, beta2_pow: paddle.Tensor, master_param: paddle.Tensor, skip_update: paddle.Tensor, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, lazy_mode: bool = False, min_row_size_to_use_multithread: int = 1000, multi_precision: bool = False, use_global_beta_pow: bool = False, amsgrad: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for adam_.
    """
def adamax_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, moment: paddle.Tensor, inf_norm: paddle.Tensor, beta1_pow: paddle.Tensor, master_param: paddle.Tensor, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for adamax_.
    """
def adamw_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, moment1: paddle.Tensor, moment2: paddle.Tensor, moment2_max: paddle.Tensor, beta1_pow: paddle.Tensor, beta2_pow: paddle.Tensor, master_param: paddle.Tensor, skip_update: paddle.Tensor, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, lr_ratio: float = 1.0, coeff: float = 0.01, with_decay: bool = False, lazy_mode: bool = False, min_row_size_to_use_multithread: int = 1000, multi_precision: bool = False, use_global_beta_pow: bool = False, amsgrad: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for adamw_.
    """
def add(*args, **kwargs):
    """
    C++ interface function for add.
    """
def add_(*args, **kwargs):
    """
    C++ interface function for add_.
    """
def add_grad(*args, **kwargs):
    """
    C++ interface function for add_grad.
    """
def add_grad_(*args, **kwargs):
    """
    C++ interface function for add_grad_.
    """
def add_group_norm_silu(*args, **kwargs):
    """
    C++ interface function for add_group_norm_silu.
    """
def add_n(*args, **kwargs):
    """
    C++ interface function for add_n.
    """
def add_n_array(*args, **kwargs):
    """
    C++ interface function for add_n_array.
    """
def add_position_encoding(x: paddle.Tensor, alpha: float = 1.0, beta: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for add_position_encoding.
    """
def add_position_encoding_grad(*args, **kwargs):
    """
    C++ interface function for add_position_encoding_grad.
    """
def addmm(input: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor, beta: float = 1.0, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for addmm.
    """
def addmm_(input: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor, beta: float = 1.0, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for addmm_.
    """
def addmm_grad(*args, **kwargs):
    """
    C++ interface function for addmm_grad.
    """
def affine_channel(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, data_layout: str = "AnyLayout") -> paddle.Tensor:
    """
    C++ interface function for affine_channel.
    """
def affine_channel_(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, data_layout: str = "AnyLayout") -> paddle.Tensor:
    """
    C++ interface function for affine_channel_.
    """
def affine_channel_grad(*args, **kwargs):
    """
    C++ interface function for affine_channel_grad.
    """
def affine_channel_grad_(*args, **kwargs):
    """
    C++ interface function for affine_channel_grad_.
    """
def affine_grid(input: paddle.Tensor, output_shape: list[int] = [], align_corners: bool = True) -> paddle.Tensor:
    """
    C++ interface function for affine_grid.
    """
def affine_grid_grad(*args, **kwargs):
    """
    C++ interface function for affine_grid_grad.
    """
def all(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for all.
    """
def all_gather(x: paddle.Tensor, ring_id: int = 0, nranks: int = 0) -> paddle.Tensor:
    """
    C++ interface function for all_gather.
    """
def all_reduce(x: paddle.Tensor, ring_id: int = 0, reduce_type: int = 0) -> paddle.Tensor:
    """
    C++ interface function for all_reduce.
    """
def all_reduce_(x: paddle.Tensor, ring_id: int = 0, reduce_type: int = 0) -> paddle.Tensor:
    """
    C++ interface function for all_reduce_.
    """
def all_to_all(x: paddle.Tensor, ring_id: int = 0) -> paddle.Tensor:
    """
    C++ interface function for all_to_all.
    """
def allclose(x: paddle.Tensor, y: paddle.Tensor, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> paddle.Tensor:
    """
    C++ interface function for allclose.
    """
def amax(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for amax.
    """
def amax_grad(*args, **kwargs):
    """
    C++ interface function for amax_grad.
    """
def amin(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for amin.
    """
def amin_grad(*args, **kwargs):
    """
    C++ interface function for amin_grad.
    """
def anchor_generator(*args, **kwargs):
    """
    C++ interface function for anchor_generator.
    """
def angle(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for angle.
    """
def angle_grad(*args, **kwargs):
    """
    C++ interface function for angle_grad.
    """
def any(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for any.
    """
def apply_per_channel_scale(x: paddle.Tensor, scales: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for apply_per_channel_scale.
    """
def arange(*args, **kwargs):
    """
    C++ interface function for arange.
    """
def argmax(x: paddle.Tensor, axis: int, keepdims: bool = False, flatten: bool = False, dtype: paddle._typing.DTypeLike = "DataType::INT64") -> paddle.Tensor:
    """
    C++ interface function for argmax.
    """
def argmin(x: paddle.Tensor, axis: int, keepdims: bool = False, flatten: bool = False, dtype: paddle._typing.DTypeLike = "DataType::INT64") -> paddle.Tensor:
    """
    C++ interface function for argmin.
    """
def argsort(x: paddle.Tensor, axis: int = -1, descending: bool = False, stable: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for argsort.
    """
def argsort_grad(*args, **kwargs):
    """
    C++ interface function for argsort_grad.
    """
def array_length(*args, **kwargs):
    """
    C++ interface function for array_length.
    """
def array_pop(*args, **kwargs):
    """
    C++ interface function for array_pop.
    """
def array_read(*args, **kwargs):
    """
    C++ interface function for array_read.
    """
def array_to_tensor(*args, **kwargs):
    """
    C++ interface function for array_to_tensor.
    """
def array_write_(*args, **kwargs):
    """
    C++ interface function for array_write_.
    """
def as_complex(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for as_complex.
    """
def as_real(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for as_real.
    """
def as_strided(input: paddle.Tensor, dims: list[int] = [], stride: list[int] = [], offset: int = 0) -> paddle.Tensor:
    """
    C++ interface function for as_strided.
    """
def as_strided_grad(*args, **kwargs):
    """
    C++ interface function for as_strided_grad.
    """
def asgd_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, d: paddle.Tensor, y: paddle.Tensor, n: paddle.Tensor, master_param: paddle.Tensor, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for asgd_.
    """
def asin(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for asin.
    """
def asin_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for asin_.
    """
def asin_grad(*args, **kwargs):
    """
    C++ interface function for asin_grad.
    """
def asin_grad_(*args, **kwargs):
    """
    C++ interface function for asin_grad_.
    """
def asinh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for asinh.
    """
def asinh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for asinh_.
    """
def asinh_grad(*args, **kwargs):
    """
    C++ interface function for asinh_grad.
    """
def asinh_grad_(*args, **kwargs):
    """
    C++ interface function for asinh_grad_.
    """
def assign(*args, **kwargs):
    """
    C++ interface function for assign.
    """
def assign_(*args, **kwargs):
    """
    C++ interface function for assign_.
    """
def assign_out_(x: paddle.Tensor, output: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for assign_out_.
    """
def assign_out__grad(*args, **kwargs):
    """
    C++ interface function for assign_out__grad.
    """
def assign_out__grad_(*args, **kwargs):
    """
    C++ interface function for assign_out__grad_.
    """
def assign_pos(x: paddle.Tensor, cum_count: paddle.Tensor, eff_num_len: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for assign_pos.
    """
def assign_value(*args, **kwargs):
    """
    C++ interface function for assign_value.
    """
def assign_value_(output: paddle.Tensor, shape: list[int], dtype: paddle._typing.DTypeLike, values: list[float], place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for assign_value_.
    """
def atan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atan.
    """
def atan2(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atan2.
    """
def atan2_grad(*args, **kwargs):
    """
    C++ interface function for atan2_grad.
    """
def atan_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atan_.
    """
def atan_grad(*args, **kwargs):
    """
    C++ interface function for atan_grad.
    """
def atan_grad_(*args, **kwargs):
    """
    C++ interface function for atan_grad_.
    """
def atanh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atanh.
    """
def atanh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atanh_.
    """
def atanh_grad(*args, **kwargs):
    """
    C++ interface function for atanh_grad.
    """
def atanh_grad_(*args, **kwargs):
    """
    C++ interface function for atanh_grad_.
    """
def attention_lstm(x: paddle.Tensor, c0: paddle.Tensor, h0: paddle.Tensor, attention_weight: paddle.Tensor, attention_bias: paddle.Tensor, attention_scalar: paddle.Tensor, attention_scalar_bias: paddle.Tensor, lstm_weight: paddle.Tensor, lstm_bias: paddle.Tensor, gate_activation: str = "sigmoid", cell_activation: str = "tanh", candidate_activation: str = "tanh") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for attention_lstm.
    """
def auc(x: paddle.Tensor, label: paddle.Tensor, stat_pos: paddle.Tensor, stat_neg: paddle.Tensor, ins_tag_weight: paddle.Tensor, curve: str = "ROC", num_thresholds: int = (2 << 12) - 1, slide_steps: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for auc.
    """
def average_accumulates_(param: paddle.Tensor, in_sum_1: paddle.Tensor, in_sum_2: paddle.Tensor, in_sum_3: paddle.Tensor, in_num_accumulates: paddle.Tensor, in_old_num_accumulates: paddle.Tensor, in_num_updates: paddle.Tensor, average_window: float = 0, max_average_window: int = "INT64_MAX", min_average_window: int = 10000) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for average_accumulates_.
    """
def baddbmm(input: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor, beta: float = 1.0, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for baddbmm.
    """
def baddbmm_(input: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor, beta: float = 1.0, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for baddbmm_.
    """
def baddbmm_grad(*args, **kwargs):
    """
    C++ interface function for baddbmm_grad.
    """
def batch_norm(*args, **kwargs):
    """
    C++ interface function for batch_norm.
    """
def batch_norm_(*args, **kwargs):
    """
    C++ interface function for batch_norm_.
    """
def batch_norm_grad(*args, **kwargs):
    """
    C++ interface function for batch_norm_grad.
    """
def bce_loss(input: paddle.Tensor, label: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bce_loss.
    """
def bce_loss_(input: paddle.Tensor, label: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bce_loss_.
    """
def bce_loss_grad(*args, **kwargs):
    """
    C++ interface function for bce_loss_grad.
    """
def bce_loss_grad_(*args, **kwargs):
    """
    C++ interface function for bce_loss_grad_.
    """
def beam_search(pre_ids: paddle.Tensor, pre_scores: paddle.Tensor, ids: paddle.Tensor, scores: paddle.Tensor, level: int, beam_size: int, end_id: int, is_accumulated: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for beam_search.
    """
def beam_search_decode(*args, **kwargs):
    """
    C++ interface function for beam_search_decode.
    """
def bernoulli(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bernoulli.
    """
def bicubic_interp(x: paddle.Tensor, out_size: paddle.Tensor, size_tensor: list[paddle.Tensor], scale_tensor: paddle.Tensor, data_format: str = "NCHW", out_d: int = 0, out_h: int = 0, out_w: int = 0, scale: list[float] = [], interp_method: str = "bilinear", align_corners: bool = True, align_mode: int = 1) -> paddle.Tensor:
    """
    C++ interface function for bicubic_interp.
    """
def bicubic_interp_grad(*args, **kwargs):
    """
    C++ interface function for bicubic_interp_grad.
    """
def bilinear(x: paddle.Tensor, y: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bilinear.
    """
def bilinear_grad(*args, **kwargs):
    """
    C++ interface function for bilinear_grad.
    """
def bilinear_interp(x: paddle.Tensor, out_size: paddle.Tensor, size_tensor: list[paddle.Tensor], scale_tensor: paddle.Tensor, data_format: str = "NCHW", out_d: int = 0, out_h: int = 0, out_w: int = 0, scale: list[float] = [], interp_method: str = "bilinear", align_corners: bool = True, align_mode: int = 1) -> paddle.Tensor:
    """
    C++ interface function for bilinear_interp.
    """
def bilinear_interp_grad(*args, **kwargs):
    """
    C++ interface function for bilinear_interp_grad.
    """
def bincount(x: paddle.Tensor, weights: paddle.Tensor, minlength: int = 0) -> paddle.Tensor:
    """
    C++ interface function for bincount.
    """
def binomial(count: paddle.Tensor, prob: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for binomial.
    """
def bipartite_match(dist_mat: paddle.Tensor, match_type: str = "bipartite", dist_threshold: float = 0.5) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for bipartite_match.
    """
def bitwise_and(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_and.
    """
def bitwise_and_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_and_.
    """
def bitwise_left_shift(x: paddle.Tensor, y: paddle.Tensor, is_arithmetic: bool = True) -> paddle.Tensor:
    """
    C++ interface function for bitwise_left_shift.
    """
def bitwise_left_shift_(x: paddle.Tensor, y: paddle.Tensor, is_arithmetic: bool = True) -> paddle.Tensor:
    """
    C++ interface function for bitwise_left_shift_.
    """
def bitwise_not(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_not.
    """
def bitwise_not_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_not_.
    """
def bitwise_or(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_or.
    """
def bitwise_or_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_or_.
    """
def bitwise_right_shift(x: paddle.Tensor, y: paddle.Tensor, is_arithmetic: bool = True) -> paddle.Tensor:
    """
    C++ interface function for bitwise_right_shift.
    """
def bitwise_right_shift_(x: paddle.Tensor, y: paddle.Tensor, is_arithmetic: bool = True) -> paddle.Tensor:
    """
    C++ interface function for bitwise_right_shift_.
    """
def bitwise_xor(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_xor.
    """
def bitwise_xor_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_xor_.
    """
def blha_get_max_len(*args, **kwargs):
    """
    C++ interface function for blha_get_max_len.
    """
def block_multihead_attention_(*args, **kwargs):
    """
    C++ interface function for block_multihead_attention_.
    """
def block_multihead_attention_xpu_(*args, **kwargs):
    """
    C++ interface function for block_multihead_attention_xpu_.
    """
def bmm(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bmm.
    """
def bmm_grad(*args, **kwargs):
    """
    C++ interface function for bmm_grad.
    """
def box_clip(input: paddle.Tensor, im_info: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for box_clip.
    """
def box_coder(prior_box: paddle.Tensor, prior_box_var: paddle.Tensor, target_box: paddle.Tensor, code_type: str = "encode_center_size", box_normalized: bool = True, axis: int = 0, variance: list[float] = []) -> paddle.Tensor:
    """
    C++ interface function for box_coder.
    """
def broadcast(x: paddle.Tensor, ring_id: int = 0, root: int = 0) -> paddle.Tensor:
    """
    C++ interface function for broadcast.
    """
def broadcast_(x: paddle.Tensor, ring_id: int = 0, root: int = 0) -> paddle.Tensor:
    """
    C++ interface function for broadcast_.
    """
def broadcast_tensors(input: list[paddle.Tensor]) -> list[paddle.Tensor]:
    """
    C++ interface function for broadcast_tensors.
    """
def broadcast_tensors_grad(*args, **kwargs):
    """
    C++ interface function for broadcast_tensors_grad.
    """
def builtin_combine(*args, **kwargs):
    """
    C++ interface function for builtin_combine_op.
    """
def builtin_split(*args, **kwargs):
    """
    C++ interface function for builtin_split_op.
    """
def c_allreduce_sum(x: paddle.Tensor, ring_id: int, use_calc_stream: bool, use_model_parallel: bool) -> paddle.Tensor:
    """
    C++ interface function for c_allreduce_sum.
    """
def c_allreduce_sum_(x: paddle.Tensor, ring_id: int, use_calc_stream: bool, use_model_parallel: bool) -> paddle.Tensor:
    """
    C++ interface function for c_allreduce_sum_.
    """
def c_concat(x: paddle.Tensor, rank: int, nranks: int, ring_id: int, use_calc_stream: bool, use_model_parallel: bool) -> paddle.Tensor:
    """
    C++ interface function for c_concat.
    """
def c_embedding(*args, **kwargs):
    """
    C++ interface function for c_embedding.
    """
def c_embedding_grad(*args, **kwargs):
    """
    C++ interface function for c_embedding_grad.
    """
def c_identity(x: paddle.Tensor, ring_id: int, use_calc_stream: bool, use_model_parallel: bool) -> paddle.Tensor:
    """
    C++ interface function for c_identity.
    """
def c_identity_(x: paddle.Tensor, ring_id: int, use_calc_stream: bool, use_model_parallel: bool) -> paddle.Tensor:
    """
    C++ interface function for c_identity_.
    """
def c_softmax_with_cross_entropy(logits: paddle.Tensor, label: paddle.Tensor, ignore_index: int = -100, ring_id: int = 0, rank: int = 0, nranks: int = 0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for c_softmax_with_cross_entropy.
    """
def c_softmax_with_cross_entropy_grad(*args, **kwargs):
    """
    C++ interface function for c_softmax_with_cross_entropy_grad.
    """
def c_softmax_with_multi_label_cross_entropy(*args, **kwargs):
    """
    C++ interface function for c_softmax_with_multi_label_cross_entropy.
    """
def c_softmax_with_multi_label_cross_entropy_grad(*args, **kwargs):
    """
    C++ interface function for c_softmax_with_multi_label_cross_entropy_grad.
    """
def calc_reduced_attn_scores(q: paddle.Tensor, k: paddle.Tensor, softmax_lse: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for calc_reduced_attn_scores.
    """
def cast(x: paddle.Tensor, dtype: paddle._typing.DTypeLike) -> paddle.Tensor:
    """
    C++ interface function for cast.
    """
def cast_(x: paddle.Tensor, dtype: paddle._typing.DTypeLike) -> paddle.Tensor:
    """
    C++ interface function for cast_.
    """
def ceil(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for ceil.
    """
def ceil_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for ceil_.
    """
def ceil_grad(*args, **kwargs):
    """
    C++ interface function for ceil_grad.
    """
def ceil_grad_(*args, **kwargs):
    """
    C++ interface function for ceil_grad_.
    """
def celu(x: paddle.Tensor, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for celu.
    """
def celu_grad(*args, **kwargs):
    """
    C++ interface function for celu_grad.
    """
def celu_grad_(*args, **kwargs):
    """
    C++ interface function for celu_grad_.
    """
def channel_shuffle(x: paddle.Tensor, groups: int, data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for channel_shuffle.
    """
def channel_shuffle_grad(*args, **kwargs):
    """
    C++ interface function for channel_shuffle_grad.
    """
def check_finite_and_unscale_(x: list[paddle.Tensor], scale: paddle.Tensor) -> tuple[list[paddle.Tensor], paddle.Tensor]:
    """
    C++ interface function for check_finite_and_unscale_.
    """
def check_numerics(tensor: paddle.Tensor, op_type: str = "", var_name: str = "", check_nan_inf_level: int = 0, stack_height_limit: int = -1, output_dir: str = "") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for check_numerics.
    """
def cholesky(x: paddle.Tensor, upper: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cholesky.
    """
def cholesky_grad(*args, **kwargs):
    """
    C++ interface function for cholesky_grad.
    """
def cholesky_solve(x: paddle.Tensor, y: paddle.Tensor, upper: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cholesky_solve.
    """
def cholesky_solve_grad(*args, **kwargs):
    """
    C++ interface function for cholesky_solve_grad.
    """
def chunk_eval(inference: paddle.Tensor, label: paddle.Tensor, seq_length: paddle.Tensor, num_chunk_types: int, chunk_scheme: str = "IOB", excluded_chunk_types: list[int] = []) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for chunk_eval.
    """
def class_center_sample(label: paddle.Tensor, num_classes: int, num_samples: int, ring_id: int = 0, rank: int = 0, nranks: int = 1, fix_seed: bool = False, seed: int = 0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for class_center_sample.
    """
def clip(x: paddle.Tensor, min: float, max: float) -> paddle.Tensor:
    """
    C++ interface function for clip.
    """
def clip_(x: paddle.Tensor, min: float, max: float) -> paddle.Tensor:
    """
    C++ interface function for clip_.
    """
def clip_by_norm(x: paddle.Tensor, max_norm: float) -> paddle.Tensor:
    """
    C++ interface function for clip_by_norm.
    """
def clip_grad(*args, **kwargs):
    """
    C++ interface function for clip_grad.
    """
def clip_grad_(*args, **kwargs):
    """
    C++ interface function for clip_grad_.
    """
def coalesce_tensor(input: list[paddle.Tensor], dtype: paddle._typing.DTypeLike, copy_data: bool = False, set_constant: bool = False, persist_output: bool = False, constant: float = 0.0, use_align: bool = True, align_size: int = -1, size_of_dtype: int = -1, concated_shapes: list[int] = [], concated_ranks: list[int] = []) -> tuple[list[paddle.Tensor], paddle.Tensor]:
    """
    C++ interface function for coalesce_tensor.
    """
def coalesce_tensor_(input: list[paddle.Tensor], dtype: paddle._typing.DTypeLike, copy_data: bool = False, set_constant: bool = False, persist_output: bool = False, constant: float = 0.0, use_align: bool = True, align_size: int = -1, size_of_dtype: int = -1, concated_shapes: list[int] = [], concated_ranks: list[int] = []) -> tuple[list[paddle.Tensor], paddle.Tensor]:
    """
    C++ interface function for coalesce_tensor_.
    """
def collect_fpn_proposals(multi_level_rois: list[paddle.Tensor], multi_level_scores: list[paddle.Tensor], multi_level_rois_num: list[paddle.Tensor], post_nms_topn: int) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for collect_fpn_proposals.
    """
def complex(real: paddle.Tensor, imag: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for complex.
    """
def complex_grad(*args, **kwargs):
    """
    C++ interface function for complex_grad.
    """
def concat(x: list[paddle.Tensor], axis: float = 0) -> paddle.Tensor:
    """
    C++ interface function for concat.
    """
def concat_grad(*args, **kwargs):
    """
    C++ interface function for concat_grad.
    """
def conj(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for conj.
    """
def conv2d(input: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1], paddings: list[int] = [0, 0], padding_algorithm: str = "EXPLICIT", dilations: list[int] = [1, 1], groups: int = 1, data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for conv2d.
    """
def conv2d_transpose(x: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1], paddings: list[int] = [0, 0], output_padding: list[int] = [], output_size: list[int] = [], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1], data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for conv2d_transpose.
    """
def conv2d_transpose_bias(x: paddle.Tensor, filter: paddle.Tensor, bias: paddle.Tensor, strides: list[int] = [1, 1], paddings: list[int] = [0, 0], output_padding: list[int] = [], output_size: list[int] = [], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1], data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for conv2d_transpose_bias.
    """
def conv2d_transpose_grad(*args, **kwargs):
    """
    C++ interface function for conv2d_transpose_grad.
    """
def conv3d(input: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1, 1], paddings: list[int] = [0, 0, 0], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1, 1], data_format: str = "NCDHW") -> paddle.Tensor:
    """
    C++ interface function for conv3d.
    """
def conv3d_grad(*args, **kwargs):
    """
    C++ interface function for conv3d_grad.
    """
def conv3d_transpose(x: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1, 1], paddings: list[int] = [0, 0, 0], output_padding: list[int] = [], output_size: list[int] = [], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1, 1], data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for conv3d_transpose.
    """
def conv3d_transpose_grad(*args, **kwargs):
    """
    C++ interface function for conv3d_transpose_grad.
    """
def copysign(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for copysign.
    """
def copysign_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for copysign_.
    """
def copysign_grad(*args, **kwargs):
    """
    C++ interface function for copysign_grad.
    """
def copysign_grad_(*args, **kwargs):
    """
    C++ interface function for copysign_grad_.
    """
def correlation(input1: paddle.Tensor, input2: paddle.Tensor, pad_size: int, kernel_size: int, max_displacement: int, stride1: int, stride2: int, corr_type_multiply: int = 1) -> paddle.Tensor:
    """
    C++ interface function for correlation.
    """
def correlation_grad(*args, **kwargs):
    """
    C++ interface function for correlation_grad.
    """
def cos(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for cos.
    """
def cos_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for cos_.
    """
def cos_grad(*args, **kwargs):
    """
    C++ interface function for cos_grad.
    """
def cos_grad_(*args, **kwargs):
    """
    C++ interface function for cos_grad_.
    """
def cosh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for cosh.
    """
def cosh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for cosh_.
    """
def cosh_grad(*args, **kwargs):
    """
    C++ interface function for cosh_grad.
    """
def cosh_grad_(*args, **kwargs):
    """
    C++ interface function for cosh_grad_.
    """
def create_array(*args, **kwargs):
    """
    C++ interface function for create_array.
    """
def create_array_like(*args, **kwargs):
    """
    C++ interface function for create_array_like.
    """
def crf_decoding(emission: paddle.Tensor, transition: paddle.Tensor, label: paddle.Tensor, length: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for crf_decoding.
    """
def crop(x: paddle.Tensor, shape: list[int] = [], offsets: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for crop.
    """
def crop_grad(*args, **kwargs):
    """
    C++ interface function for crop_grad.
    """
def cross(x: paddle.Tensor, y: paddle.Tensor, axis: int = 9) -> paddle.Tensor:
    """
    C++ interface function for cross.
    """
def cross_entropy_with_softmax(input: paddle.Tensor, label: paddle.Tensor, soft_label: bool = False, use_softmax: bool = True, numeric_stable_mode: bool = True, ignore_index: int = -100, axis: int = -1) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for cross_entropy_with_softmax.
    """
def cross_entropy_with_softmax_(input: paddle.Tensor, label: paddle.Tensor, soft_label: bool = False, use_softmax: bool = True, numeric_stable_mode: bool = True, ignore_index: int = -100, axis: int = -1) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for cross_entropy_with_softmax_.
    """
def cross_entropy_with_softmax_grad(*args, **kwargs):
    """
    C++ interface function for cross_entropy_with_softmax_grad.
    """
def cross_entropy_with_softmax_grad_(*args, **kwargs):
    """
    C++ interface function for cross_entropy_with_softmax_grad_.
    """
def cross_grad(*args, **kwargs):
    """
    C++ interface function for cross_grad.
    """
def ctc_align(input: paddle.Tensor, input_length: paddle.Tensor, blank: int = 0, merge_repeated: bool = True, padding_value: int = 0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for ctc_align.
    """
def cudnn_lstm(x: paddle.Tensor, init_h: paddle.Tensor, init_c: paddle.Tensor, w: paddle.Tensor, weight_list: list[paddle.Tensor], sequence_length: paddle.Tensor, dropout_prob: float = 0.0, is_bidirec: bool = False, hidden_size: int = 100, num_layers: int = 1, is_test: bool = False, seed: int = 0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for cudnn_lstm.
    """
def cudnn_lstm_grad(*args, **kwargs):
    """
    C++ interface function for cudnn_lstm_grad.
    """
def cummax(x: paddle.Tensor, axis: int = -1, dtype: paddle._typing.DTypeLike = "DataType::INT64") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for cummax.
    """
def cummax_grad(*args, **kwargs):
    """
    C++ interface function for cummax_grad.
    """
def cummin(x: paddle.Tensor, axis: int = -1, dtype: paddle._typing.DTypeLike = "DataType::INT64") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for cummin.
    """
def cummin_grad(*args, **kwargs):
    """
    C++ interface function for cummin_grad.
    """
def cumprod(x: paddle.Tensor, dim: int, exclusive: bool = False, reverse: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cumprod.
    """
def cumprod_(x: paddle.Tensor, dim: int, exclusive: bool = False, reverse: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cumprod_.
    """
def cumprod_grad(*args, **kwargs):
    """
    C++ interface function for cumprod_grad.
    """
def cumsum(x: paddle.Tensor, axis: float = -1, flatten: bool = False, exclusive: bool = False, reverse: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cumsum.
    """
def cumsum_(x: paddle.Tensor, axis: float = -1, flatten: bool = False, exclusive: bool = False, reverse: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cumsum_.
    """
def cumsum_grad(*args, **kwargs):
    """
    C++ interface function for cumsum_grad.
    """
def cvm(x: paddle.Tensor, cvm: paddle.Tensor, use_cvm: bool = True) -> paddle.Tensor:
    """
    C++ interface function for cvm.
    """
def cvm_grad(*args, **kwargs):
    """
    C++ interface function for cvm_grad.
    """
def data(name: str, shape: list[int], dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike) -> paddle.Tensor:
    """
    C++ interface function for data.
    """
def decode_jpeg(x: paddle.Tensor, mode: str, place: paddle._typing.PlaceLike) -> paddle.Tensor:
    """
    C++ interface function for decode_jpeg.
    """
def deformable_conv(x: paddle.Tensor, offset: paddle.Tensor, filter: paddle.Tensor, mask: paddle.Tensor, strides: list[int], paddings: list[int], dilations: list[int], deformable_groups: int, groups: int, im2col_step: int) -> paddle.Tensor:
    """
    C++ interface function for deformable_conv.
    """
def deformable_conv_grad(*args, **kwargs):
    """
    C++ interface function for deformable_conv_grad.
    """
def depend(x: paddle.Tensor, dep: list[paddle.Tensor]) -> paddle.Tensor:
    """
    C++ interface function for depend.
    """
def depthwise_conv2d(input: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1], paddings: list[int] = [0, 0], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1], data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for depthwise_conv2d.
    """
def depthwise_conv2d_grad(*args, **kwargs):
    """
    C++ interface function for depthwise_conv2d_grad.
    """
def depthwise_conv2d_transpose(x: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1], paddings: list[int] = [0, 0], output_padding: list[int] = [], output_size: list[int] = [], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1], data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for depthwise_conv2d_transpose.
    """
def depthwise_conv2d_transpose_grad(*args, **kwargs):
    """
    C++ interface function for depthwise_conv2d_transpose_grad.
    """
def dequantize_abs_max(x: paddle.Tensor, scale: paddle.Tensor, max_range: float) -> paddle.Tensor:
    """
    C++ interface function for dequantize_abs_max.
    """
def dequantize_linear(*args, **kwargs):
    """
    C++ interface function for dequantize_linear.
    """
def dequantize_linear_(*args, **kwargs):
    """
    C++ interface function for dequantize_linear_.
    """
def dequantize_log(x: paddle.Tensor, dict: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for dequantize_log.
    """
def det(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for det.
    """
def det_grad(*args, **kwargs):
    """
    C++ interface function for det_grad.
    """
def dgc_clip_by_norm(x: paddle.Tensor, current_step: paddle.Tensor, max_norm: float, rampup_begin_step: float = -1.0) -> paddle.Tensor:
    """
    C++ interface function for dgc_clip_by_norm.
    """
def diag(x: paddle.Tensor, offset: int = 0, padding_value: float = 0.0) -> paddle.Tensor:
    """
    C++ interface function for diag.
    """
def diag_embed(input: paddle.Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> paddle.Tensor:
    """
    C++ interface function for diag_embed.
    """
def diag_grad(*args, **kwargs):
    """
    C++ interface function for diag_grad.
    """
def diagonal(x: paddle.Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> paddle.Tensor:
    """
    C++ interface function for diagonal.
    """
def diagonal_grad(*args, **kwargs):
    """
    C++ interface function for diagonal_grad.
    """
def digamma(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for digamma.
    """
def digamma_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for digamma_.
    """
def digamma_grad(*args, **kwargs):
    """
    C++ interface function for digamma_grad.
    """
def dirichlet(alpha: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for dirichlet.
    """
def disable_check_model_nan_inf(x: paddle.Tensor, flag: int = 0) -> paddle.Tensor:
    """
    C++ interface function for disable_check_model_nan_inf.
    """
def disable_check_model_nan_inf_grad(*args, **kwargs):
    """
    C++ interface function for disable_check_model_nan_inf_grad.
    """
def dist(x: paddle.Tensor, y: paddle.Tensor, p: float = 2.0) -> paddle.Tensor:
    """
    C++ interface function for dist.
    """
def dist_grad(*args, **kwargs):
    """
    C++ interface function for dist_grad.
    """
def distribute_fpn_proposals(*args, **kwargs):
    """
    C++ interface function for distribute_fpn_proposals.
    """
def distributed_fused_lamb_init(*args, **kwargs):
    """
    C++ interface function for distributed_fused_lamb_init.
    """
def distributed_fused_lamb_init_(*args, **kwargs):
    """
    C++ interface function for distributed_fused_lamb_init_.
    """
def divide(*args, **kwargs):
    """
    C++ interface function for divide.
    """
def divide_(*args, **kwargs):
    """
    C++ interface function for divide_.
    """
def divide_grad(*args, **kwargs):
    """
    C++ interface function for divide_grad.
    """
def dot(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for dot.
    """
def dot_grad(*args, **kwargs):
    """
    C++ interface function for dot_grad.
    """
def dropout(x: paddle.Tensor, seed_tensor: paddle.Tensor, p: float, is_test: bool, mode: str, seed: int, fix_seed: bool) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for dropout.
    """
def dropout_grad(*args, **kwargs):
    """
    C++ interface function for dropout_grad.
    """
def edit_distance(hyps: paddle.Tensor, refs: paddle.Tensor, hypslength: paddle.Tensor, refslength: paddle.Tensor, normalized: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for edit_distance.
    """
def eig(x: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for eig.
    """
def eig_grad(*args, **kwargs):
    """
    C++ interface function for eig_grad.
    """
def eigh(x: paddle.Tensor, UPLO: str = "L") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for eigh.
    """
def eigh_grad(*args, **kwargs):
    """
    C++ interface function for eigh_grad.
    """
def eigvals(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for eigvals.
    """
def eigvalsh(x: paddle.Tensor, uplo: str = "L", is_test: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for eigvalsh.
    """
def eigvalsh_grad(*args, **kwargs):
    """
    C++ interface function for eigvalsh_grad.
    """
def einsum(*args, **kwargs):
    """
    C++ interface function for einsum.
    """
def einsum_grad(*args, **kwargs):
    """
    C++ interface function for einsum_grad.
    """
def elementwise_pow(*args, **kwargs):
    """
    C++ interface function for elementwise_pow.
    """
def elementwise_pow_grad(*args, **kwargs):
    """
    C++ interface function for elementwise_pow_grad.
    """
def elu(x: paddle.Tensor, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for elu.
    """
def elu_(x: paddle.Tensor, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for elu_.
    """
def elu_grad(*args, **kwargs):
    """
    C++ interface function for elu_grad.
    """
def elu_grad_(*args, **kwargs):
    """
    C++ interface function for elu_grad_.
    """
def embedding(*args, **kwargs):
    """
    C++ interface function for embedding.
    """
def embedding_grad(*args, **kwargs):
    """
    C++ interface function for embedding_grad.
    """
def embedding_with_scaled_gradient(x: paddle.Tensor, weight: paddle.Tensor, padding_idx: int = -1) -> paddle.Tensor:
    """
    C++ interface function for embedding_with_scaled_gradient.
    """
def embedding_with_scaled_gradient_grad(*args, **kwargs):
    """
    C++ interface function for embedding_with_scaled_gradient_grad.
    """
def empty(shape: list[int], dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for empty.
    """
def empty_like(x: paddle.Tensor, dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for empty_like.
    """
def enable_check_model_nan_inf(x: paddle.Tensor, flag: int = 1) -> paddle.Tensor:
    """
    C++ interface function for enable_check_model_nan_inf.
    """
def enable_check_model_nan_inf_grad(*args, **kwargs):
    """
    C++ interface function for enable_check_model_nan_inf_grad.
    """
def equal(*args, **kwargs):
    """
    C++ interface function for equal.
    """
def equal_(*args, **kwargs):
    """
    C++ interface function for equal_.
    """
def equal_all(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for equal_all.
    """
def erf(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for erf.
    """
def erf_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for erf_.
    """
def erf_grad(*args, **kwargs):
    """
    C++ interface function for erf_grad.
    """
def erfinv(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for erfinv.
    """
def erfinv_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for erfinv_.
    """
def erfinv_grad(*args, **kwargs):
    """
    C++ interface function for erfinv_grad.
    """
def exp(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for exp.
    """
def exp_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for exp_.
    """
def exp_grad(*args, **kwargs):
    """
    C++ interface function for exp_grad.
    """
def exp_grad_(*args, **kwargs):
    """
    C++ interface function for exp_grad_.
    """
def expand(x: paddle.Tensor, shape: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for expand.
    """
def expand_as(x: paddle.Tensor, y: paddle.Tensor, target_shape: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for expand_as.
    """
def expand_as_grad(*args, **kwargs):
    """
    C++ interface function for expand_as_grad.
    """
def expand_grad(*args, **kwargs):
    """
    C++ interface function for expand_grad.
    """
def expm1(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for expm1.
    """
def expm1_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for expm1_.
    """
def expm1_grad(*args, **kwargs):
    """
    C++ interface function for expm1_grad.
    """
def expm1_grad_(*args, **kwargs):
    """
    C++ interface function for expm1_grad_.
    """
def exponential_(x: paddle.Tensor, lam: float) -> paddle.Tensor:
    """
    C++ interface function for exponential_.
    """
def eye(num_rows: float, num_columns: float, dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for eye.
    """
def fake_channel_wise_dequantize_max_abs(x: paddle.Tensor, scales: list[paddle.Tensor], quant_bits: list[int] = [8], quant_axis: int = 0, x_num_col_dims: int = 1) -> paddle.Tensor:
    """
    C++ interface function for fake_channel_wise_dequantize_max_abs.
    """
def fake_channel_wise_quantize_abs_max(x: paddle.Tensor, bit_length: int = 8, round_type: int = 1, quant_axis: int = 0, is_test: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_channel_wise_quantize_abs_max.
    """
def fake_channel_wise_quantize_dequantize_abs_max(x: paddle.Tensor, bit_length: int = 8, round_type: int = 1, quant_axis: int = 0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_channel_wise_quantize_dequantize_abs_max.
    """
def fake_channel_wise_quantize_dequantize_abs_max_grad(*args, **kwargs):
    """
    C++ interface function for fake_channel_wise_quantize_dequantize_abs_max_grad.
    """
def fake_dequantize_max_abs(x: paddle.Tensor, scale: paddle.Tensor, max_range: float) -> paddle.Tensor:
    """
    C++ interface function for fake_dequantize_max_abs.
    """
def fake_quantize_abs_max(x: paddle.Tensor, bit_length: int = 8, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_abs_max.
    """
def fake_quantize_dequantize_abs_max(x: paddle.Tensor, bit_length: int = 8, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_dequantize_abs_max.
    """
def fake_quantize_dequantize_abs_max_grad(*args, **kwargs):
    """
    C++ interface function for fake_quantize_dequantize_abs_max_grad.
    """
def fake_quantize_dequantize_moving_average_abs_max(x: paddle.Tensor, in_scale: paddle.Tensor, in_accum: paddle.Tensor, in_state: paddle.Tensor, moving_rate: float = 0.9, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_dequantize_moving_average_abs_max.
    """
def fake_quantize_dequantize_moving_average_abs_max_(x: paddle.Tensor, in_scale: paddle.Tensor, in_accum: paddle.Tensor, in_state: paddle.Tensor, moving_rate: float = 0.9, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_dequantize_moving_average_abs_max_.
    """
def fake_quantize_dequantize_moving_average_abs_max_grad(*args, **kwargs):
    """
    C++ interface function for fake_quantize_dequantize_moving_average_abs_max_grad.
    """
def fake_quantize_moving_average_abs_max(x: paddle.Tensor, in_scale: paddle.Tensor, in_accum: paddle.Tensor, in_state: paddle.Tensor, moving_rate: float = 0.9, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_moving_average_abs_max.
    """
def fake_quantize_moving_average_abs_max_(x: paddle.Tensor, in_scale: paddle.Tensor, in_accum: paddle.Tensor, in_state: paddle.Tensor, moving_rate: float = 0.9, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_moving_average_abs_max_.
    """
def fake_quantize_range_abs_max(x: paddle.Tensor, in_scale: paddle.Tensor, iter: paddle.Tensor, window_size: int = 10000, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_range_abs_max.
    """
def fake_quantize_range_abs_max_(x: paddle.Tensor, in_scale: paddle.Tensor, iter: paddle.Tensor, window_size: int = 10000, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_range_abs_max_.
    """
def fc(*args, **kwargs):
    """
    C++ interface function for fc.
    """
def fetch(*args, **kwargs):
    """
    C++ interface function for fetch.
    """
def fft_c2c(x: paddle.Tensor, axes: list[int], normalization: str, forward: bool) -> paddle.Tensor:
    """
    C++ interface function for fft_c2c.
    """
def fft_c2c_grad(*args, **kwargs):
    """
    C++ interface function for fft_c2c_grad.
    """
def fft_c2r(x: paddle.Tensor, axes: list[int], normalization: str, forward: bool, last_dim_size: int = 0) -> paddle.Tensor:
    """
    C++ interface function for fft_c2r.
    """
def fft_c2r_grad(*args, **kwargs):
    """
    C++ interface function for fft_c2r_grad.
    """
def fft_r2c(x: paddle.Tensor, axes: list[int], normalization: str, forward: bool, onesided: bool) -> paddle.Tensor:
    """
    C++ interface function for fft_r2c.
    """
def fft_r2c_grad(*args, **kwargs):
    """
    C++ interface function for fft_r2c_grad.
    """
def fill(x: paddle.Tensor, value: float = 0) -> paddle.Tensor:
    """
    C++ interface function for fill.
    """
def fill_(x: paddle.Tensor, value: float = 0) -> paddle.Tensor:
    """
    C++ interface function for fill_.
    """
def fill_diagonal(x: paddle.Tensor, value: float = 0, offset: int = 0, wrap: bool = False) -> paddle.Tensor:
    """
    C++ interface function for fill_diagonal.
    """
def fill_diagonal_(x: paddle.Tensor, value: float = 0, offset: int = 0, wrap: bool = False) -> paddle.Tensor:
    """
    C++ interface function for fill_diagonal_.
    """
def fill_diagonal_grad(*args, **kwargs):
    """
    C++ interface function for fill_diagonal_grad.
    """
def fill_diagonal_tensor(x: paddle.Tensor, y: paddle.Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> paddle.Tensor:
    """
    C++ interface function for fill_diagonal_tensor.
    """
def fill_diagonal_tensor_(x: paddle.Tensor, y: paddle.Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> paddle.Tensor:
    """
    C++ interface function for fill_diagonal_tensor_.
    """
def fill_diagonal_tensor_grad(*args, **kwargs):
    """
    C++ interface function for fill_diagonal_tensor_grad.
    """
def fill_diagonal_tensor_grad_(*args, **kwargs):
    """
    C++ interface function for fill_diagonal_tensor_grad_.
    """
def fill_grad(*args, **kwargs):
    """
    C++ interface function for fill_grad.
    """
def fill_grad_(*args, **kwargs):
    """
    C++ interface function for fill_grad_.
    """
def flash_attn(q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, fixed_seed_offset: paddle.Tensor, attn_mask: paddle.Tensor, dropout: float = 0.0, causal: bool = False, return_softmax: bool = False, is_test: bool = False, rng_name: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for flash_attn.
    """
def flash_attn_grad(*args, **kwargs):
    """
    C++ interface function for flash_attn_grad.
    """
def flash_attn_qkvpacked(qkv: paddle.Tensor, fixed_seed_offset: paddle.Tensor, attn_mask: paddle.Tensor, dropout: float = 0.0, causal: bool = False, return_softmax: bool = False, is_test: bool = False, rng_name: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for flash_attn_qkvpacked.
    """
def flash_attn_qkvpacked_grad(*args, **kwargs):
    """
    C++ interface function for flash_attn_qkvpacked_grad.
    """
def flash_attn_unpadded(q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, cu_seqlens_q: paddle.Tensor, cu_seqlens_k: paddle.Tensor, fixed_seed_offset: paddle.Tensor, attn_mask: paddle.Tensor, max_seqlen_q: float, max_seqlen_k: float, scale: float, dropout: float = 0.0, causal: bool = False, return_softmax: bool = False, is_test: bool = False, rng_name: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for flash_attn_unpadded.
    """
def flash_attn_unpadded_grad(*args, **kwargs):
    """
    C++ interface function for flash_attn_unpadded_grad.
    """
def flash_attn_varlen_qkvpacked(qkv: paddle.Tensor, cu_seqlens_q: paddle.Tensor, cu_seqlens_k: paddle.Tensor, fixed_seed_offset: paddle.Tensor, attn_mask: paddle.Tensor, max_seqlen_q: float, max_seqlen_k: float, scale: float, dropout: float = 0.0, causal: bool = False, return_softmax: bool = False, is_test: bool = False, rng_name: str = "", varlen_padded: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for flash_attn_varlen_qkvpacked.
    """
def flash_attn_varlen_qkvpacked_grad(*args, **kwargs):
    """
    C++ interface function for flash_attn_varlen_qkvpacked_grad.
    """
def flashmask_attention(q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, startend_row_indices: paddle.Tensor, fixed_seed_offset: paddle.Tensor, dropout: float = 0.0, causal: bool = False, return_softmax: bool = False, is_test: bool = False, rng_name: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for flashmask_attention.
    """
def flashmask_attention_grad(*args, **kwargs):
    """
    C++ interface function for flashmask_attention_grad.
    """
def flatten(x: paddle.Tensor, start_axis: int = 1, stop_axis: int = 1) -> paddle.Tensor:
    """
    C++ interface function for flatten.
    """
def flatten_(x: paddle.Tensor, start_axis: int = 1, stop_axis: int = 1) -> paddle.Tensor:
    """
    C++ interface function for flatten_.
    """
def flatten_grad(*args, **kwargs):
    """
    C++ interface function for flatten_grad.
    """
def flatten_grad_(*args, **kwargs):
    """
    C++ interface function for flatten_grad_.
    """
def flip(x: paddle.Tensor, axis: list[int]) -> paddle.Tensor:
    """
    C++ interface function for flip.
    """
def floor(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for floor.
    """
def floor_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for floor_.
    """
def floor_divide(*args, **kwargs):
    """
    C++ interface function for floor_divide.
    """
def floor_divide_(*args, **kwargs):
    """
    C++ interface function for floor_divide_.
    """
def floor_grad(*args, **kwargs):
    """
    C++ interface function for floor_grad.
    """
def floor_grad_(*args, **kwargs):
    """
    C++ interface function for floor_grad_.
    """
def fmax(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for fmax.
    """
def fmax_grad(*args, **kwargs):
    """
    C++ interface function for fmax_grad.
    """
def fmin(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for fmin.
    """
def fmin_grad(*args, **kwargs):
    """
    C++ interface function for fmin_grad.
    """
def fold(x: paddle.Tensor, output_sizes: list[int], kernel_sizes: list[int], strides: list[int], paddings: list[int], dilations: list[int]) -> paddle.Tensor:
    """
    C++ interface function for fold.
    """
def fold_grad(*args, **kwargs):
    """
    C++ interface function for fold_grad.
    """
def fp8_fp8_half_gemm_fused(*args, **kwargs):
    """
    C++ interface function for fp8_fp8_half_gemm_fused.
    """
def fractional_max_pool2d(x: paddle.Tensor, output_size: list[int], kernel_size: list[int] = [0, 0], random_u: float = 0.0, return_mask: bool = True) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fractional_max_pool2d.
    """
def fractional_max_pool2d_grad(*args, **kwargs):
    """
    C++ interface function for fractional_max_pool2d_grad.
    """
def fractional_max_pool3d(x: paddle.Tensor, output_size: list[int], kernel_size: list[int] = [0, 0, 0], random_u: float = 0.0, return_mask: bool = True) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fractional_max_pool3d.
    """
def fractional_max_pool3d_grad(*args, **kwargs):
    """
    C++ interface function for fractional_max_pool3d_grad.
    """
def frame(x: paddle.Tensor, frame_length: int, hop_length: int, axis: int = -1) -> paddle.Tensor:
    """
    C++ interface function for frame.
    """
def frame_grad(*args, **kwargs):
    """
    C++ interface function for frame_grad.
    """
def frobenius_norm(x: paddle.Tensor, axis: list[int], keep_dim: bool, reduce_all: bool) -> paddle.Tensor:
    """
    C++ interface function for frobenius_norm.
    """
def frobenius_norm_grad(*args, **kwargs):
    """
    C++ interface function for frobenius_norm_grad.
    """
def full(shape: list[int], value: float, dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for full.
    """
def full_(output: paddle.Tensor, shape: list[int], value: float, dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for full_.
    """
def full_batch_size_like(input: paddle.Tensor, shape: list[int], dtype: paddle._typing.DTypeLike, value: float, input_dim_idx: int, output_dim_idx: int, place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for full_batch_size_like.
    """
def full_int_array(value: list[int], dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for full_int_array.
    """
def full_like(x: paddle.Tensor, value: float, dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for full_like.
    """
def full_with_tensor(value: paddle.Tensor, shape: list[int], dtype: paddle._typing.DTypeLike = "DataType::FLOAT32") -> paddle.Tensor:
    """
    C++ interface function for full_with_tensor.
    """
def fused_attention(*args, **kwargs):
    """
    C++ interface function for fused_attention.
    """
def fused_attention_grad(*args, **kwargs):
    """
    C++ interface function for fused_attention_grad.
    """
def fused_batch_norm_act(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, mean: paddle.Tensor, variance: paddle.Tensor, momentum: float, epsilon: float, act_type: str) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fused_batch_norm_act.
    """
def fused_batch_norm_act_grad(*args, **kwargs):
    """
    C++ interface function for fused_batch_norm_act_grad.
    """
def fused_bias_act(*args, **kwargs):
    """
    C++ interface function for fused_bias_act.
    """
def fused_bias_dropout_residual_layer_norm(*args, **kwargs):
    """
    C++ interface function for fused_bias_dropout_residual_layer_norm.
    """
def fused_bias_dropout_residual_layer_norm_grad(*args, **kwargs):
    """
    C++ interface function for fused_bias_dropout_residual_layer_norm_grad.
    """
def fused_bias_residual_layernorm(*args, **kwargs):
    """
    C++ interface function for fused_bias_residual_layernorm.
    """
def fused_bn_add_activation(x: paddle.Tensor, z: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, mean: paddle.Tensor, variance: paddle.Tensor, momentum: float, epsilon: float, act_type: str) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fused_bn_add_activation.
    """
def fused_bn_add_activation_grad(*args, **kwargs):
    """
    C++ interface function for fused_bn_add_activation_grad.
    """
def fused_conv2d_add_act(*args, **kwargs):
    """
    C++ interface function for fused_conv2d_add_act.
    """
def fused_dropout_add(*args, **kwargs):
    """
    C++ interface function for fused_dropout_add.
    """
def fused_dropout_add_grad(*args, **kwargs):
    """
    C++ interface function for fused_dropout_add_grad.
    """
def fused_embedding_eltwise_layernorm(*args, **kwargs):
    """
    C++ interface function for fused_embedding_eltwise_layernorm.
    """
def fused_fc_elementwise_layernorm(*args, **kwargs):
    """
    C++ interface function for fused_fc_elementwise_layernorm.
    """
def fused_feedforward(*args, **kwargs):
    """
    C++ interface function for fused_feedforward.
    """
def fused_feedforward_grad(*args, **kwargs):
    """
    C++ interface function for fused_feedforward_grad.
    """
def fused_gemm_epilogue(*args, **kwargs):
    """
    C++ interface function for fused_gemm_epilogue.
    """
def fused_linear_param_grad_add(*args, **kwargs):
    """
    C++ interface function for fused_linear_param_grad_add.
    """
def fused_moe(*args, **kwargs):
    """
    C++ interface function for fused_moe.
    """
def fused_multi_transformer_(*args, **kwargs):
    """
    C++ interface function for fused_multi_transformer_.
    """
def fused_rotary_position_embedding(*args, **kwargs):
    """
    C++ interface function for fused_rotary_position_embedding.
    """
def fused_rotary_position_embedding_grad(*args, **kwargs):
    """
    C++ interface function for fused_rotary_position_embedding_grad.
    """
def fused_softmax_mask(x: paddle.Tensor, mask: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for fused_softmax_mask.
    """
def fused_softmax_mask_grad(*args, **kwargs):
    """
    C++ interface function for fused_softmax_mask_grad.
    """
def fused_softmax_mask_upper_triangle(X: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for fused_softmax_mask_upper_triangle.
    """
def fused_softmax_mask_upper_triangle_grad(*args, **kwargs):
    """
    C++ interface function for fused_softmax_mask_upper_triangle_grad.
    """
def fusion_gru(*args, **kwargs):
    """
    C++ interface function for fusion_gru.
    """
def fusion_repeated_fc_relu(*args, **kwargs):
    """
    C++ interface function for fusion_repeated_fc_relu.
    """
def fusion_seqconv_eltadd_relu(*args, **kwargs):
    """
    C++ interface function for fusion_seqconv_eltadd_relu.
    """
def fusion_seqpool_concat(*args, **kwargs):
    """
    C++ interface function for fusion_seqpool_concat.
    """
def fusion_squared_mat_sub(*args, **kwargs):
    """
    C++ interface function for fusion_squared_mat_sub.
    """
def fusion_transpose_flatten_concat(*args, **kwargs):
    """
    C++ interface function for fusion_transpose_flatten_concat.
    """
def gammaincc(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gammaincc.
    """
def gammaincc_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gammaincc_.
    """
def gammaincc_grad(*args, **kwargs):
    """
    C++ interface function for gammaincc_grad.
    """
def gammaln(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gammaln.
    """
def gammaln_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gammaln_.
    """
def gammaln_grad(*args, **kwargs):
    """
    C++ interface function for gammaln_grad.
    """
def gather(x: paddle.Tensor, index: paddle.Tensor, axis: float = 0) -> paddle.Tensor:
    """
    C++ interface function for gather.
    """
def gather_grad(*args, **kwargs):
    """
    C++ interface function for gather_grad.
    """
def gather_nd(x: paddle.Tensor, index: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gather_nd.
    """
def gather_nd_grad(*args, **kwargs):
    """
    C++ interface function for gather_nd_grad.
    """
def gather_tree(ids: paddle.Tensor, parents: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gather_tree.
    """
def gaussian(shape: list[int], mean: float, std: float, seed: int, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for gaussian.
    """
def gaussian_inplace(x: paddle.Tensor, mean: float = 0, std: float = 1.0, seed: int = 0) -> paddle.Tensor:
    """
    C++ interface function for gaussian_inplace.
    """
def gaussian_inplace_(x: paddle.Tensor, mean: float = 0, std: float = 1.0, seed: int = 0) -> paddle.Tensor:
    """
    C++ interface function for gaussian_inplace_.
    """
def gaussian_inplace_grad(*args, **kwargs):
    """
    C++ interface function for gaussian_inplace_grad.
    """
def gaussian_inplace_grad_(*args, **kwargs):
    """
    C++ interface function for gaussian_inplace_grad_.
    """
def gelu(x: paddle.Tensor, approximate: bool = False) -> paddle.Tensor:
    """
    C++ interface function for gelu.
    """
def gelu_grad(*args, **kwargs):
    """
    C++ interface function for gelu_grad.
    """
def generate_proposals(scores: paddle.Tensor, bbox_deltas: paddle.Tensor, im_shape: paddle.Tensor, anchors: paddle.Tensor, variances: paddle.Tensor, pre_nms_top_n: int, post_nms_top_n: int, nms_thresh: float, min_size: float, eta: float, pixel_offset: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for generate_proposals.
    """
def get_tensor_from_selected_rows(*args, **kwargs):
    """
    C++ interface function for get_tensor_from_selected_rows.
    """
def graph_khop_sampler(row: paddle.Tensor, colptr: paddle.Tensor, x: paddle.Tensor, eids: paddle.Tensor, sample_sizes: list[int], return_eids: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for graph_khop_sampler.
    """
def graph_sample_neighbors(row: paddle.Tensor, colptr: paddle.Tensor, x: paddle.Tensor, eids: paddle.Tensor, perm_buffer: paddle.Tensor, sample_size: int, return_eids: bool, flag_perm_buffer: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for graph_sample_neighbors.
    """
def greater_equal(*args, **kwargs):
    """
    C++ interface function for greater_equal.
    """
def greater_equal_(*args, **kwargs):
    """
    C++ interface function for greater_equal_.
    """
def greater_than(*args, **kwargs):
    """
    C++ interface function for greater_than.
    """
def greater_than_(*args, **kwargs):
    """
    C++ interface function for greater_than_.
    """
def grid_sample(x: paddle.Tensor, grid: paddle.Tensor, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True) -> paddle.Tensor:
    """
    C++ interface function for grid_sample.
    """
def grid_sample_grad(*args, **kwargs):
    """
    C++ interface function for grid_sample_grad.
    """
def group_norm(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, epsilon: float = 1e-5, groups: int = -1, data_format: str = "NCHW") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for group_norm.
    """
def group_norm_grad(*args, **kwargs):
    """
    C++ interface function for group_norm_grad.
    """
def group_norm_grad_(*args, **kwargs):
    """
    C++ interface function for group_norm_grad_.
    """
def gru(input: paddle.Tensor, h0: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, activation: str = "tanh", gate_activation: str = "sigmoid", is_reverse: bool = False, origin_mode: bool = False, is_test: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for gru.
    """
def gru_grad(*args, **kwargs):
    """
    C++ interface function for gru_grad.
    """
def gru_unit(input: paddle.Tensor, hidden_prev: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, activation: int = 2, gate_activation: int = 1, origin_mode: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for gru_unit.
    """
def gru_unit_grad(*args, **kwargs):
    """
    C++ interface function for gru_unit_grad.
    """
def gumbel_softmax(x: paddle.Tensor, temperature: float = 1.0, hard: bool = False, axis: int = -1) -> paddle.Tensor:
    """
    C++ interface function for gumbel_softmax.
    """
def gumbel_softmax_grad(*args, **kwargs):
    """
    C++ interface function for gumbel_softmax_grad.
    """
def hardshrink(x: paddle.Tensor, threshold: float = 0.5) -> paddle.Tensor:
    """
    C++ interface function for hardshrink.
    """
def hardshrink_grad(*args, **kwargs):
    """
    C++ interface function for hardshrink_grad.
    """
def hardshrink_grad_(*args, **kwargs):
    """
    C++ interface function for hardshrink_grad_.
    """
def hardsigmoid(x: paddle.Tensor, slope: float = 0.2, offset: float = 0.5) -> paddle.Tensor:
    """
    C++ interface function for hardsigmoid.
    """
def hardsigmoid_grad(*args, **kwargs):
    """
    C++ interface function for hardsigmoid_grad.
    """
def hardsigmoid_grad_(*args, **kwargs):
    """
    C++ interface function for hardsigmoid_grad_.
    """
def hardswish(*args, **kwargs):
    """
    C++ interface function for hardswish.
    """
def hardswish_grad(*args, **kwargs):
    """
    C++ interface function for hardswish_grad.
    """
def hardswish_grad_(*args, **kwargs):
    """
    C++ interface function for hardswish_grad_.
    """
def hardtanh(x: paddle.Tensor, t_min: float = 0, t_max: float = 24) -> paddle.Tensor:
    """
    C++ interface function for hardtanh.
    """
def hardtanh_(x: paddle.Tensor, t_min: float = 0, t_max: float = 24) -> paddle.Tensor:
    """
    C++ interface function for hardtanh_.
    """
def hardtanh_grad(*args, **kwargs):
    """
    C++ interface function for hardtanh_grad.
    """
def hardtanh_grad_(*args, **kwargs):
    """
    C++ interface function for hardtanh_grad_.
    """
def hash(*args, **kwargs):
    """
    C++ interface function for hash.
    """
def heaviside(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for heaviside.
    """
def heaviside_grad(*args, **kwargs):
    """
    C++ interface function for heaviside_grad.
    """
def hinge_loss(logits: paddle.Tensor, labels: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for hinge_loss.
    """
def hinge_loss_grad(*args, **kwargs):
    """
    C++ interface function for hinge_loss_grad.
    """
def histogram(input: paddle.Tensor, weight: paddle.Tensor, bins: int = 100, min: float = 0.0, max: float = 0.0, density: bool = False) -> paddle.Tensor:
    """
    C++ interface function for histogram.
    """
def hsigmoid_loss(x: paddle.Tensor, label: paddle.Tensor, w: paddle.Tensor, bias: paddle.Tensor, path: paddle.Tensor, code: paddle.Tensor, num_classes: int, is_sparse: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for hsigmoid_loss.
    """
def hsigmoid_loss_grad(*args, **kwargs):
    """
    C++ interface function for hsigmoid_loss_grad.
    """
def huber_loss(input: paddle.Tensor, label: paddle.Tensor, delta: float) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for huber_loss.
    """
def huber_loss_grad(*args, **kwargs):
    """
    C++ interface function for huber_loss_grad.
    """
def i0(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for i0.
    """
def i0_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for i0_.
    """
def i0_grad(*args, **kwargs):
    """
    C++ interface function for i0_grad.
    """
def i0e(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for i0e.
    """
def i0e_grad(*args, **kwargs):
    """
    C++ interface function for i0e_grad.
    """
def i1(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for i1.
    """
def i1_grad(*args, **kwargs):
    """
    C++ interface function for i1_grad.
    """
def i1e(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for i1e.
    """
def i1e_grad(*args, **kwargs):
    """
    C++ interface function for i1e_grad.
    """
def identity_loss(x: paddle.Tensor, reduction: int = 1) -> paddle.Tensor:
    """
    C++ interface function for identity_loss.
    """
def identity_loss_(x: paddle.Tensor, reduction: int = 1) -> paddle.Tensor:
    """
    C++ interface function for identity_loss_.
    """
def identity_loss_grad(*args, **kwargs):
    """
    C++ interface function for identity_loss_grad.
    """
def identity_loss_grad_(*args, **kwargs):
    """
    C++ interface function for identity_loss_grad_.
    """
def im2sequence(x: paddle.Tensor, y: paddle.Tensor, kernels: list[int], strides: list[int] = [1, 1], paddings: list[int] = [0, 0, 0, 0], out_stride: list[int] = [1, 1]) -> paddle.Tensor:
    """
    C++ interface function for im2sequence.
    """
def im2sequence_grad(*args, **kwargs):
    """
    C++ interface function for im2sequence_grad.
    """
def imag(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for imag.
    """
def imag_grad(*args, **kwargs):
    """
    C++ interface function for imag_grad.
    """
def increment(x: paddle.Tensor, value: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for increment.
    """
def increment_(x: paddle.Tensor, value: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for increment_.
    """
def index_add(x: paddle.Tensor, index: paddle.Tensor, add_value: paddle.Tensor, axis: int = 0) -> paddle.Tensor:
    """
    C++ interface function for index_add.
    """
def index_add_(x: paddle.Tensor, index: paddle.Tensor, add_value: paddle.Tensor, axis: int = 0) -> paddle.Tensor:
    """
    C++ interface function for index_add_.
    """
def index_add_grad(*args, **kwargs):
    """
    C++ interface function for index_add_grad.
    """
def index_add_grad_(*args, **kwargs):
    """
    C++ interface function for index_add_grad_.
    """
def index_put(x: paddle.Tensor, indices: list[paddle.Tensor], value: paddle.Tensor, accumulate: bool = False) -> paddle.Tensor:
    """
    C++ interface function for index_put.
    """
def index_put_(x: paddle.Tensor, indices: list[paddle.Tensor], value: paddle.Tensor, accumulate: bool = False) -> paddle.Tensor:
    """
    C++ interface function for index_put_.
    """
def index_put_grad(*args, **kwargs):
    """
    C++ interface function for index_put_grad.
    """
def index_sample(x: paddle.Tensor, index: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for index_sample.
    """
def index_sample_grad(*args, **kwargs):
    """
    C++ interface function for index_sample_grad.
    """
def index_select(x: paddle.Tensor, index: paddle.Tensor, axis: int = 0) -> paddle.Tensor:
    """
    C++ interface function for index_select.
    """
def index_select_grad(*args, **kwargs):
    """
    C++ interface function for index_select_grad.
    """
def index_select_strided(x: paddle.Tensor, index: int, axis: int = 0) -> paddle.Tensor:
    """
    C++ interface function for index_select_strided.
    """
def index_select_strided_grad(*args, **kwargs):
    """
    C++ interface function for index_select_strided_grad.
    """
def instance_norm(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, epsilon: float = 1e-5) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for instance_norm.
    """
def instance_norm_grad(*args, **kwargs):
    """
    C++ interface function for instance_norm_grad.
    """
def inverse(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for inverse.
    """
def inverse_grad(*args, **kwargs):
    """
    C++ interface function for inverse_grad.
    """
def is_empty(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for is_empty.
    """
def isclose(x: paddle.Tensor, y: paddle.Tensor, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> paddle.Tensor:
    """
    C++ interface function for isclose.
    """
def isfinite(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for isfinite.
    """
def isinf(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for isinf.
    """
def isnan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for isnan.
    """
def kldiv_loss(x: paddle.Tensor, label: paddle.Tensor, reduction: str = "mean", log_target: bool = False) -> paddle.Tensor:
    """
    C++ interface function for kldiv_loss.
    """
def kldiv_loss_grad(*args, **kwargs):
    """
    C++ interface function for kldiv_loss_grad.
    """
def kron(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for kron.
    """
def kron_grad(*args, **kwargs):
    """
    C++ interface function for kron_grad.
    """
def kthvalue(x: paddle.Tensor, k: int = 1, axis: int = -1, keepdim: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for kthvalue.
    """
def kthvalue_grad(*args, **kwargs):
    """
    C++ interface function for kthvalue_grad.
    """
def l1_norm(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for l1_norm.
    """
def l1_norm_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for l1_norm_.
    """
def l1_norm_grad(*args, **kwargs):
    """
    C++ interface function for l1_norm_grad.
    """
def label_smooth(label: paddle.Tensor, prior_dist: paddle.Tensor, epsilon: float = 0.0) -> paddle.Tensor:
    """
    C++ interface function for label_smooth.
    """
def label_smooth_grad(*args, **kwargs):
    """
    C++ interface function for label_smooth_grad.
    """
def lamb_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, moment1: paddle.Tensor, moment2: paddle.Tensor, beta1_pow: paddle.Tensor, beta2_pow: paddle.Tensor, master_param: paddle.Tensor, skip_update: paddle.Tensor, weight_decay: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-6, always_adapt: bool = False, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lamb_.
    """
def lars_momentum_(*args, **kwargs):
    """
    C++ interface function for lars_momentum_.
    """
def layer_norm(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, epsilon: float = 1e-5, begin_norm_axis: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for layer_norm.
    """
def layer_norm_grad(*args, **kwargs):
    """
    C++ interface function for layer_norm_grad.
    """
def leaky_relu(x: paddle.Tensor, negative_slope: float = 0.02) -> paddle.Tensor:
    """
    C++ interface function for leaky_relu.
    """
def leaky_relu_(x: paddle.Tensor, negative_slope: float = 0.02) -> paddle.Tensor:
    """
    C++ interface function for leaky_relu_.
    """
def leaky_relu_grad(*args, **kwargs):
    """
    C++ interface function for leaky_relu_grad.
    """
def leaky_relu_grad_(*args, **kwargs):
    """
    C++ interface function for leaky_relu_grad_.
    """
def lerp(x: paddle.Tensor, y: paddle.Tensor, weight: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for lerp.
    """
def lerp_(x: paddle.Tensor, y: paddle.Tensor, weight: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for lerp_.
    """
def lerp_grad(*args, **kwargs):
    """
    C++ interface function for lerp_grad.
    """
def less_equal(*args, **kwargs):
    """
    C++ interface function for less_equal.
    """
def less_equal_(*args, **kwargs):
    """
    C++ interface function for less_equal_.
    """
def less_than(*args, **kwargs):
    """
    C++ interface function for less_than.
    """
def less_than_(*args, **kwargs):
    """
    C++ interface function for less_than_.
    """
def lgamma(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for lgamma.
    """
def lgamma_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for lgamma_.
    """
def lgamma_grad(*args, **kwargs):
    """
    C++ interface function for lgamma_grad.
    """
def limit_by_capacity(expert_count: paddle.Tensor, capacity: paddle.Tensor, n_worker: int) -> paddle.Tensor:
    """
    C++ interface function for limit_by_capacity.
    """
def linear_interp(x: paddle.Tensor, out_size: paddle.Tensor, size_tensor: list[paddle.Tensor], scale_tensor: paddle.Tensor, data_format: str = "NCHW", out_d: int = 0, out_h: int = 0, out_w: int = 0, scale: list[float] = [], interp_method: str = "bilinear", align_corners: bool = True, align_mode: int = 1) -> paddle.Tensor:
    """
    C++ interface function for linear_interp.
    """
def linear_interp_grad(*args, **kwargs):
    """
    C++ interface function for linear_interp_grad.
    """
def linspace(start: paddle.Tensor, stop: paddle.Tensor, number: paddle.Tensor, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike) -> paddle.Tensor:
    """
    C++ interface function for linspace.
    """
def llm_int8_linear(x: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, weight_scale: paddle.Tensor, threshold: float = 6.0) -> paddle.Tensor:
    """
    C++ interface function for llm_int8_linear.
    """
def lod_reset_grad_(*args, **kwargs):
    """
    C++ interface function for lod_reset_grad_.
    """
def log(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log.
    """
def log10(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log10.
    """
def log10_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log10_.
    """
def log10_grad(*args, **kwargs):
    """
    C++ interface function for log10_grad.
    """
def log10_grad_(*args, **kwargs):
    """
    C++ interface function for log10_grad_.
    """
def log1p(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log1p.
    """
def log1p_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log1p_.
    """
def log1p_grad(*args, **kwargs):
    """
    C++ interface function for log1p_grad.
    """
def log1p_grad_(*args, **kwargs):
    """
    C++ interface function for log1p_grad_.
    """
def log2(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log2.
    """
def log2_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log2_.
    """
def log2_grad(*args, **kwargs):
    """
    C++ interface function for log2_grad.
    """
def log2_grad_(*args, **kwargs):
    """
    C++ interface function for log2_grad_.
    """
def log_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log_.
    """
def log_grad(*args, **kwargs):
    """
    C++ interface function for log_grad.
    """
def log_grad_(*args, **kwargs):
    """
    C++ interface function for log_grad_.
    """
def log_loss(input: paddle.Tensor, label: paddle.Tensor, epsilon: float) -> paddle.Tensor:
    """
    C++ interface function for log_loss.
    """
def log_loss_grad(*args, **kwargs):
    """
    C++ interface function for log_loss_grad.
    """
def log_softmax(x: paddle.Tensor, axis: int = -1) -> paddle.Tensor:
    """
    C++ interface function for log_softmax.
    """
def log_softmax_grad(*args, **kwargs):
    """
    C++ interface function for log_softmax_grad.
    """
def logcumsumexp(x: paddle.Tensor, axis: int = -1, flatten: bool = False, exclusive: bool = False, reverse: bool = False) -> paddle.Tensor:
    """
    C++ interface function for logcumsumexp.
    """
def logcumsumexp_grad(*args, **kwargs):
    """
    C++ interface function for logcumsumexp_grad.
    """
def logical_and(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_and.
    """
def logical_and_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_and_.
    """
def logical_not(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_not.
    """
def logical_not_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_not_.
    """
def logical_or(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_or.
    """
def logical_or_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_or_.
    """
def logical_xor(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_xor.
    """
def logical_xor_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_xor_.
    """
def logit(x: paddle.Tensor, eps: float = 1e-6) -> paddle.Tensor:
    """
    C++ interface function for logit.
    """
def logit_(x: paddle.Tensor, eps: float = 1e-6) -> paddle.Tensor:
    """
    C++ interface function for logit_.
    """
def logit_grad(*args, **kwargs):
    """
    C++ interface function for logit_grad.
    """
def logsigmoid(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logsigmoid.
    """
def logsigmoid_grad(*args, **kwargs):
    """
    C++ interface function for logsigmoid_grad.
    """
def logsigmoid_grad_(*args, **kwargs):
    """
    C++ interface function for logsigmoid_grad_.
    """
def logspace(start: paddle.Tensor, stop: paddle.Tensor, num: paddle.Tensor, base: paddle.Tensor, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for logspace.
    """
def logsumexp(x: paddle.Tensor, axis: list[int] = [0], keepdim: bool = False, reduce_all: bool = False) -> paddle.Tensor:
    """
    C++ interface function for logsumexp.
    """
def logsumexp_grad(*args, **kwargs):
    """
    C++ interface function for logsumexp_grad.
    """
def lookup_table_dequant(w: paddle.Tensor, ids: paddle.Tensor, padding_idx: int = -1) -> paddle.Tensor:
    """
    C++ interface function for lookup_table_dequant.
    """
def lp_pool2d(x: paddle.Tensor, kernel_size: list[int], strides: list[int] = [1,1], paddings: list[int] = [0,0], ceil_mode: bool = False, exclusive: bool = True, data_format: str = "NCHW", pooling_type: str = "", global_pooling: bool = False, adaptive: bool = False, padding_algorithm: str = "EXPLICIT", norm_type: float = 0.0) -> paddle.Tensor:
    """
    C++ interface function for lp_pool2d.
    """
def lp_pool2d_grad(*args, **kwargs):
    """
    C++ interface function for lp_pool2d_grad.
    """
def lstm(input: paddle.Tensor, h0: paddle.Tensor, c0: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, use_peepholes: bool = True, is_reverse: bool = False, is_test: bool = False, gate_activation: str = "sigmoid", cell_activation: str = "tanh", candidate_activation: str = "tanh") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lstm.
    """
def lstm_grad(*args, **kwargs):
    """
    C++ interface function for lstm_grad.
    """
def lstsq(x: paddle.Tensor, y: paddle.Tensor, rcond: float = 0.0, driver: str = "gels") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lstsq.
    """
def lu(x: paddle.Tensor, pivot: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lu.
    """
def lu_(x: paddle.Tensor, pivot: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lu_.
    """
def lu_grad(*args, **kwargs):
    """
    C++ interface function for lu_grad.
    """
def lu_grad_(*args, **kwargs):
    """
    C++ interface function for lu_grad_.
    """
def lu_solve(b: paddle.Tensor, lu: paddle.Tensor, pivots: paddle.Tensor, trans: str) -> paddle.Tensor:
    """
    C++ interface function for lu_solve.
    """
def lu_solve_grad(*args, **kwargs):
    """
    C++ interface function for lu_solve_grad.
    """
def lu_unpack(x: paddle.Tensor, y: paddle.Tensor, unpack_ludata: bool = True, unpack_pivots: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lu_unpack.
    """
def lu_unpack_grad(*args, **kwargs):
    """
    C++ interface function for lu_unpack_grad.
    """
def margin_cross_entropy(logits: paddle.Tensor, label: paddle.Tensor, return_softmax: bool = False, ring_id: int = 0, rank: int = 0, nranks: int = 1, margin1: float = 1.0, margin2: float = 0.5, margin3: float = 0.0, scale: float = 64.0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for margin_cross_entropy.
    """
def margin_cross_entropy_grad(*args, **kwargs):
    """
    C++ interface function for margin_cross_entropy_grad.
    """
def margin_cross_entropy_grad_(*args, **kwargs):
    """
    C++ interface function for margin_cross_entropy_grad_.
    """
def masked_multihead_attention_(x: paddle.Tensor, cache_kv: paddle.Tensor, bias: paddle.Tensor, src_mask: paddle.Tensor, cum_offsets: paddle.Tensor, sequence_lengths: paddle.Tensor, rotary_tensor: paddle.Tensor, beam_cache_offset: paddle.Tensor, qkv_out_scale: paddle.Tensor, out_shift: paddle.Tensor, out_smooth: paddle.Tensor, seq_len: int, rotary_emb_dims: int, use_neox_rotary_style: bool = False, compute_dtype: str = "default", out_scale: float = -1, quant_round_type: int = 1, quant_max_bound: float = 127.0, quant_min_bound: float = -127.0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for masked_multihead_attention_.
    """
def masked_select(x: paddle.Tensor, mask: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for masked_select.
    """
def masked_select_grad(*args, **kwargs):
    """
    C++ interface function for masked_select_grad.
    """
def matmul(*args, **kwargs):
    """
    C++ interface function for matmul.
    """
def matmul_grad(*args, **kwargs):
    """
    C++ interface function for matmul_grad.
    """
def matmul_with_flatten(*args, **kwargs):
    """
    C++ interface function for matmul_with_flatten.
    """
def matmul_with_flatten_grad(*args, **kwargs):
    """
    C++ interface function for matmul_with_flatten_grad.
    """
def matrix_nms(bboxes: paddle.Tensor, scores: paddle.Tensor, score_threshold: float, nms_top_k: int, keep_top_k: int, post_threshold: float = 0., use_gaussian: bool = False, gaussian_sigma: float = 2., background_label: int = 0, normalized: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for matrix_nms.
    """
def matrix_power(x: paddle.Tensor, n: int) -> paddle.Tensor:
    """
    C++ interface function for matrix_power.
    """
def matrix_power_grad(*args, **kwargs):
    """
    C++ interface function for matrix_power_grad.
    """
def matrix_rank(x: paddle.Tensor, tol: float, use_default_tol: bool = True, hermitian: bool = False) -> paddle.Tensor:
    """
    C++ interface function for matrix_rank.
    """
def matrix_rank_atol_rtol(x: paddle.Tensor, atol: paddle.Tensor, rtol: paddle.Tensor, hermitian: bool = False) -> paddle.Tensor:
    """
    C++ interface function for matrix_rank_atol_rtol.
    """
def matrix_rank_tol(x: paddle.Tensor, atol_tensor: paddle.Tensor, use_default_tol: bool = True, hermitian: bool = False) -> paddle.Tensor:
    """
    C++ interface function for matrix_rank_tol.
    """
def max(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for max.
    """
def max_grad(*args, **kwargs):
    """
    C++ interface function for max_grad.
    """
def max_pool2d_with_index(x: paddle.Tensor, kernel_size: list[int], strides: list[int] = [1, 1], paddings: list[int] = [0, 0], global_pooling: bool = False, adaptive: bool = False, ceil_mode: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for max_pool2d_with_index.
    """
def max_pool2d_with_index_grad(*args, **kwargs):
    """
    C++ interface function for max_pool2d_with_index_grad.
    """
def max_pool3d_with_index(x: paddle.Tensor, kernel_size: list[int], strides: list[int] = [1, 1, 1], paddings: list[int] = [0, 0, 0], global_pooling: bool = False, adaptive: bool = False, ceil_mode: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for max_pool3d_with_index.
    """
def max_pool3d_with_index_grad(*args, **kwargs):
    """
    C++ interface function for max_pool3d_with_index_grad.
    """
def maximum(*args, **kwargs):
    """
    C++ interface function for maximum.
    """
def maximum_grad(*args, **kwargs):
    """
    C++ interface function for maximum_grad.
    """
def maxout(x: paddle.Tensor, groups: int, axis: int = 1) -> paddle.Tensor:
    """
    C++ interface function for maxout.
    """
def maxout_grad(*args, **kwargs):
    """
    C++ interface function for maxout_grad.
    """
def mean(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for mean.
    """
def mean_all(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for mean_all.
    """
def mean_all_grad(*args, **kwargs):
    """
    C++ interface function for mean_all_grad.
    """
def mean_grad(*args, **kwargs):
    """
    C++ interface function for mean_grad.
    """
def memcpy(*args, **kwargs):
    """
    C++ interface function for memcpy.
    """
def memcpy_d2h(x: paddle.Tensor, dst_place_type: int) -> paddle.Tensor:
    """
    C++ interface function for memcpy_d2h.
    """
def memcpy_h2d(x: paddle.Tensor, dst_place_type: int) -> paddle.Tensor:
    """
    C++ interface function for memcpy_h2d.
    """
def memory_efficient_attention(query: paddle.Tensor, key: paddle.Tensor, value: paddle.Tensor, bias: paddle.Tensor, cu_seqlens_q: paddle.Tensor, cu_seqlens_k: paddle.Tensor, causal_diagonal: paddle.Tensor, seqlen_k: paddle.Tensor, max_seqlen_q: float, max_seqlen_k: float, causal: bool, dropout_p: float, scale: float, is_test: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for memory_efficient_attention.
    """
def memory_efficient_attention_grad(*args, **kwargs):
    """
    C++ interface function for memory_efficient_attention_grad.
    """
def merge_selected_rows(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for merge_selected_rows.
    """
def merged_adam_(param: list[paddle.Tensor], grad: list[paddle.Tensor], learning_rate: list[paddle.Tensor], moment1: list[paddle.Tensor], moment2: list[paddle.Tensor], moment2_max: list[paddle.Tensor], beta1_pow: list[paddle.Tensor], beta2_pow: list[paddle.Tensor], master_param: list[paddle.Tensor], beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, multi_precision: bool = False, use_global_beta_pow: bool = False, amsgrad: bool = False) -> tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]:
    """
    C++ interface function for merged_adam_.
    """
def merged_momentum_(param: list[paddle.Tensor], grad: list[paddle.Tensor], velocity: list[paddle.Tensor], learning_rate: list[paddle.Tensor], master_param: list[paddle.Tensor], mu: float, use_nesterov: bool = False, regularization_method: list[str] = [], regularization_coeff: list[float] = [], multi_precision: bool = False, rescale_grad: float = 1.0) -> tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]:
    """
    C++ interface function for merged_momentum_.
    """
def meshgrid(inputs: list[paddle.Tensor]) -> list[paddle.Tensor]:
    """
    C++ interface function for meshgrid.
    """
def meshgrid_grad(*args, **kwargs):
    """
    C++ interface function for meshgrid_grad.
    """
def min(*args, **kwargs):
    """
    C++ interface function for min.
    """
def min_grad(*args, **kwargs):
    """
    C++ interface function for min_grad.
    """
def minimum(*args, **kwargs):
    """
    C++ interface function for minimum.
    """
def minimum_grad(*args, **kwargs):
    """
    C++ interface function for minimum_grad.
    """
def mish(x: paddle.Tensor, lambda_: float) -> paddle.Tensor:
    """
    C++ interface function for mish.
    """
def mish_grad(*args, **kwargs):
    """
    C++ interface function for mish_grad.
    """
def mish_grad_(*args, **kwargs):
    """
    C++ interface function for mish_grad_.
    """
def mode(x: paddle.Tensor, axis: int = -1, keepdim: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for mode.
    """
def mode_grad(*args, **kwargs):
    """
    C++ interface function for mode_grad.
    """
def moe_dispatch(X: paddle.Tensor, gating_output: paddle.Tensor, moe_topk: int = 2, group_moe: bool = False, topk_only_mode: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for moe_dispatch.
    """
def moe_ffn(permute_input: paddle.Tensor, token_nums_per_expert: paddle.Tensor, ffn1_weight: paddle.Tensor, ffn2_weight: paddle.Tensor, ffn1_bias: paddle.Tensor, ffn1_scale: paddle.Tensor, ffn2_scale: paddle.Tensor, quant_method: str) -> paddle.Tensor:
    """
    C++ interface function for moe_ffn.
    """
def moe_reduce(ffn_out: paddle.Tensor, expert_scales_float: paddle.Tensor, permute_indices_per_token: paddle.Tensor, top_k_indices: paddle.Tensor, ffn2_bias: paddle.Tensor, norm_topk_prob: bool = False, routed_scaling_factor: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for moe_reduce.
    """
def momentum_(param: paddle.Tensor, grad: paddle.Tensor, velocity: paddle.Tensor, learning_rate: paddle.Tensor, master_param: paddle.Tensor, mu: float, use_nesterov: bool = False, regularization_method: str = "", regularization_coeff: float = 0.0, multi_precision: bool = False, rescale_grad: float = 1.0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for momentum_.
    """
def moving_average_abs_max_scale(*args, **kwargs):
    """
    C++ interface function for moving_average_abs_max_scale.
    """
def moving_average_abs_max_scale_(*args, **kwargs):
    """
    C++ interface function for moving_average_abs_max_scale_.
    """
def mp_allreduce_sum(x: paddle.Tensor, ring_id: int = 0) -> paddle.Tensor:
    """
    C++ interface function for mp_allreduce_sum.
    """
def mp_allreduce_sum_(x: paddle.Tensor, ring_id: int = 0) -> paddle.Tensor:
    """
    C++ interface function for mp_allreduce_sum_.
    """
def multi_dot(x: list[paddle.Tensor]) -> paddle.Tensor:
    """
    C++ interface function for multi_dot.
    """
def multi_dot_grad(*args, **kwargs):
    """
    C++ interface function for multi_dot_grad.
    """
def multiclass_nms3(bboxes: paddle.Tensor, scores: paddle.Tensor, rois_num: paddle.Tensor, score_threshold: float, nms_top_k: int, keep_top_k: int, nms_threshold: float = 0.3, normalized: bool = True, nms_eta: float = 1.0, background_label: int = 0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for multiclass_nms3.
    """
def multihead_matmul(*args, **kwargs):
    """
    C++ interface function for multihead_matmul.
    """
def multinomial(x: paddle.Tensor, num_samples: int = 1, replacement: bool = False) -> paddle.Tensor:
    """
    C++ interface function for multinomial.
    """
def multiplex(inputs: list[paddle.Tensor], index: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for multiplex.
    """
def multiplex_grad(*args, **kwargs):
    """
    C++ interface function for multiplex_grad.
    """
def multiply(*args, **kwargs):
    """
    C++ interface function for multiply.
    """
def multiply_(*args, **kwargs):
    """
    C++ interface function for multiply_.
    """
def mv(x: paddle.Tensor, vec: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for mv.
    """
def mv_grad(*args, **kwargs):
    """
    C++ interface function for mv_grad.
    """
def nadam_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, momentum_decay_pow: paddle.Tensor, beta2_pow: paddle.Tensor, mu_product: paddle.Tensor, moment1: paddle.Tensor, moment2: paddle.Tensor, master_param: paddle.Tensor, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, momentum_decay: float = 0.004, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for nadam_.
    """
def nanmedian(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = True, mode: str = "avg") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for nanmedian.
    """
def nanmedian_grad(*args, **kwargs):
    """
    C++ interface function for nanmedian_grad.
    """
def nearest_interp(x: paddle.Tensor, out_size: paddle.Tensor, size_tensor: list[paddle.Tensor], scale_tensor: paddle.Tensor, data_format: str = "NCHW", out_d: int = 0, out_h: int = 0, out_w: int = 0, scale: list[float] = [], interp_method: str = "bilinear", align_corners: bool = True, align_mode: int = 1) -> paddle.Tensor:
    """
    C++ interface function for nearest_interp.
    """
def nearest_interp_grad(*args, **kwargs):
    """
    C++ interface function for nearest_interp_grad.
    """
def nextafter(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for nextafter.
    """
def nll_loss(input: paddle.Tensor, label: paddle.Tensor, weight: paddle.Tensor, ignore_index: int = -100, reduction: str = "mean") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for nll_loss.
    """
def nll_loss_grad(*args, **kwargs):
    """
    C++ interface function for nll_loss_grad.
    """
def nms(x: paddle.Tensor, threshold: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for nms.
    """
def nonzero(condition: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for nonzero.
    """
def nop(*args, **kwargs):
    """
    C++ interface function for nop.
    """
def nop_(*args, **kwargs):
    """
    C++ interface function for nop_.
    """
def norm(x: paddle.Tensor, axis: int, epsilon: float, is_test: bool) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for norm.
    """
def norm_grad(*args, **kwargs):
    """
    C++ interface function for norm_grad.
    """
def not_equal(*args, **kwargs):
    """
    C++ interface function for not_equal.
    """
def not_equal_(*args, **kwargs):
    """
    C++ interface function for not_equal_.
    """
def npu_identity(x: paddle.Tensor, format: int = -1) -> paddle.Tensor:
    """
    C++ interface function for npu_identity.
    """
def number_count(numbers: paddle.Tensor, upper_range: int) -> paddle.Tensor:
    """
    C++ interface function for number_count.
    """
def numel(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for numel.
    """
def one_hot(x: paddle.Tensor, num_classes: int) -> paddle.Tensor:
    """
    C++ interface function for one_hot.
    """
def onednn_to_paddle_layout(*args, **kwargs):
    """
    C++ interface function for onednn_to_paddle_layout.
    """
def overlap_add(x: paddle.Tensor, hop_length: int, axis: int = -1) -> paddle.Tensor:
    """
    C++ interface function for overlap_add.
    """
def overlap_add_grad(*args, **kwargs):
    """
    C++ interface function for overlap_add_grad.
    """
def p_norm(x: paddle.Tensor, porder: float = 2, axis: int = -1, epsilon: float = 1.0e-12, keepdim: bool = False, asvector: bool = False) -> paddle.Tensor:
    """
    C++ interface function for p_norm.
    """
def p_norm_grad(*args, **kwargs):
    """
    C++ interface function for p_norm_grad.
    """
def pad(x: paddle.Tensor, paddings: list[int], pad_value: float) -> paddle.Tensor:
    """
    C++ interface function for pad.
    """
def pad3d(x: paddle.Tensor, paddings: list[int], mode: str = "constant", pad_value: float = 0.0, data_format: str = "NCDHW") -> paddle.Tensor:
    """
    C++ interface function for pad3d.
    """
def pad3d_grad(*args, **kwargs):
    """
    C++ interface function for pad3d_grad.
    """
def pad_grad(*args, **kwargs):
    """
    C++ interface function for pad_grad.
    """
def parameter(*args, **kwargs):
    """
    C++ interface function for parameter.
    """
def pixel_shuffle(x: paddle.Tensor, upscale_factor: int = 1, data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for pixel_shuffle.
    """
def pixel_shuffle_grad(*args, **kwargs):
    """
    C++ interface function for pixel_shuffle_grad.
    """
def pixel_unshuffle(x: paddle.Tensor, downscale_factor: int = 1, data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for pixel_unshuffle.
    """
def pixel_unshuffle_grad(*args, **kwargs):
    """
    C++ interface function for pixel_unshuffle_grad.
    """
def poisson(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for poisson.
    """
def poisson_grad(*args, **kwargs):
    """
    C++ interface function for poisson_grad.
    """
def polygamma(x: paddle.Tensor, n: int) -> paddle.Tensor:
    """
    C++ interface function for polygamma.
    """
def polygamma_(x: paddle.Tensor, n: int) -> paddle.Tensor:
    """
    C++ interface function for polygamma_.
    """
def polygamma_grad(*args, **kwargs):
    """
    C++ interface function for polygamma_grad.
    """
def pool2d(x: paddle.Tensor, kernel_size: list[int], strides: list[int], paddings: list[int], ceil_mode: bool, exclusive: bool, data_format: str, pooling_type: str, global_pooling: bool, adaptive: bool, padding_algorithm: str) -> paddle.Tensor:
    """
    C++ interface function for pool2d.
    """
def pool2d_grad(*args, **kwargs):
    """
    C++ interface function for pool2d_grad.
    """
def pool3d(x: paddle.Tensor, kernel_size: list[int], strides: list[int], paddings: list[int], ceil_mode: bool, exclusive: bool, data_format: str, pooling_type: str, global_pooling: bool, adaptive: bool, padding_algorithm: str) -> paddle.Tensor:
    """
    C++ interface function for pool3d.
    """
def pool3d_grad(*args, **kwargs):
    """
    C++ interface function for pool3d_grad.
    """
def pow(x: paddle.Tensor, y: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for pow.
    """
def pow_(x: paddle.Tensor, y: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for pow_.
    """
def pow_grad(*args, **kwargs):
    """
    C++ interface function for pow_grad.
    """
def pow_grad_(*args, **kwargs):
    """
    C++ interface function for pow_grad_.
    """
def prelu(x: paddle.Tensor, alpha: paddle.Tensor, data_format: str = "NCHW", mode: str = "all") -> paddle.Tensor:
    """
    C++ interface function for prelu.
    """
def prelu_grad(*args, **kwargs):
    """
    C++ interface function for prelu_grad.
    """
def print(*args, **kwargs):
    """
    C++ interface function for print.
    """
def prior_box(input: paddle.Tensor, image: paddle.Tensor, min_sizes: list[float], max_sizes: list[float] = [], aspect_ratios: list[float] = [], variances: list[float] = [], flip: bool = True, clip: bool = True, step_w: float = 0.0, step_h: float = 0.0, offset: float = 0.5, min_max_aspect_ratios_order: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for prior_box.
    """
def prod(x: paddle.Tensor, axis: list[int], keepdim: bool, reduce_all: bool) -> paddle.Tensor:
    """
    C++ interface function for prod.
    """
def prod_grad(*args, **kwargs):
    """
    C++ interface function for prod_grad.
    """
def prune_gate_by_capacity(gate_idx: paddle.Tensor, expert_count: paddle.Tensor, n_expert: int = 0, n_worker: int = 0) -> paddle.Tensor:
    """
    C++ interface function for prune_gate_by_capacity.
    """
def psroi_pool(x: paddle.Tensor, boxes: paddle.Tensor, boxes_num: paddle.Tensor, pooled_height: int = 1, pooled_width: int = 1, output_channels: int = 1, spatial_scale: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for psroi_pool.
    """
def psroi_pool_grad(*args, **kwargs):
    """
    C++ interface function for psroi_pool_grad.
    """
def put_along_axis(arr: paddle.Tensor, indices: paddle.Tensor, values: paddle.Tensor, axis: int, reduce: str = "assign", include_self: bool = True) -> paddle.Tensor:
    """
    C++ interface function for put_along_axis.
    """
def put_along_axis_(arr: paddle.Tensor, indices: paddle.Tensor, values: paddle.Tensor, axis: int, reduce: str = "assign", include_self: bool = True) -> paddle.Tensor:
    """
    C++ interface function for put_along_axis_.
    """
def put_along_axis_grad(*args, **kwargs):
    """
    C++ interface function for put_along_axis_grad.
    """
def pyramid_hash(x: paddle.Tensor, w: paddle.Tensor, white_list: paddle.Tensor, black_list: paddle.Tensor, num_emb: int = 0, space_len: int = 0, pyramid_layer: int = 2, rand_len: int = 0, drop_out_percent: float = 0, is_training: int = 0, use_filter: bool = True, white_list_len: int = 0, black_list_len: int = 0, seed: int = 0, lr: float = 0.0, distribute_update_vars: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for pyramid_hash.
    """
def pyramid_hash_grad(*args, **kwargs):
    """
    C++ interface function for pyramid_hash_grad.
    """
def qkv_unpack_mha(*args, **kwargs):
    """
    C++ interface function for qkv_unpack_mha.
    """
def qr(x: paddle.Tensor, mode: str = "reduced") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for qr.
    """
def qr_grad(*args, **kwargs):
    """
    C++ interface function for qr_grad.
    """
def quantize_linear(*args, **kwargs):
    """
    C++ interface function for quantize_linear.
    """
def quantize_linear_(*args, **kwargs):
    """
    C++ interface function for quantize_linear_.
    """
def radam_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, beta1_pow: paddle.Tensor, beta2_pow: paddle.Tensor, rho: paddle.Tensor, moment1: paddle.Tensor, moment2: paddle.Tensor, master_param: paddle.Tensor, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for radam_.
    """
def randint(low: int, high: int, shape: list[int], dtype: paddle._typing.DTypeLike = "DataType::INT64", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for randint.
    """
def random_routing_(prob: paddle.Tensor, topk_value: paddle.Tensor, topk_idx: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for random_routing_.
    """
def randperm(n: int, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for randperm.
    """
def rank_attention(x: paddle.Tensor, rank_offset: paddle.Tensor, rank_param: paddle.Tensor, max_rank: int = 3, max_size: int = 0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for rank_attention.
    """
def rank_attention_grad(*args, **kwargs):
    """
    C++ interface function for rank_attention_grad.
    """
def read_file(filename: str = "", dtype: paddle._typing.DTypeLike = "DataType::UINT8", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for read_file.
    """
def real(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for real.
    """
def real_grad(*args, **kwargs):
    """
    C++ interface function for real_grad.
    """
def reciprocal(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for reciprocal.
    """
def reciprocal_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for reciprocal_.
    """
def reciprocal_grad(*args, **kwargs):
    """
    C++ interface function for reciprocal_grad.
    """
def reciprocal_grad_(*args, **kwargs):
    """
    C++ interface function for reciprocal_grad_.
    """
def recv_v2(*args, **kwargs):
    """
    C++ interface function for recv_v2.
    """
def reduce(x: paddle.Tensor, ring_id: int = 0, root_id: int = 0, reduce_type: int = 0) -> paddle.Tensor:
    """
    C++ interface function for reduce.
    """
def reduce_(x: paddle.Tensor, ring_id: int = 0, root_id: int = 0, reduce_type: int = 0) -> paddle.Tensor:
    """
    C++ interface function for reduce_.
    """
def reduce_as(x: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for reduce_as.
    """
def reduce_as_grad(*args, **kwargs):
    """
    C++ interface function for reduce_as_grad.
    """
def reduce_scatter(x: paddle.Tensor, ring_id: int = 0, nranks: int = 1) -> paddle.Tensor:
    """
    C++ interface function for reduce_scatter.
    """
def reindex_graph(x: paddle.Tensor, neighbors: paddle.Tensor, count: paddle.Tensor, hashtable_value: paddle.Tensor, hashtable_index: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for reindex_graph.
    """
def relu(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for relu.
    """
def relu6(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for relu6.
    """
def relu6_grad(*args, **kwargs):
    """
    C++ interface function for relu6_grad.
    """
def relu6_grad_(*args, **kwargs):
    """
    C++ interface function for relu6_grad_.
    """
def relu_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for relu_.
    """
def relu_grad(*args, **kwargs):
    """
    C++ interface function for relu_grad.
    """
def relu_grad_(*args, **kwargs):
    """
    C++ interface function for relu_grad_.
    """
def remainder(*args, **kwargs):
    """
    C++ interface function for remainder.
    """
def remainder_(*args, **kwargs):
    """
    C++ interface function for remainder_.
    """
def remainder_grad(*args, **kwargs):
    """
    C++ interface function for remainder_grad.
    """
def renorm(x: paddle.Tensor, p: float, axis: int, max_norm: float) -> paddle.Tensor:
    """
    C++ interface function for renorm.
    """
def renorm_(x: paddle.Tensor, p: float, axis: int, max_norm: float) -> paddle.Tensor:
    """
    C++ interface function for renorm_.
    """
def renorm_grad(*args, **kwargs):
    """
    C++ interface function for renorm_grad.
    """
def repeat_interleave(x: paddle.Tensor, repeats: int, axis: int) -> paddle.Tensor:
    """
    C++ interface function for repeat_interleave.
    """
def repeat_interleave_grad(*args, **kwargs):
    """
    C++ interface function for repeat_interleave_grad.
    """
def repeat_interleave_with_tensor_index(x: paddle.Tensor, repeats: paddle.Tensor, axis: int) -> paddle.Tensor:
    """
    C++ interface function for repeat_interleave_with_tensor_index.
    """
def repeat_interleave_with_tensor_index_grad(*args, **kwargs):
    """
    C++ interface function for repeat_interleave_with_tensor_index_grad.
    """
def reshape(x: paddle.Tensor, shape: list[int]) -> paddle.Tensor:
    """
    C++ interface function for reshape.
    """
def reshape_(x: paddle.Tensor, shape: list[int]) -> paddle.Tensor:
    """
    C++ interface function for reshape_.
    """
def reshape_grad(*args, **kwargs):
    """
    C++ interface function for reshape_grad.
    """
def reshape_grad_(*args, **kwargs):
    """
    C++ interface function for reshape_grad_.
    """
def resnet_basic_block(*args, **kwargs):
    """
    C++ interface function for resnet_basic_block.
    """
def resnet_basic_block_grad(*args, **kwargs):
    """
    C++ interface function for resnet_basic_block_grad.
    """
def resnet_unit(*args, **kwargs):
    """
    C++ interface function for resnet_unit.
    """
def resnet_unit_grad(*args, **kwargs):
    """
    C++ interface function for resnet_unit_grad.
    """
def reverse(x: paddle.Tensor, axis: list[int]) -> paddle.Tensor:
    """
    C++ interface function for reverse.
    """
def rms_norm(x: paddle.Tensor, bias: paddle.Tensor, residual: paddle.Tensor, norm_weight: paddle.Tensor, norm_bias: paddle.Tensor, epsilon: float, begin_norm_axis: int, quant_scale: float, quant_round_type: int, quant_max_bound: float, quant_min_bound: float) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for rms_norm.
    """
def rms_norm_grad(*args, **kwargs):
    """
    C++ interface function for rms_norm_grad.
    """
def rmsprop_(param: paddle.Tensor, mean_square: paddle.Tensor, grad: paddle.Tensor, moment: paddle.Tensor, learning_rate: paddle.Tensor, mean_grad: paddle.Tensor, master_param: paddle.Tensor, epsilon: float = 1.0e-10, decay: float = 0.9, momentum: float = 0.0, centered: bool = False, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for rmsprop_.
    """
def rnn(x: paddle.Tensor, pre_state: list[paddle.Tensor], weight_list: list[paddle.Tensor], sequence_length: paddle.Tensor, dropout_state_in: paddle.Tensor, dropout_prob: float = 0.0, is_bidirec: bool = False, input_size: int = 10, hidden_size: int = 100, num_layers: int = 1, mode: str = "RNN_TANH", seed: int = 0, is_test: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, list[paddle.Tensor], paddle.Tensor]:
    """
    C++ interface function for rnn.
    """
def rnn_grad(*args, **kwargs):
    """
    C++ interface function for rnn_grad.
    """
def roi_align(x: paddle.Tensor, boxes: paddle.Tensor, boxes_num: paddle.Tensor, pooled_height: int = 1, pooled_width: int = 1, spatial_scale: float = 1.0, sampling_ratio: int = -1, aligned: bool = False) -> paddle.Tensor:
    """
    C++ interface function for roi_align.
    """
def roi_align_grad(*args, **kwargs):
    """
    C++ interface function for roi_align_grad.
    """
def roi_pool(x: paddle.Tensor, boxes: paddle.Tensor, boxes_num: paddle.Tensor, pooled_height: int = 1, pooled_width: int = 1, spatial_scale: float = 1.0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for roi_pool.
    """
def roi_pool_grad(*args, **kwargs):
    """
    C++ interface function for roi_pool_grad.
    """
def roll(x: paddle.Tensor, shifts: list[int] = [], axis: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for roll.
    """
def roll_grad(*args, **kwargs):
    """
    C++ interface function for roll_grad.
    """
def round(x: paddle.Tensor, decimals: int = 0) -> paddle.Tensor:
    """
    C++ interface function for round.
    """
def round_(x: paddle.Tensor, decimals: int = 0) -> paddle.Tensor:
    """
    C++ interface function for round_.
    """
def round_grad(*args, **kwargs):
    """
    C++ interface function for round_grad.
    """
def round_grad_(*args, **kwargs):
    """
    C++ interface function for round_grad_.
    """
def rprop_(param: paddle.Tensor, grad: paddle.Tensor, prev: paddle.Tensor, learning_rate: paddle.Tensor, master_param: paddle.Tensor, learning_rate_range: paddle.Tensor, etas: paddle.Tensor, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for rprop_.
    """
def rrelu(x: paddle.Tensor, lower: float = "1.0f/8", upper: float = "1.0f/3", is_test: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for rrelu.
    """
def rrelu_grad(*args, **kwargs):
    """
    C++ interface function for rrelu_grad.
    """
def rsqrt(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for rsqrt.
    """
def rsqrt_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for rsqrt_.
    """
def rsqrt_grad(*args, **kwargs):
    """
    C++ interface function for rsqrt_grad.
    """
def rsqrt_grad_(*args, **kwargs):
    """
    C++ interface function for rsqrt_grad_.
    """
def scale(x: paddle.Tensor, scale: float = 1.0, bias: float = 0.0, bias_after_scale: bool = True) -> paddle.Tensor:
    """
    C++ interface function for scale.
    """
def scale_(x: paddle.Tensor, scale: float = 1.0, bias: float = 0.0, bias_after_scale: bool = True) -> paddle.Tensor:
    """
    C++ interface function for scale_.
    """
def scatter(x: paddle.Tensor, index: paddle.Tensor, updates: paddle.Tensor, overwrite: bool = True) -> paddle.Tensor:
    """
    C++ interface function for scatter.
    """
def scatter_(x: paddle.Tensor, index: paddle.Tensor, updates: paddle.Tensor, overwrite: bool = True) -> paddle.Tensor:
    """
    C++ interface function for scatter_.
    """
def scatter_grad(*args, **kwargs):
    """
    C++ interface function for scatter_grad.
    """
def scatter_nd_add(x: paddle.Tensor, index: paddle.Tensor, updates: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for scatter_nd_add.
    """
def scatter_nd_add_grad(*args, **kwargs):
    """
    C++ interface function for scatter_nd_add_grad.
    """
def searchsorted(sorted_sequence: paddle.Tensor, values: paddle.Tensor, out_int32: bool = False, right: bool = False) -> paddle.Tensor:
    """
    C++ interface function for searchsorted.
    """
def segment_pool(x: paddle.Tensor, segment_ids: paddle.Tensor, pooltype: str = "SUM") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for segment_pool.
    """
def segment_pool_grad(*args, **kwargs):
    """
    C++ interface function for segment_pool_grad.
    """
def self_dp_attention(*args, **kwargs):
    """
    C++ interface function for self_dp_attention.
    """
def selu(x: paddle.Tensor, scale: float = 1.0507009873554804934193349852946, alpha: float = 1.6732632423543772848170429916717) -> paddle.Tensor:
    """
    C++ interface function for selu.
    """
def selu_grad(*args, **kwargs):
    """
    C++ interface function for selu_grad.
    """
def send_u_recv(x: paddle.Tensor, src_index: paddle.Tensor, dst_index: paddle.Tensor, reduce_op: str = "SUM", out_size: list[int] = [0]) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for send_u_recv.
    """
def send_u_recv_grad(*args, **kwargs):
    """
    C++ interface function for send_u_recv_grad.
    """
def send_ue_recv(x: paddle.Tensor, y: paddle.Tensor, src_index: paddle.Tensor, dst_index: paddle.Tensor, message_op: str = "ADD", reduce_op: str = "SUM", out_size: list[int] = [0]) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for send_ue_recv.
    """
def send_ue_recv_grad(*args, **kwargs):
    """
    C++ interface function for send_ue_recv_grad.
    """
def send_uv(x: paddle.Tensor, y: paddle.Tensor, src_index: paddle.Tensor, dst_index: paddle.Tensor, message_op: str = "ADD") -> paddle.Tensor:
    """
    C++ interface function for send_uv.
    """
def send_uv_grad(*args, **kwargs):
    """
    C++ interface function for send_uv_grad.
    """
def send_v2(*args, **kwargs):
    """
    C++ interface function for send_v2.
    """
def sequence_conv(x: paddle.Tensor, padding_data: paddle.Tensor, filter: paddle.Tensor, context_length: int, padding_trainable: bool = False, context_start: int = 0, context_stride: int = 1) -> paddle.Tensor:
    """
    C++ interface function for sequence_conv.
    """
def sequence_conv_grad(*args, **kwargs):
    """
    C++ interface function for sequence_conv_grad.
    """
def sequence_expand(*args, **kwargs):
    """
    C++ interface function for sequence_expand.
    """
def sequence_expand_grad(*args, **kwargs):
    """
    C++ interface function for sequence_expand_grad.
    """
def sequence_mask(x: paddle.Tensor, max_len: int, out_dtype: paddle._typing.DTypeLike) -> paddle.Tensor:
    """
    C++ interface function for sequence_mask.
    """
def sequence_pool(x: paddle.Tensor, is_test: bool = False, pooltype: str = "AVERAGE", pad_value: float = 0.0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sequence_pool.
    """
def sequence_pool_grad(*args, **kwargs):
    """
    C++ interface function for sequence_pool_grad.
    """
def sequence_softmax(*args, **kwargs):
    """
    C++ interface function for sequence_softmax.
    """
def sequence_softmax_grad(*args, **kwargs):
    """
    C++ interface function for sequence_softmax_grad.
    """
def set(x: paddle.Tensor, source: paddle.Tensor, dims: list[int] = [], stride: list[int] = [], offset: int = 0) -> paddle.Tensor:
    """
    C++ interface function for set.
    """
def set_(x: paddle.Tensor, source: paddle.Tensor, dims: list[int] = [], stride: list[int] = [], offset: int = 0) -> paddle.Tensor:
    """
    C++ interface function for set_.
    """
def set_parameter(*args, **kwargs):
    """
    C++ interface function for set_parameter.
    """
def set_persistable_value(*args, **kwargs):
    """
    C++ interface function for set_persistable_value.
    """
def set_value(*args, **kwargs):
    """
    C++ interface function for set_value.
    """
def set_value_(*args, **kwargs):
    """
    C++ interface function for set_value_.
    """
def set_value_grad(*args, **kwargs):
    """
    C++ interface function for set_value_grad.
    """
def set_value_with_tensor(x: paddle.Tensor, values: paddle.Tensor, starts: list[int], ends: list[int], steps: list[int], axes: list[int], decrease_axes: list[int], none_axes: list[int]) -> paddle.Tensor:
    """
    C++ interface function for set_value_with_tensor.
    """
def set_value_with_tensor_(x: paddle.Tensor, values: paddle.Tensor, starts: list[int], ends: list[int], steps: list[int], axes: list[int], decrease_axes: list[int], none_axes: list[int]) -> paddle.Tensor:
    """
    C++ interface function for set_value_with_tensor_.
    """
def set_value_with_tensor_grad(*args, **kwargs):
    """
    C++ interface function for set_value_with_tensor_grad.
    """
def sgd_(param: paddle.Tensor, learning_rate: paddle.Tensor, grad: paddle.Tensor, master_param: paddle.Tensor, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sgd_.
    """
def shape(input: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for shape.
    """
def shape64(input: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for shape64.
    """
def shard_index(input: paddle.Tensor, index_num: int, nshards: int, shard_id: int, ignore_value: int = -1) -> paddle.Tensor:
    """
    C++ interface function for shard_index.
    """
def share_data(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for share_data.
    """
def share_data_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for share_data_.
    """
def share_var(*args, **kwargs):
    """
    C++ interface function for share_var_op.
    """
def shuffle_batch(x: paddle.Tensor, seed: paddle.Tensor, startup_seed: int = 0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for shuffle_batch.
    """
def shuffle_batch_grad(*args, **kwargs):
    """
    C++ interface function for shuffle_batch_grad.
    """
def shuffle_channel(x: paddle.Tensor, group: int = 1) -> paddle.Tensor:
    """
    C++ interface function for shuffle_channel.
    """
def shuffle_channel_grad(*args, **kwargs):
    """
    C++ interface function for shuffle_channel_grad.
    """
def sigmoid(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sigmoid.
    """
def sigmoid_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sigmoid_.
    """
def sigmoid_cross_entropy_with_logits(x: paddle.Tensor, label: paddle.Tensor, pos_weight: paddle.Tensor, normalize: bool = False, ignore_index: int = -100) -> paddle.Tensor:
    """
    C++ interface function for sigmoid_cross_entropy_with_logits.
    """
def sigmoid_cross_entropy_with_logits_(x: paddle.Tensor, label: paddle.Tensor, pos_weight: paddle.Tensor, normalize: bool = False, ignore_index: int = -100) -> paddle.Tensor:
    """
    C++ interface function for sigmoid_cross_entropy_with_logits_.
    """
def sigmoid_cross_entropy_with_logits_grad(*args, **kwargs):
    """
    C++ interface function for sigmoid_cross_entropy_with_logits_grad.
    """
def sigmoid_cross_entropy_with_logits_grad_(*args, **kwargs):
    """
    C++ interface function for sigmoid_cross_entropy_with_logits_grad_.
    """
def sigmoid_grad(*args, **kwargs):
    """
    C++ interface function for sigmoid_grad.
    """
def sigmoid_grad_(*args, **kwargs):
    """
    C++ interface function for sigmoid_grad_.
    """
def sign(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sign.
    """
def silu(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for silu.
    """
def silu_grad(*args, **kwargs):
    """
    C++ interface function for silu_grad.
    """
def silu_grad_(*args, **kwargs):
    """
    C++ interface function for silu_grad_.
    """
def sin(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sin.
    """
def sin_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sin_.
    """
def sin_grad(*args, **kwargs):
    """
    C++ interface function for sin_grad.
    """
def sin_grad_(*args, **kwargs):
    """
    C++ interface function for sin_grad_.
    """
def sinh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sinh.
    """
def sinh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sinh_.
    """
def sinh_grad(*args, **kwargs):
    """
    C++ interface function for sinh_grad.
    """
def sinh_grad_(*args, **kwargs):
    """
    C++ interface function for sinh_grad_.
    """
def skip_layernorm(*args, **kwargs):
    """
    C++ interface function for skip_layernorm.
    """
def slice(input: paddle.Tensor, axes: list[int], starts: list[int], ends: list[int], infer_flags: list[int], decrease_axis: list[int]) -> paddle.Tensor:
    """
    C++ interface function for slice.
    """
def slice_array(*args, **kwargs):
    """
    C++ interface function for slice_array.
    """
def slice_array_dense(*args, **kwargs):
    """
    C++ interface function for slice_array_dense.
    """
def slice_grad(*args, **kwargs):
    """
    C++ interface function for slice_grad.
    """
def slogdet(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for slogdet.
    """
def slogdet_grad(*args, **kwargs):
    """
    C++ interface function for slogdet_grad.
    """
def softmax(*args, **kwargs):
    """
    C++ interface function for softmax.
    """
def softmax_(*args, **kwargs):
    """
    C++ interface function for softmax_.
    """
def softmax_grad(*args, **kwargs):
    """
    C++ interface function for softmax_grad.
    """
def softplus(x: paddle.Tensor, beta: float = 1.0, threshold: float = 20.0) -> paddle.Tensor:
    """
    C++ interface function for softplus.
    """
def softplus_grad(*args, **kwargs):
    """
    C++ interface function for softplus_grad.
    """
def softplus_grad_(*args, **kwargs):
    """
    C++ interface function for softplus_grad_.
    """
def softshrink(x: paddle.Tensor, threshold: float = 0.5) -> paddle.Tensor:
    """
    C++ interface function for softshrink.
    """
def softshrink_grad(*args, **kwargs):
    """
    C++ interface function for softshrink_grad.
    """
def softshrink_grad_(*args, **kwargs):
    """
    C++ interface function for softshrink_grad_.
    """
def softsign(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for softsign.
    """
def softsign_grad(*args, **kwargs):
    """
    C++ interface function for softsign_grad.
    """
def softsign_grad_(*args, **kwargs):
    """
    C++ interface function for softsign_grad_.
    """
def solve(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for solve.
    """
def solve_grad(*args, **kwargs):
    """
    C++ interface function for solve_grad.
    """
def sparse_abs(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_abs.
    """
def sparse_abs_grad(*args, **kwargs):
    """
    C++ interface function for sparse_abs_grad.
    """
def sparse_acos(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_acos.
    """
def sparse_acos_grad(*args, **kwargs):
    """
    C++ interface function for sparse_acos_grad.
    """
def sparse_acosh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_acosh.
    """
def sparse_acosh_grad(*args, **kwargs):
    """
    C++ interface function for sparse_acosh_grad.
    """
def sparse_add(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_add.
    """
def sparse_add_grad(*args, **kwargs):
    """
    C++ interface function for sparse_add_grad.
    """
def sparse_addmm(input: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor, beta: float = 1.0, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for sparse_addmm.
    """
def sparse_addmm_grad(*args, **kwargs):
    """
    C++ interface function for sparse_addmm_grad.
    """
def sparse_asin(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_asin.
    """
def sparse_asin_grad(*args, **kwargs):
    """
    C++ interface function for sparse_asin_grad.
    """
def sparse_asinh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_asinh.
    """
def sparse_asinh_grad(*args, **kwargs):
    """
    C++ interface function for sparse_asinh_grad.
    """
def sparse_atan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_atan.
    """
def sparse_atan_grad(*args, **kwargs):
    """
    C++ interface function for sparse_atan_grad.
    """
def sparse_atanh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_atanh.
    """
def sparse_atanh_grad(*args, **kwargs):
    """
    C++ interface function for sparse_atanh_grad.
    """
def sparse_attention(q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, offset: paddle.Tensor, columns: paddle.Tensor, key_padding_mask: paddle.Tensor, attn_mask: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sparse_attention.
    """
def sparse_attention_grad(*args, **kwargs):
    """
    C++ interface function for sparse_attention_grad.
    """
def sparse_batch_norm_(x: paddle.Tensor, mean: paddle.Tensor, variance: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, is_test: bool, momentum: float, epsilon: float, data_format: str, use_global_stats: bool, trainable_statistics: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sparse_batch_norm_.
    """
def sparse_batch_norm_grad(*args, **kwargs):
    """
    C++ interface function for sparse_batch_norm_grad.
    """
def sparse_cast(x: paddle.Tensor, index_dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", value_dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED") -> paddle.Tensor:
    """
    C++ interface function for sparse_cast.
    """
def sparse_cast_grad(*args, **kwargs):
    """
    C++ interface function for sparse_cast_grad.
    """
def sparse_coalesce(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_coalesce.
    """
def sparse_conv3d(x: paddle.Tensor, kernel: paddle.Tensor, paddings: list[int], dilations: list[int], strides: list[int], groups: int, subm: bool, key: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sparse_conv3d.
    """
def sparse_conv3d_grad(*args, **kwargs):
    """
    C++ interface function for sparse_conv3d_grad.
    """
def sparse_conv3d_implicit_gemm(x: paddle.Tensor, kernel: paddle.Tensor, paddings: list[int], dilations: list[int], strides: list[int], groups: int, subm: bool, key: str = "") -> paddle.Tensor:
    """
    C++ interface function for sparse_conv3d_implicit_gemm.
    """
def sparse_divide(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_divide.
    """
def sparse_divide_grad(*args, **kwargs):
    """
    C++ interface function for sparse_divide_grad.
    """
def sparse_divide_scalar(x: paddle.Tensor, scalar: float) -> paddle.Tensor:
    """
    C++ interface function for sparse_divide_scalar.
    """
def sparse_expm1(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_expm1.
    """
def sparse_expm1_grad(*args, **kwargs):
    """
    C++ interface function for sparse_expm1_grad.
    """
def sparse_full_like(x: paddle.Tensor, value: float, dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED") -> paddle.Tensor:
    """
    C++ interface function for sparse_full_like.
    """
def sparse_fused_attention(query: paddle.Tensor, key: paddle.Tensor, value: paddle.Tensor, sparse_mask: paddle.Tensor, key_padding_mask: paddle.Tensor, attn_mask: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sparse_fused_attention.
    """
def sparse_fused_attention_grad(*args, **kwargs):
    """
    C++ interface function for sparse_fused_attention_grad.
    """
def sparse_indices(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_indices.
    """
def sparse_isnan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_isnan.
    """
def sparse_leaky_relu(x: paddle.Tensor, alpha: float) -> paddle.Tensor:
    """
    C++ interface function for sparse_leaky_relu.
    """
def sparse_leaky_relu_grad(*args, **kwargs):
    """
    C++ interface function for sparse_leaky_relu_grad.
    """
def sparse_log1p(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_log1p.
    """
def sparse_log1p_grad(*args, **kwargs):
    """
    C++ interface function for sparse_log1p_grad.
    """
def sparse_mask_as(x: paddle.Tensor, mask: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_mask_as.
    """
def sparse_mask_as_grad(*args, **kwargs):
    """
    C++ interface function for sparse_mask_as_grad.
    """
def sparse_masked_matmul(x: paddle.Tensor, y: paddle.Tensor, mask: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_masked_matmul.
    """
def sparse_masked_matmul_grad(*args, **kwargs):
    """
    C++ interface function for sparse_masked_matmul_grad.
    """
def sparse_matmul(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_matmul.
    """
def sparse_matmul_grad(*args, **kwargs):
    """
    C++ interface function for sparse_matmul_grad.
    """
def sparse_maxpool(x: paddle.Tensor, kernel_sizes: list[int], paddings: list[int], dilations: list[int], strides: list[int]) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sparse_maxpool.
    """
def sparse_maxpool_grad(*args, **kwargs):
    """
    C++ interface function for sparse_maxpool_grad.
    """
def sparse_multiply(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_multiply.
    """
def sparse_mv(x: paddle.Tensor, vec: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_mv.
    """
def sparse_mv_grad(*args, **kwargs):
    """
    C++ interface function for sparse_mv_grad.
    """
def sparse_pow(x: paddle.Tensor, factor: float) -> paddle.Tensor:
    """
    C++ interface function for sparse_pow.
    """
def sparse_pow_grad(*args, **kwargs):
    """
    C++ interface function for sparse_pow_grad.
    """
def sparse_relu(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_relu.
    """
def sparse_relu6(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_relu6.
    """
def sparse_relu6_grad(*args, **kwargs):
    """
    C++ interface function for sparse_relu6_grad.
    """
def sparse_relu_grad(*args, **kwargs):
    """
    C++ interface function for sparse_relu_grad.
    """
def sparse_reshape(x: paddle.Tensor, shape: list[int]) -> paddle.Tensor:
    """
    C++ interface function for sparse_reshape.
    """
def sparse_reshape_grad(*args, **kwargs):
    """
    C++ interface function for sparse_reshape_grad.
    """
def sparse_scale(x: paddle.Tensor, scale: float, bias: float, bias_after_scale: bool) -> paddle.Tensor:
    """
    C++ interface function for sparse_scale.
    """
def sparse_sin(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_sin.
    """
def sparse_sin_grad(*args, **kwargs):
    """
    C++ interface function for sparse_sin_grad.
    """
def sparse_sinh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_sinh.
    """
def sparse_sinh_grad(*args, **kwargs):
    """
    C++ interface function for sparse_sinh_grad.
    """
def sparse_slice(x: paddle.Tensor, axes: list[int], starts: list[int], ends: list[int]) -> paddle.Tensor:
    """
    C++ interface function for sparse_slice.
    """
def sparse_slice_grad(*args, **kwargs):
    """
    C++ interface function for sparse_slice_grad.
    """
def sparse_softmax(x: paddle.Tensor, axis: int = -1) -> paddle.Tensor:
    """
    C++ interface function for sparse_softmax.
    """
def sparse_softmax_grad(*args, **kwargs):
    """
    C++ interface function for sparse_softmax_grad.
    """
def sparse_sparse_coo_tensor(values: paddle.Tensor, indices: paddle.Tensor, shape: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for sparse_sparse_coo_tensor.
    """
def sparse_sparse_coo_tensor_grad(*args, **kwargs):
    """
    C++ interface function for sparse_sparse_coo_tensor_grad.
    """
def sparse_sqrt(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_sqrt.
    """
def sparse_sqrt_grad(*args, **kwargs):
    """
    C++ interface function for sparse_sqrt_grad.
    """
def sparse_square(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_square.
    """
def sparse_square_grad(*args, **kwargs):
    """
    C++ interface function for sparse_square_grad.
    """
def sparse_subtract(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_subtract.
    """
def sparse_subtract_grad(*args, **kwargs):
    """
    C++ interface function for sparse_subtract_grad.
    """
def sparse_sum(x: paddle.Tensor, axis: list[int] = [], dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for sparse_sum.
    """
def sparse_sum_grad(*args, **kwargs):
    """
    C++ interface function for sparse_sum_grad.
    """
def sparse_sync_batch_norm_(x: paddle.Tensor, mean: paddle.Tensor, variance: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, is_test: bool, momentum: float, epsilon: float, data_format: str, use_global_stats: bool, trainable_statistics: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sparse_sync_batch_norm_.
    """
def sparse_sync_batch_norm_grad(*args, **kwargs):
    """
    C++ interface function for sparse_sync_batch_norm_grad.
    """
def sparse_tan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_tan.
    """
def sparse_tan_grad(*args, **kwargs):
    """
    C++ interface function for sparse_tan_grad.
    """
def sparse_tanh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_tanh.
    """
def sparse_tanh_grad(*args, **kwargs):
    """
    C++ interface function for sparse_tanh_grad.
    """
def sparse_to_dense(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_to_dense.
    """
def sparse_to_dense_grad(*args, **kwargs):
    """
    C++ interface function for sparse_to_dense_grad.
    """
def sparse_to_sparse_coo(x: paddle.Tensor, sparse_dim: int) -> paddle.Tensor:
    """
    C++ interface function for sparse_to_sparse_coo.
    """
def sparse_to_sparse_coo_grad(*args, **kwargs):
    """
    C++ interface function for sparse_to_sparse_coo_grad.
    """
def sparse_to_sparse_csr(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_to_sparse_csr.
    """
def sparse_transpose(x: paddle.Tensor, perm: list[int]) -> paddle.Tensor:
    """
    C++ interface function for sparse_transpose.
    """
def sparse_transpose_grad(*args, **kwargs):
    """
    C++ interface function for sparse_transpose_grad.
    """
def sparse_values(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sparse_values.
    """
def sparse_values_grad(*args, **kwargs):
    """
    C++ interface function for sparse_values_grad.
    """
def spectral_norm(weight: paddle.Tensor, u: paddle.Tensor, v: paddle.Tensor, dim: int = 0, power_iters: int = 1, eps: float = 1e-12) -> paddle.Tensor:
    """
    C++ interface function for spectral_norm.
    """
def spectral_norm_grad(*args, **kwargs):
    """
    C++ interface function for spectral_norm_grad.
    """
def split(x: paddle.Tensor, sections: list[int], axis: int) -> list[paddle.Tensor]:
    """
    C++ interface function for split.
    """
def split_grad(*args, **kwargs):
    """
    C++ interface function for split_grad.
    """
def split_with_num(x: paddle.Tensor, num: int, axis: int) -> list[paddle.Tensor]:
    """
    C++ interface function for split_with_num.
    """
def sqrt(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sqrt.
    """
def sqrt_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sqrt_.
    """
def sqrt_grad(*args, **kwargs):
    """
    C++ interface function for sqrt_grad.
    """
def sqrt_grad_(*args, **kwargs):
    """
    C++ interface function for sqrt_grad_.
    """
def square(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for square.
    """
def square_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for square_.
    """
def square_grad(*args, **kwargs):
    """
    C++ interface function for square_grad.
    """
def square_grad_(*args, **kwargs):
    """
    C++ interface function for square_grad_.
    """
def squared_l2_norm(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for squared_l2_norm.
    """
def squared_l2_norm_grad(*args, **kwargs):
    """
    C++ interface function for squared_l2_norm_grad.
    """
def squeeze(x: paddle.Tensor, axis: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for squeeze.
    """
def squeeze_(x: paddle.Tensor, axis: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for squeeze_.
    """
def squeeze_excitation_block(*args, **kwargs):
    """
    C++ interface function for squeeze_excitation_block.
    """
def squeeze_grad(*args, **kwargs):
    """
    C++ interface function for squeeze_grad.
    """
def squeeze_grad_(*args, **kwargs):
    """
    C++ interface function for squeeze_grad_.
    """
def stack(x: list[paddle.Tensor], axis: int = 0) -> paddle.Tensor:
    """
    C++ interface function for stack.
    """
def stack_grad(*args, **kwargs):
    """
    C++ interface function for stack_grad.
    """
def standard_gamma(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for standard_gamma.
    """
def stanh(x: paddle.Tensor, scale_a: float = 0.67, scale_b: float = 1.7159) -> paddle.Tensor:
    """
    C++ interface function for stanh.
    """
def stanh_grad(*args, **kwargs):
    """
    C++ interface function for stanh_grad.
    """
def stft(x: paddle.Tensor, window: paddle.Tensor, n_fft: int, hop_length: int, normalized: bool, onesided: bool) -> paddle.Tensor:
    """
    C++ interface function for stft.
    """
def stft_grad(*args, **kwargs):
    """
    C++ interface function for stft_grad.
    """
def strided_slice(x: paddle.Tensor, axes: list[int], starts: list[int], ends: list[int], strides: list[int]) -> paddle.Tensor:
    """
    C++ interface function for strided_slice.
    """
def strided_slice_grad(*args, **kwargs):
    """
    C++ interface function for strided_slice_grad.
    """
def subtract(*args, **kwargs):
    """
    C++ interface function for subtract.
    """
def subtract_(*args, **kwargs):
    """
    C++ interface function for subtract_.
    """
def subtract_grad(*args, **kwargs):
    """
    C++ interface function for subtract_grad.
    """
def subtract_grad_(*args, **kwargs):
    """
    C++ interface function for subtract_grad_.
    """
def sum(x: paddle.Tensor, axis: list[int] = [], dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for sum.
    """
def sum_grad(*args, **kwargs):
    """
    C++ interface function for sum_grad.
    """
def svd(x: paddle.Tensor, full_matrices: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for svd.
    """
def svd_grad(*args, **kwargs):
    """
    C++ interface function for svd_grad.
    """
def svdvals(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for svdvals.
    """
def svdvals_grad(*args, **kwargs):
    """
    C++ interface function for svdvals_grad.
    """
def swiglu(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for swiglu.
    """
def swiglu_grad(*args, **kwargs):
    """
    C++ interface function for swiglu_grad.
    """
def swish(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for swish.
    """
def swish_grad(*args, **kwargs):
    """
    C++ interface function for swish_grad.
    """
def swish_grad_(*args, **kwargs):
    """
    C++ interface function for swish_grad_.
    """
def sync_batch_norm_(x: paddle.Tensor, mean: paddle.Tensor, variance: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, is_test: bool, momentum: float, epsilon: float, data_format: str, use_global_stats: bool, trainable_statistics: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sync_batch_norm_.
    """
def sync_batch_norm_grad(*args, **kwargs):
    """
    C++ interface function for sync_batch_norm_grad.
    """
def sync_calc_stream(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sync_calc_stream.
    """
def sync_calc_stream_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sync_calc_stream_.
    """
def take_along_axis(arr: paddle.Tensor, indices: paddle.Tensor, axis: int) -> paddle.Tensor:
    """
    C++ interface function for take_along_axis.
    """
def take_along_axis_grad(*args, **kwargs):
    """
    C++ interface function for take_along_axis_grad.
    """
def tan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tan.
    """
def tan_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tan_.
    """
def tan_grad(*args, **kwargs):
    """
    C++ interface function for tan_grad.
    """
def tan_grad_(*args, **kwargs):
    """
    C++ interface function for tan_grad_.
    """
def tanh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tanh.
    """
def tanh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tanh_.
    """
def tanh_grad(*args, **kwargs):
    """
    C++ interface function for tanh_grad.
    """
def tanh_grad_(*args, **kwargs):
    """
    C++ interface function for tanh_grad_.
    """
def tanh_shrink(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tanh_shrink.
    """
def tanh_shrink_grad(*args, **kwargs):
    """
    C++ interface function for tanh_shrink_grad.
    """
def tanh_shrink_grad_(*args, **kwargs):
    """
    C++ interface function for tanh_shrink_grad_.
    """
def tdm_child(x: paddle.Tensor, tree_info: paddle.Tensor, child_nums: int, dtype: paddle._typing.DTypeLike = "DataType::INT32") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for tdm_child.
    """
def tdm_sampler(x: paddle.Tensor, travel: paddle.Tensor, layer: paddle.Tensor, output_positive: bool = True, neg_samples_num_list: list[int] = [], layer_offset: list[int] = [], seed: int = 0, dtype: int = 2) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for tdm_sampler.
    """
def temporal_shift(x: paddle.Tensor, seg_num: int, shift_ratio: float = 0.25, data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for temporal_shift.
    """
def temporal_shift_grad(*args, **kwargs):
    """
    C++ interface function for temporal_shift_grad.
    """
def tensorrt_engine(*args, **kwargs):
    """
    C++ interface function for tensorrt_engine.
    """
def thresholded_relu(x: paddle.Tensor, threshold: float = 1.0, value: float = 0.0) -> paddle.Tensor:
    """
    C++ interface function for thresholded_relu.
    """
def thresholded_relu_(x: paddle.Tensor, threshold: float = 1.0, value: float = 0.0) -> paddle.Tensor:
    """
    C++ interface function for thresholded_relu_.
    """
def thresholded_relu_grad(*args, **kwargs):
    """
    C++ interface function for thresholded_relu_grad.
    """
def thresholded_relu_grad_(*args, **kwargs):
    """
    C++ interface function for thresholded_relu_grad_.
    """
def tile(*args, **kwargs):
    """
    C++ interface function for tile.
    """
def tile_grad(*args, **kwargs):
    """
    C++ interface function for tile_grad.
    """
def top_p_sampling(x: paddle.Tensor, ps: paddle.Tensor, threshold: paddle.Tensor, topp_seed: paddle.Tensor, seed: int = -1, k: int = 0, mode: str = "truncate") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for top_p_sampling.
    """
def topk(x: paddle.Tensor, k: int = 1, axis: int = -1, largest: bool = True, sorted: bool = True) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for topk.
    """
def topk_grad(*args, **kwargs):
    """
    C++ interface function for topk_grad.
    """
def trace(x: paddle.Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> paddle.Tensor:
    """
    C++ interface function for trace.
    """
def trace_grad(*args, **kwargs):
    """
    C++ interface function for trace_grad.
    """
def trans_layout(x: paddle.Tensor, perm: list[int]) -> paddle.Tensor:
    """
    C++ interface function for trans_layout.
    """
def trans_layout_grad(*args, **kwargs):
    """
    C++ interface function for trans_layout_grad.
    """
def transpose(x: paddle.Tensor, perm: list[int]) -> paddle.Tensor:
    """
    C++ interface function for transpose.
    """
def transpose_(x: paddle.Tensor, perm: list[int]) -> paddle.Tensor:
    """
    C++ interface function for transpose_.
    """
def transpose_grad(*args, **kwargs):
    """
    C++ interface function for transpose_grad.
    """
def triangular_solve(x: paddle.Tensor, y: paddle.Tensor, upper: bool = True, transpose: bool = False, unitriangular: bool = False) -> paddle.Tensor:
    """
    C++ interface function for triangular_solve.
    """
def triangular_solve_grad(*args, **kwargs):
    """
    C++ interface function for triangular_solve_grad.
    """
def tril(x: paddle.Tensor, diagonal: int) -> paddle.Tensor:
    """
    C++ interface function for tril.
    """
def tril_(x: paddle.Tensor, diagonal: int) -> paddle.Tensor:
    """
    C++ interface function for tril_.
    """
def tril_grad(*args, **kwargs):
    """
    C++ interface function for tril_grad.
    """
def tril_indices(rows: int, cols: int, offset: int, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for tril_indices.
    """
def trilinear_interp(x: paddle.Tensor, out_size: paddle.Tensor, size_tensor: list[paddle.Tensor], scale_tensor: paddle.Tensor, data_format: str = "NCHW", out_d: int = 0, out_h: int = 0, out_w: int = 0, scale: list[float] = [], interp_method: str = "bilinear", align_corners: bool = True, align_mode: int = 1) -> paddle.Tensor:
    """
    C++ interface function for trilinear_interp.
    """
def trilinear_interp_grad(*args, **kwargs):
    """
    C++ interface function for trilinear_interp_grad.
    """
def triu(x: paddle.Tensor, diagonal: int) -> paddle.Tensor:
    """
    C++ interface function for triu.
    """
def triu_(x: paddle.Tensor, diagonal: int) -> paddle.Tensor:
    """
    C++ interface function for triu_.
    """
def triu_grad(*args, **kwargs):
    """
    C++ interface function for triu_grad.
    """
def triu_indices(row: int, col: int, offset: int, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for triu_indices.
    """
def trunc(input: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for trunc.
    """
def trunc_(input: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for trunc_.
    """
def trunc_grad(*args, **kwargs):
    """
    C++ interface function for trunc_grad.
    """
def truncated_gaussian_random(shape: list[int], mean: float, std: float, seed: int, a: float, b: float, dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for truncated_gaussian_random.
    """
def unbind(input: paddle.Tensor, axis: int = 0) -> list[paddle.Tensor]:
    """
    C++ interface function for unbind.
    """
def unfold(x: paddle.Tensor, kernel_sizes: list[int], strides: list[int], paddings: list[int], dilations: list[int]) -> paddle.Tensor:
    """
    C++ interface function for unfold.
    """
def unfold_grad(*args, **kwargs):
    """
    C++ interface function for unfold_grad.
    """
def uniform(shape: list[int], dtype: paddle._typing.DTypeLike, min: float, max: float, seed: int, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for uniform.
    """
def uniform_inplace(x: paddle.Tensor, min: float = -1.0, max: float = 1.0, seed: int = 0, diag_num: int = 0, diag_step: int = 0, diag_val: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for uniform_inplace.
    """
def uniform_inplace_(x: paddle.Tensor, min: float = -1.0, max: float = 1.0, seed: int = 0, diag_num: int = 0, diag_step: int = 0, diag_val: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for uniform_inplace_.
    """
def uniform_inplace_grad(*args, **kwargs):
    """
    C++ interface function for uniform_inplace_grad.
    """
def uniform_inplace_grad_(*args, **kwargs):
    """
    C++ interface function for uniform_inplace_grad_.
    """
def uniform_random_batch_size_like(input: paddle.Tensor, shape: list[int], input_dim_idx: int = 0, output_dim_idx: int = 0, min: float = -1.0, max: float = 1.0, seed: int = 0, diag_num: int = 0, diag_step: int = 0, diag_val: float = 1.0, dtype: paddle._typing.DTypeLike = "DataType::FLOAT32") -> paddle.Tensor:
    """
    C++ interface function for uniform_random_batch_size_like.
    """
def unique(*args, **kwargs):
    """
    C++ interface function for unique.
    """
def unique_consecutive(x: paddle.Tensor, return_inverse: bool = False, return_counts: bool = False, axis: list[int] = [], dtype: paddle._typing.DTypeLike = "DataType::FLOAT32") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for unique_consecutive.
    """
def unpool(x: paddle.Tensor, indices: paddle.Tensor, ksize: list[int], strides: list[int], padding: list[int], output_size: list[int], data_format: str) -> paddle.Tensor:
    """
    C++ interface function for unpool.
    """
def unpool3d(x: paddle.Tensor, indices: paddle.Tensor, ksize: list[int], strides: list[int] = [1,1,1], paddings: list[int] = [0,0,0], output_size: list[int] = [0,0,0], data_format: str = "NCDHW") -> paddle.Tensor:
    """
    C++ interface function for unpool3d.
    """
def unpool3d_grad(*args, **kwargs):
    """
    C++ interface function for unpool3d_grad.
    """
def unpool_grad(*args, **kwargs):
    """
    C++ interface function for unpool_grad.
    """
def unsqueeze(x: paddle.Tensor, axis: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for unsqueeze.
    """
def unsqueeze_(x: paddle.Tensor, axis: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for unsqueeze_.
    """
def unsqueeze_grad(*args, **kwargs):
    """
    C++ interface function for unsqueeze_grad.
    """
def unsqueeze_grad_(*args, **kwargs):
    """
    C++ interface function for unsqueeze_grad_.
    """
def unstack(x: paddle.Tensor, axis: int = 0, num: int = 0) -> list[paddle.Tensor]:
    """
    C++ interface function for unstack.
    """
def unstack_grad(*args, **kwargs):
    """
    C++ interface function for unstack_grad.
    """
def update_loss_scaling_(x: list[paddle.Tensor], found_infinite: paddle.Tensor, prev_loss_scaling: paddle.Tensor, in_good_steps: paddle.Tensor, in_bad_steps: paddle.Tensor, incr_every_n_steps: int, decr_every_n_nan_or_inf: int, incr_ratio: float, decr_ratio: float, stop_update: float = False) -> tuple[list[paddle.Tensor], paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for update_loss_scaling_.
    """
def update_parameter(*args, **kwargs):
    """
    C++ interface function for update_parameter.
    """
def variable_length_memory_efficient_attention(*args, **kwargs):
    """
    C++ interface function for variable_length_memory_efficient_attention.
    """
def variance(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for variance.
    """
def view_dtype(input: paddle.Tensor, dtype: paddle._typing.DTypeLike) -> paddle.Tensor:
    """
    C++ interface function for view_dtype.
    """
def view_dtype_grad(*args, **kwargs):
    """
    C++ interface function for view_dtype_grad.
    """
def view_slice(input: paddle.Tensor, begin_idx: int, end_idx: int) -> paddle.Tensor:
    """
    C++ interface function for view_slice.
    """
def viterbi_decode(potentials: paddle.Tensor, transition_params: paddle.Tensor, lengths: paddle.Tensor, include_bos_eos_tag: bool = True) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for viterbi_decode.
    """
def warpctc(logits: paddle.Tensor, label: paddle.Tensor, logits_length: paddle.Tensor, labels_length: paddle.Tensor, blank: int = 0, norm_by_times: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for warpctc.
    """
def warpctc_grad(*args, **kwargs):
    """
    C++ interface function for warpctc_grad.
    """
def warprnnt(input: paddle.Tensor, label: paddle.Tensor, input_lengths: paddle.Tensor, label_lengths: paddle.Tensor, blank: int = 0, fastemit_lambda: float = 0.0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for warprnnt.
    """
def warprnnt_grad(*args, **kwargs):
    """
    C++ interface function for warprnnt_grad.
    """
def weight_dequantize(x: paddle.Tensor, scale: paddle.Tensor, algo: str = "weight_only_int8", out_dtype: paddle._typing.DTypeLike = "DataType::FLOAT16", group_size: int = -1) -> paddle.Tensor:
    """
    C++ interface function for weight_dequantize.
    """
def weight_only_linear(x: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, weight_scale: paddle.Tensor, weight_dtype: str, arch: int = 80, group_size: int = -1) -> paddle.Tensor:
    """
    C++ interface function for weight_only_linear.
    """
def weight_only_linear_grad(*args, **kwargs):
    """
    C++ interface function for weight_only_linear_grad.
    """
def weight_quantize(x: paddle.Tensor, algo: str = "weight_only_int8", arch: int = 80, group_size: int = -1) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for weight_quantize.
    """
def weighted_sample_neighbors(row: paddle.Tensor, colptr: paddle.Tensor, edge_weight: paddle.Tensor, input_nodes: paddle.Tensor, eids: paddle.Tensor, sample_size: int, return_eids: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for weighted_sample_neighbors.
    """
def where(condition: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for where.
    """
def where_(condition: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for where_.
    """
def where_grad(*args, **kwargs):
    """
    C++ interface function for where_grad.
    """
def yolo_box(x: paddle.Tensor, img_size: paddle.Tensor, anchors: list[int] = [], class_num: int = 1, conf_thresh: float = 0.01, downsample_ratio: int = 32, clip_bbox: bool = True, scale_x_y: float = 1.0, iou_aware: bool = False, iou_aware_factor: float = 0.5) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for yolo_box.
    """
def yolo_box_head(x: paddle.Tensor, anchors: list[int], class_num: int) -> paddle.Tensor:
    """
    C++ interface function for yolo_box_head.
    """
def yolo_box_post(boxes0: paddle.Tensor, boxes1: paddle.Tensor, boxes2: paddle.Tensor, image_shape: paddle.Tensor, image_scale: paddle.Tensor, anchors0: list[int], anchors1: list[int], anchors2: list[int], class_num: int, conf_thresh: float, downsample_ratio0: int, downsample_ratio1: int, downsample_ratio2: int, clip_bbox: bool, scale_x_y: float, nms_threshold: float) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for yolo_box_post.
    """
def yolo_loss(x: paddle.Tensor, gt_box: paddle.Tensor, gt_label: paddle.Tensor, gt_score: paddle.Tensor, anchors: list[int] = [], anchor_mask: list[int] = [], class_num: int = 1, ignore_thresh: float = 0.7, downsample_ratio: int = 32, use_label_smooth: bool = True, scale_x_y: float = 1.0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for yolo_loss.
    """
def yolo_loss_grad(*args, **kwargs):
    """
    C++ interface function for yolo_loss_grad.
    """
