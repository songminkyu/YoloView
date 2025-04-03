from __future__ import annotations
import paddle
from . import legacy
__all__ = ['abs', 'abs_', 'abs_grad', 'accuracy', 'accuracy_check', 'acos', 'acos_', 'acos_grad', 'acos_grad_', 'acosh', 'acosh_', 'acosh_grad', 'acosh_grad_', 'adadelta_', 'adagrad_', 'adam_', 'adamax_', 'adamw_', 'add', 'add_', 'add_grad', 'add_grad_', 'add_group_norm_silu', 'add_n', 'add_position_encoding', 'add_position_encoding_grad', 'addmm', 'addmm_', 'addmm_grad', 'affine_channel', 'affine_channel_', 'affine_channel_grad', 'affine_channel_grad_', 'affine_grid', 'affine_grid_grad', 'all', 'all_gather', 'all_reduce', 'all_reduce_', 'all_to_all', 'allclose', 'amax', 'amax_grad', 'amin', 'amin_grad', 'angle', 'angle_grad', 'any', 'apply_per_channel_scale', 'arange', 'argmax', 'argmin', 'argsort', 'argsort_grad', 'as_complex', 'as_complex_grad', 'as_real', 'as_real_grad', 'as_strided', 'as_strided_grad', 'asgd_', 'asin', 'asin_', 'asin_grad', 'asin_grad_', 'asinh', 'asinh_', 'asinh_grad', 'asinh_grad_', 'assign', 'assign_', 'assign_grad', 'assign_out_', 'assign_out__grad', 'assign_out__grad_', 'assign_pos', 'assign_value_', 'atan', 'atan2', 'atan2_grad', 'atan_', 'atan_grad', 'atan_grad_', 'atanh', 'atanh_', 'atanh_grad', 'atanh_grad_', 'attention_lstm', 'auc', 'average_accumulates_', 'baddbmm', 'baddbmm_', 'baddbmm_grad', 'barrier', 'batch_fc', 'batch_fc_grad', 'batch_norm', 'batch_norm_grad', 'bce_loss', 'bce_loss_', 'bce_loss_grad', 'bce_loss_grad_', 'beam_search', 'bernoulli', 'bicubic_interp', 'bicubic_interp_grad', 'bilinear', 'bilinear_grad', 'bilinear_interp', 'bilinear_interp_grad', 'bincount', 'binomial', 'bipartite_match', 'bitwise_and', 'bitwise_and_', 'bitwise_left_shift', 'bitwise_left_shift_', 'bitwise_not', 'bitwise_not_', 'bitwise_or', 'bitwise_or_', 'bitwise_right_shift', 'bitwise_right_shift_', 'bitwise_xor', 'bitwise_xor_', 'blha_get_max_len', 'block_multihead_attention_', 'block_multihead_attention_xpu', 'block_multihead_attention_xpu_', 'bmm', 'bmm_grad', 'box_clip', 'box_coder', 'broadcast', 'broadcast_', 'broadcast_tensors', 'broadcast_tensors_grad', 'c_allreduce_sum', 'c_allreduce_sum_', 'c_concat', 'c_concat_grad', 'c_embedding', 'c_embedding_grad', 'c_identity', 'c_identity_', 'c_scatter', 'c_softmax_with_cross_entropy', 'c_softmax_with_cross_entropy_grad', 'c_split', 'calc_reduced_attn_scores', 'cast', 'cast_', 'cast_grad', 'ceil', 'ceil_', 'ceil_grad', 'ceil_grad_', 'celu', 'celu_grad', 'celu_grad_', 'channel_shuffle', 'channel_shuffle_grad', 'check_finite_and_unscale_', 'check_numerics', 'cholesky', 'cholesky_grad', 'cholesky_solve', 'cholesky_solve_grad', 'chunk_eval', 'class_center_sample', 'clip', 'clip_', 'clip_by_norm', 'clip_grad', 'clip_grad_', 'coalesce_tensor', 'collect_fpn_proposals', 'complex', 'complex_grad', 'concat', 'concat_grad', 'conj', 'conj_grad', 'conv2d', 'conv2d_transpose', 'conv2d_transpose_bias', 'conv2d_transpose_grad', 'conv3d', 'conv3d_grad', 'conv3d_transpose', 'conv3d_transpose_grad', 'copy_to', 'copysign', 'copysign_', 'copysign_grad', 'copysign_grad_', 'correlation', 'correlation_grad', 'cos', 'cos_', 'cos_grad', 'cos_grad_', 'cosh', 'cosh_', 'cosh_grad', 'cosh_grad_', 'crf_decoding', 'crop', 'crop_grad', 'cross', 'cross_entropy_with_softmax', 'cross_entropy_with_softmax_', 'cross_entropy_with_softmax_grad', 'cross_entropy_with_softmax_grad_', 'cross_grad', 'ctc_align', 'cudnn_lstm', 'cudnn_lstm_grad', 'cummax', 'cummax_grad', 'cummin', 'cummin_grad', 'cumprod', 'cumprod_', 'cumprod_grad', 'cumsum', 'cumsum_', 'cumsum_grad', 'cvm', 'cvm_grad', 'data', 'decayed_adagrad', 'decode_jpeg', 'deformable_conv', 'deformable_conv_grad', 'depend', 'depthwise_conv2d', 'depthwise_conv2d_grad', 'depthwise_conv2d_transpose', 'depthwise_conv2d_transpose_grad', 'dequantize_abs_max', 'dequantize_log', 'det', 'det_grad', 'dgc', 'dgc_clip_by_norm', 'dgc_momentum', 'diag', 'diag_embed', 'diag_grad', 'diagonal', 'diagonal_grad', 'digamma', 'digamma_', 'digamma_grad', 'dirichlet', 'disable_check_model_nan_inf', 'disable_check_model_nan_inf_grad', 'dist', 'dist_grad', 'distribute_fpn_proposals', 'divide', 'divide_', 'divide_grad', 'dot', 'dot_grad', 'dpsgd', 'dropout', 'dropout_grad', 'edit_distance', 'eig', 'eig_grad', 'eigh', 'eigh_grad', 'eigvals', 'eigvalsh', 'eigvalsh_grad', 'einsum', 'einsum_grad', 'elementwise_pow', 'elementwise_pow_grad', 'elu', 'elu_', 'elu_grad', 'elu_grad_', 'embedding', 'embedding_grad', 'embedding_grad_dense', 'embedding_with_scaled_gradient', 'embedding_with_scaled_gradient_grad', 'empty', 'empty_like', 'enable_check_model_nan_inf', 'enable_check_model_nan_inf_grad', 'equal', 'equal_', 'equal_all', 'erf', 'erf_', 'erf_grad', 'erfinv', 'erfinv_', 'erfinv_grad', 'exp', 'exp_', 'exp_grad', 'exp_grad_', 'expand', 'expand_as', 'expand_as_grad', 'expand_grad', 'expm1', 'expm1_', 'expm1_grad', 'expm1_grad_', 'exponential_', 'exponential__grad', 'eye', 'fake_channel_wise_dequantize_max_abs', 'fake_channel_wise_quantize_abs_max', 'fake_channel_wise_quantize_dequantize_abs_max', 'fake_channel_wise_quantize_dequantize_abs_max_grad', 'fake_dequantize_max_abs', 'fake_quantize_abs_max', 'fake_quantize_dequantize_abs_max', 'fake_quantize_dequantize_abs_max_grad', 'fake_quantize_dequantize_moving_average_abs_max', 'fake_quantize_dequantize_moving_average_abs_max_', 'fake_quantize_dequantize_moving_average_abs_max_grad', 'fake_quantize_moving_average_abs_max', 'fake_quantize_moving_average_abs_max_', 'fake_quantize_range_abs_max', 'fake_quantize_range_abs_max_', 'fft_c2c', 'fft_c2c_grad', 'fft_c2r', 'fft_c2r_grad', 'fft_r2c', 'fft_r2c_grad', 'fill', 'fill_', 'fill_diagonal', 'fill_diagonal_', 'fill_diagonal_grad', 'fill_diagonal_tensor', 'fill_diagonal_tensor_', 'fill_diagonal_tensor_grad', 'fill_diagonal_tensor_grad_', 'fill_grad', 'fill_grad_', 'flash_attn', 'flash_attn_grad', 'flash_attn_qkvpacked', 'flash_attn_qkvpacked_grad', 'flash_attn_unpadded', 'flash_attn_unpadded_grad', 'flash_attn_varlen_qkvpacked', 'flash_attn_varlen_qkvpacked_grad', 'flashmask_attention', 'flashmask_attention_grad', 'flatten', 'flatten_', 'flatten_grad', 'flatten_grad_', 'flip', 'flip_grad', 'floor', 'floor_', 'floor_divide', 'floor_divide_', 'floor_grad', 'floor_grad_', 'fmax', 'fmax_grad', 'fmin', 'fmin_grad', 'fold', 'fold_grad', 'fp8_fp8_half_gemm_fused', 'fractional_max_pool2d', 'fractional_max_pool2d_grad', 'fractional_max_pool3d', 'fractional_max_pool3d_grad', 'frame', 'frame_grad', 'frobenius_norm', 'frobenius_norm_grad', 'ftrl', 'full', 'full_', 'full_batch_size_like', 'full_int_array', 'full_like', 'full_with_tensor', 'fused_adam_', 'fused_batch_norm_act', 'fused_batch_norm_act_grad', 'fused_bias_act', 'fused_bias_dropout_residual_layer_norm', 'fused_bias_dropout_residual_layer_norm_grad', 'fused_bias_residual_layernorm', 'fused_bn_add_activation', 'fused_bn_add_activation_grad', 'fused_dot_product_attention', 'fused_dot_product_attention_grad', 'fused_dropout_add', 'fused_dropout_add_grad', 'fused_elementwise_add', 'fused_elementwise_div', 'fused_elementwise_mul', 'fused_elementwise_sub', 'fused_gemm_epilogue', 'fused_gemm_epilogue_grad', 'fused_linear_param_grad_add', 'fused_moe', 'fused_multi_transformer_', 'fused_rotary_position_embedding', 'fused_rotary_position_embedding_grad', 'fused_softmax_mask', 'fused_softmax_mask_grad', 'fused_softmax_mask_upper_triangle', 'fused_softmax_mask_upper_triangle_grad', 'fused_token_prune', 'fusion_seqpool_concat', 'fusion_seqpool_cvm_concat', 'gammaincc', 'gammaincc_', 'gammaincc_grad', 'gammaln', 'gammaln_', 'gammaln_grad', 'gather', 'gather_grad', 'gather_nd', 'gather_nd_grad', 'gather_tree', 'gaussian', 'gaussian_inplace', 'gaussian_inplace_', 'gaussian_inplace_grad', 'gaussian_inplace_grad_', 'gelu', 'gelu_grad', 'gemm_epilogue', 'generate_proposals', 'get_core_ops_args_info', 'get_core_ops_args_type_info', 'get_core_ops_returns_info', 'global_gather', 'global_gather_grad', 'global_scatter', 'global_scatter_grad', 'graph_khop_sampler', 'graph_sample_neighbors', 'greater_equal', 'greater_equal_', 'greater_than', 'greater_than_', 'grid_sample', 'grid_sample_grad', 'group_norm', 'group_norm_grad', 'group_norm_grad_', 'gru', 'gru_grad', 'gru_unit', 'gru_unit_grad', 'gumbel_softmax', 'gumbel_softmax_grad', 'hardshrink', 'hardshrink_grad', 'hardshrink_grad_', 'hardsigmoid', 'hardsigmoid_grad', 'hardsigmoid_grad_', 'hardswish', 'hardswish_grad', 'hardswish_grad_', 'hardtanh', 'hardtanh_', 'hardtanh_grad', 'hardtanh_grad_', 'heaviside', 'heaviside_grad', 'hinge_loss', 'hinge_loss_grad', 'histogram', 'hsigmoid_loss', 'hsigmoid_loss_grad', 'huber_loss', 'huber_loss_grad', 'i0', 'i0_', 'i0_grad', 'i0e', 'i0e_grad', 'i1', 'i1_grad', 'i1e', 'i1e_grad', 'identity_loss', 'identity_loss_', 'identity_loss_grad', 'identity_loss_grad_', 'im2sequence', 'im2sequence_grad', 'imag', 'imag_grad', 'increment', 'increment_', 'index_add', 'index_add_', 'index_add_grad', 'index_add_grad_', 'index_put', 'index_put_', 'index_put_grad', 'index_sample', 'index_sample_grad', 'index_select', 'index_select_grad', 'index_select_strided', 'index_select_strided_grad', 'instance_norm', 'instance_norm_grad', 'inverse', 'inverse_grad', 'is_empty', 'isclose', 'isfinite', 'isinf', 'isnan', 'kldiv_loss', 'kldiv_loss_grad', 'kron', 'kron_grad', 'kthvalue', 'kthvalue_grad', 'l1_norm', 'l1_norm_', 'l1_norm_grad', 'label_smooth', 'label_smooth_grad', 'lamb_', 'layer_norm', 'layer_norm_grad', 'leaky_relu', 'leaky_relu_', 'leaky_relu_grad', 'leaky_relu_grad_', 'legacy', 'lerp', 'lerp_', 'lerp_grad', 'less_equal', 'less_equal_', 'less_than', 'less_than_', 'lgamma', 'lgamma_', 'lgamma_grad', 'limit_by_capacity', 'linear', 'linear_interp', 'linear_interp_grad', 'linspace', 'llm_int8_linear', 'log', 'log10', 'log10_', 'log10_grad', 'log10_grad_', 'log1p', 'log1p_', 'log1p_grad', 'log1p_grad_', 'log2', 'log2_', 'log2_grad', 'log2_grad_', 'log_', 'log_grad', 'log_grad_', 'log_loss', 'log_loss_grad', 'log_softmax', 'log_softmax_grad', 'logcumsumexp', 'logcumsumexp_grad', 'logical_and', 'logical_and_', 'logical_not', 'logical_not_', 'logical_or', 'logical_or_', 'logical_xor', 'logical_xor_', 'logit', 'logit_', 'logit_grad', 'logsigmoid', 'logsigmoid_grad', 'logsigmoid_grad_', 'logspace', 'logsumexp', 'logsumexp_grad', 'lookup_table_dequant', 'lp_pool2d', 'lp_pool2d_grad', 'lstm', 'lstm_grad', 'lstsq', 'lu', 'lu_', 'lu_grad', 'lu_grad_', 'lu_solve', 'lu_solve_grad', 'lu_unpack', 'lu_unpack_grad', 'margin_cross_entropy', 'margin_cross_entropy_grad', 'margin_cross_entropy_grad_', 'masked_multihead_attention_', 'masked_select', 'masked_select_grad', 'match_matrix_tensor', 'match_matrix_tensor_grad', 'matmul', 'matmul_grad', 'matrix_nms', 'matrix_power', 'matrix_power_grad', 'matrix_rank', 'matrix_rank_atol_rtol', 'matrix_rank_tol', 'max', 'max_grad', 'max_pool2d_with_index', 'max_pool2d_with_index_grad', 'max_pool3d_with_index', 'max_pool3d_with_index_grad', 'maximum', 'maximum_grad', 'maxout', 'maxout_grad', 'mean', 'mean_all', 'mean_all_grad', 'mean_grad', 'memcpy_d2h', 'memcpy_h2d', 'memory_efficient_attention', 'memory_efficient_attention_grad', 'merge_selected_rows', 'merged_adam_', 'merged_momentum_', 'meshgrid', 'meshgrid_grad', 'min', 'min_grad', 'minimum', 'minimum_grad', 'mish', 'mish_grad', 'mish_grad_', 'mode', 'mode_grad', 'moe_dispatch', 'moe_ffn', 'moe_reduce', 'momentum_', 'mp_allreduce_sum', 'mp_allreduce_sum_', 'mp_allreduce_sum_grad', 'multi_dot', 'multi_dot_grad', 'multiclass_nms3', 'multinomial', 'multiplex', 'multiplex_grad', 'multiply', 'multiply_', 'mv', 'mv_grad', 'nadam_', 'nanmedian', 'nanmedian_grad', 'nearest_interp', 'nearest_interp_grad', 'nextafter', 'nll_loss', 'nll_loss_grad', 'nms', 'nonzero', 'norm', 'norm_grad', 'not_equal', 'not_equal_', 'npu_identity', 'number_count', 'numel', 'one_hot', 'ones', 'ones_like', 'overlap_add', 'overlap_add_grad', 'p_norm', 'p_norm_grad', 'pad', 'pad3d', 'pad3d_grad', 'pad_grad', 'partial_allgather', 'partial_allgather_', 'partial_concat', 'partial_concat_grad', 'partial_sum', 'partial_sum_grad', 'pixel_shuffle', 'pixel_shuffle_grad', 'pixel_unshuffle', 'pixel_unshuffle_grad', 'poisson', 'poisson_grad', 'polygamma', 'polygamma_', 'polygamma_grad', 'pool2d', 'pool2d_grad', 'pool3d', 'pool3d_grad', 'pow', 'pow_', 'pow_grad', 'pow_grad_', 'prelu', 'prelu_grad', 'prior_box', 'prod', 'prod_grad', 'prune_gate_by_capacity', 'psroi_pool', 'psroi_pool_grad', 'put_along_axis', 'put_along_axis_', 'put_along_axis_grad', 'pyramid_hash', 'pyramid_hash_grad', 'qr', 'qr_grad', 'radam_', 'randint', 'random_routing', 'random_routing_', 'randperm', 'rank_attention', 'rank_attention_grad', 'read_file', 'real', 'real_grad', 'reciprocal', 'reciprocal_', 'reciprocal_grad', 'reciprocal_grad_', 'reduce', 'reduce_', 'reduce_as', 'reduce_as_grad', 'reduce_scatter', 'reindex_graph', 'relu', 'relu6', 'relu6_grad', 'relu6_grad_', 'relu_', 'relu_grad', 'relu_grad_', 'remainder', 'remainder_', 'remainder_grad', 'renorm', 'renorm_', 'renorm_grad', 'repeat_interleave', 'repeat_interleave_grad', 'repeat_interleave_with_tensor_index', 'repeat_interleave_with_tensor_index_grad', 'reshape', 'reshape_', 'reshape_grad', 'reshape_grad_', 'resnet_basic_block', 'resnet_basic_block_grad', 'resnet_unit', 'resnet_unit_grad', 'reverse', 'reverse_grad', 'rms_norm', 'rms_norm_grad', 'rmsprop_', 'rnn', 'rnn_grad', 'roi_align', 'roi_align_grad', 'roi_pool', 'roi_pool_grad', 'roll', 'roll_grad', 'round', 'round_', 'round_grad', 'round_grad_', 'rprop_', 'rrelu', 'rrelu_grad', 'rsqrt', 'rsqrt_', 'rsqrt_grad', 'rsqrt_grad_', 'scale', 'scale_', 'scatter', 'scatter_', 'scatter_grad', 'scatter_nd_add', 'scatter_nd_add_grad', 'searchsorted', 'segment_pool', 'segment_pool_grad', 'selu', 'selu_grad', 'send_u_recv', 'send_u_recv_grad', 'send_ue_recv', 'send_ue_recv_grad', 'send_uv', 'send_uv_grad', 'sequence_conv', 'sequence_conv_grad', 'sequence_mask', 'sequence_pool', 'sequence_pool_grad', 'set', 'set_', 'set_value', 'set_value_', 'set_value_grad', 'set_value_with_tensor', 'set_value_with_tensor_', 'set_value_with_tensor_grad', 'sgd_', 'shape', 'shape64', 'shard_index', 'share_data', 'shuffle_batch', 'shuffle_batch_grad', 'shuffle_channel', 'shuffle_channel_grad', 'sigmoid', 'sigmoid_', 'sigmoid_cross_entropy_with_logits', 'sigmoid_cross_entropy_with_logits_', 'sigmoid_cross_entropy_with_logits_grad', 'sigmoid_cross_entropy_with_logits_grad_', 'sigmoid_grad', 'sigmoid_grad_', 'sign', 'sign_grad', 'silu', 'silu_grad', 'silu_grad_', 'sin', 'sin_', 'sin_grad', 'sin_grad_', 'sinh', 'sinh_', 'sinh_grad', 'sinh_grad_', 'slice', 'slice_grad', 'slogdet', 'slogdet_grad', 'softmax', 'softmax_', 'softmax_grad', 'softplus', 'softplus_grad', 'softplus_grad_', 'softshrink', 'softshrink_grad', 'softshrink_grad_', 'softsign', 'softsign_grad', 'softsign_grad_', 'solve', 'solve_grad', 'sparse_abs', 'sparse_abs_grad', 'sparse_acos', 'sparse_acos_grad', 'sparse_acosh', 'sparse_acosh_grad', 'sparse_add', 'sparse_add_grad', 'sparse_addmm', 'sparse_addmm_grad', 'sparse_asin', 'sparse_asin_grad', 'sparse_asinh', 'sparse_asinh_grad', 'sparse_atan', 'sparse_atan_grad', 'sparse_atanh', 'sparse_atanh_grad', 'sparse_attention', 'sparse_attention_grad', 'sparse_batch_norm_', 'sparse_batch_norm_grad', 'sparse_cast', 'sparse_cast_grad', 'sparse_coalesce', 'sparse_conv3d', 'sparse_conv3d_grad', 'sparse_conv3d_implicit_gemm', 'sparse_divide', 'sparse_divide_grad', 'sparse_divide_scalar', 'sparse_divide_scalar_grad', 'sparse_expm1', 'sparse_expm1_grad', 'sparse_full_like', 'sparse_fused_attention', 'sparse_fused_attention_grad', 'sparse_indices', 'sparse_isnan', 'sparse_leaky_relu', 'sparse_leaky_relu_grad', 'sparse_log1p', 'sparse_log1p_grad', 'sparse_mask_as', 'sparse_mask_as_grad', 'sparse_masked_matmul', 'sparse_masked_matmul_grad', 'sparse_matmul', 'sparse_matmul_grad', 'sparse_maxpool', 'sparse_maxpool_grad', 'sparse_multiply', 'sparse_mv', 'sparse_mv_grad', 'sparse_pow', 'sparse_pow_grad', 'sparse_relu', 'sparse_relu6', 'sparse_relu6_grad', 'sparse_relu_grad', 'sparse_reshape', 'sparse_reshape_grad', 'sparse_scale', 'sparse_sin', 'sparse_sin_grad', 'sparse_sinh', 'sparse_sinh_grad', 'sparse_slice', 'sparse_slice_grad', 'sparse_softmax', 'sparse_softmax_grad', 'sparse_sparse_coo_tensor', 'sparse_sparse_coo_tensor_grad', 'sparse_sqrt', 'sparse_sqrt_grad', 'sparse_square', 'sparse_square_grad', 'sparse_subtract', 'sparse_subtract_grad', 'sparse_sum', 'sparse_sum_grad', 'sparse_sync_batch_norm_', 'sparse_sync_batch_norm_grad', 'sparse_tan', 'sparse_tan_grad', 'sparse_tanh', 'sparse_tanh_grad', 'sparse_to_dense', 'sparse_to_dense_grad', 'sparse_to_sparse_coo', 'sparse_to_sparse_coo_grad', 'sparse_to_sparse_csr', 'sparse_transpose', 'sparse_transpose_grad', 'sparse_values', 'sparse_values_grad', 'spectral_norm', 'spectral_norm_grad', 'split', 'split_grad', 'split_with_num', 'split_with_num_grad', 'sqrt', 'sqrt_', 'sqrt_grad', 'sqrt_grad_', 'square', 'square_', 'square_grad', 'square_grad_', 'squared_l2_norm', 'squared_l2_norm_grad', 'squeeze', 'squeeze_', 'squeeze_grad', 'squeeze_grad_', 'stack', 'stack_grad', 'standard_gamma', 'stanh', 'stanh_grad', 'stft', 'stft_grad', 'strided_slice', 'strided_slice_grad', 'strings_empty', 'strings_empty_like', 'strings_lower', 'strings_upper', 'subtract', 'subtract_', 'subtract_grad', 'subtract_grad_', 'sum', 'sum_grad', 'svd', 'svd_grad', 'svdvals', 'svdvals_grad', 'swiglu', 'swiglu_grad', 'swish', 'swish_grad', 'swish_grad_', 'sync_batch_norm_', 'sync_batch_norm_grad', 'sync_calc_stream', 'sync_calc_stream_', 'sync_comm_stream', 'take_along_axis', 'take_along_axis_grad', 'tan', 'tan_', 'tan_grad', 'tan_grad_', 'tanh', 'tanh_', 'tanh_grad', 'tanh_grad_', 'tanh_shrink', 'tanh_shrink_grad', 'tanh_shrink_grad_', 'tdm_child', 'tdm_sampler', 'temporal_shift', 'temporal_shift_grad', 'tensor_unfold', 'tensor_unfold_grad', 'thresholded_relu', 'thresholded_relu_', 'thresholded_relu_grad', 'thresholded_relu_grad_', 'tile', 'tile_grad', 'top_p_sampling', 'topk', 'topk_grad', 'trace', 'trace_grad', 'trans_layout', 'trans_layout_grad', 'transpose', 'transpose_', 'transpose_grad', 'triangular_solve', 'triangular_solve_grad', 'tril', 'tril_', 'tril_grad', 'tril_indices', 'trilinear_interp', 'trilinear_interp_grad', 'triu', 'triu_', 'triu_grad', 'triu_indices', 'trunc', 'trunc_', 'trunc_grad', 'truncated_gaussian_random', 'unbind', 'unbind_grad', 'unfold', 'unfold_grad', 'uniform', 'uniform_inplace', 'uniform_inplace_', 'uniform_inplace_grad', 'uniform_inplace_grad_', 'uniform_random_batch_size_like', 'unique', 'unique_consecutive', 'unpool', 'unpool3d', 'unpool3d_grad', 'unpool_grad', 'unsqueeze', 'unsqueeze_', 'unsqueeze_grad', 'unsqueeze_grad_', 'unstack', 'unstack_grad', 'update_loss_scaling_', 'variable_length_memory_efficient_attention', 'variance', 'view_dtype', 'view_dtype_grad', 'view_shape', 'view_shape_grad', 'view_slice', 'viterbi_decode', 'warpctc', 'warpctc_grad', 'warprnnt', 'warprnnt_grad', 'weight_dequantize', 'weight_only_linear', 'weight_only_linear_grad', 'weight_quantize', 'weighted_sample_neighbors', 'where', 'where_', 'where_grad', 'yolo_box', 'yolo_box_head', 'yolo_box_post', 'yolo_loss', 'yolo_loss_grad', 'zeros', 'zeros_like']
def abs(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for abs in dygraph.
    """
def abs_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for abs_ in dygraph.
    """
def abs_grad(*args, **kwargs):
    """
    C++ interface function for abs_grad in dygraph.
    """
def accuracy(x: paddle.Tensor, indices: paddle.Tensor, label: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for accuracy in dygraph.
    """
def accuracy_check(x: paddle.Tensor, y: paddle.Tensor, fn_name: str, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> paddle.Tensor:
    """
    C++ interface function for accuracy_check in dygraph.
    """
def acos(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for acos in dygraph.
    """
def acos_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for acos_ in dygraph.
    """
def acos_grad(*args, **kwargs):
    """
    C++ interface function for acos_grad in dygraph.
    """
def acos_grad_(*args, **kwargs):
    """
    C++ interface function for acos_grad_ in dygraph.
    """
def acosh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for acosh in dygraph.
    """
def acosh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for acosh_ in dygraph.
    """
def acosh_grad(*args, **kwargs):
    """
    C++ interface function for acosh_grad in dygraph.
    """
def acosh_grad_(*args, **kwargs):
    """
    C++ interface function for acosh_grad_ in dygraph.
    """
def adadelta_(param: paddle.Tensor, grad: paddle.Tensor, avg_squared_grad: paddle.Tensor, avg_squared_update: paddle.Tensor, learning_rate: paddle.Tensor, master_param: paddle.Tensor, rho: float = 0.95, epsilon: float = 1.0e-6, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for adadelta_ in dygraph.
    """
def adagrad_(param: paddle.Tensor, grad: paddle.Tensor, moment: paddle.Tensor, learning_rate: paddle.Tensor, master_param: paddle.Tensor, epsilon: float = 1.0e-6, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for adagrad_ in dygraph.
    """
def adam_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, moment1: paddle.Tensor, moment2: paddle.Tensor, moment2_max: paddle.Tensor, beta1_pow: paddle.Tensor, beta2_pow: paddle.Tensor, master_param: paddle.Tensor, skip_update: paddle.Tensor, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, lazy_mode: bool = False, min_row_size_to_use_multithread: int = 1000, multi_precision: bool = False, use_global_beta_pow: bool = False, amsgrad: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for adam_ in dygraph.
    """
def adamax_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, moment: paddle.Tensor, inf_norm: paddle.Tensor, beta1_pow: paddle.Tensor, master_param: paddle.Tensor, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for adamax_ in dygraph.
    """
def adamw_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, moment1: paddle.Tensor, moment2: paddle.Tensor, moment2_max: paddle.Tensor, beta1_pow: paddle.Tensor, beta2_pow: paddle.Tensor, master_param: paddle.Tensor, skip_update: paddle.Tensor, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, lr_ratio: float = 1.0, coeff: float = 0.01, with_decay: bool = False, lazy_mode: bool = False, min_row_size_to_use_multithread: int = 1000, multi_precision: bool = False, use_global_beta_pow: bool = False, amsgrad: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for adamw_ in dygraph.
    """
def add(*args, **kwargs):
    """
    C++ interface function for add in dygraph.
    """
def add_(*args, **kwargs):
    """
    C++ interface function for add_ in dygraph.
    """
def add_grad(*args, **kwargs):
    """
    C++ interface function for add_grad in dygraph.
    """
def add_grad_(*args, **kwargs):
    """
    C++ interface function for add_grad_ in dygraph.
    """
def add_group_norm_silu(*args, **kwargs):
    """
    C++ interface function for add_group_norm_silu in dygraph.
    """
def add_n(*args, **kwargs):
    """
    C++ interface function for add_n in dygraph.
    """
def add_position_encoding(x: paddle.Tensor, alpha: float = 1.0, beta: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for add_position_encoding in dygraph.
    """
def add_position_encoding_grad(*args, **kwargs):
    """
    C++ interface function for add_position_encoding_grad in dygraph.
    """
def addmm(input: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor, beta: float = 1.0, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for addmm in dygraph.
    """
def addmm_(input: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor, beta: float = 1.0, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for addmm_ in dygraph.
    """
def addmm_grad(*args, **kwargs):
    """
    C++ interface function for addmm_grad in dygraph.
    """
def affine_channel(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, data_layout: str = "AnyLayout") -> paddle.Tensor:
    """
    C++ interface function for affine_channel in dygraph.
    """
def affine_channel_(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, data_layout: str = "AnyLayout") -> paddle.Tensor:
    """
    C++ interface function for affine_channel_ in dygraph.
    """
def affine_channel_grad(*args, **kwargs):
    """
    C++ interface function for affine_channel_grad in dygraph.
    """
def affine_channel_grad_(*args, **kwargs):
    """
    C++ interface function for affine_channel_grad_ in dygraph.
    """
def affine_grid(input: paddle.Tensor, output_shape: list[int] = [], align_corners: bool = True) -> paddle.Tensor:
    """
    C++ interface function for affine_grid in dygraph.
    """
def affine_grid_grad(*args, **kwargs):
    """
    C++ interface function for affine_grid_grad in dygraph.
    """
def all(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for all in dygraph.
    """
def all_gather(x: paddle.Tensor, ring_id: int = 0, nranks: int = 0) -> paddle.Tensor:
    """
    C++ interface function for all_gather in dygraph.
    """
def all_reduce(x: paddle.Tensor, ring_id: int = 0, reduce_type: int = 0) -> paddle.Tensor:
    """
    C++ interface function for all_reduce in dygraph.
    """
def all_reduce_(x: paddle.Tensor, ring_id: int = 0, reduce_type: int = 0) -> paddle.Tensor:
    """
    C++ interface function for all_reduce_ in dygraph.
    """
def all_to_all(x: paddle.Tensor, ring_id: int = 0) -> paddle.Tensor:
    """
    C++ interface function for all_to_all in dygraph.
    """
def allclose(x: paddle.Tensor, y: paddle.Tensor, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> paddle.Tensor:
    """
    C++ interface function for allclose in dygraph.
    """
def amax(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for amax in dygraph.
    """
def amax_grad(*args, **kwargs):
    """
    C++ interface function for amax_grad in dygraph.
    """
def amin(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for amin in dygraph.
    """
def amin_grad(*args, **kwargs):
    """
    C++ interface function for amin_grad in dygraph.
    """
def angle(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for angle in dygraph.
    """
def angle_grad(*args, **kwargs):
    """
    C++ interface function for angle_grad in dygraph.
    """
def any(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for any in dygraph.
    """
def apply_per_channel_scale(x: paddle.Tensor, scales: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for apply_per_channel_scale in dygraph.
    """
def arange(*args, **kwargs):
    """
    C++ interface function for arange in dygraph.
    """
def argmax(x: paddle.Tensor, axis: int, keepdims: bool = False, flatten: bool = False, dtype: paddle._typing.DTypeLike = "DataType::INT64") -> paddle.Tensor:
    """
    C++ interface function for argmax in dygraph.
    """
def argmin(x: paddle.Tensor, axis: int, keepdims: bool = False, flatten: bool = False, dtype: paddle._typing.DTypeLike = "DataType::INT64") -> paddle.Tensor:
    """
    C++ interface function for argmin in dygraph.
    """
def argsort(x: paddle.Tensor, axis: int = -1, descending: bool = False, stable: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for argsort in dygraph.
    """
def argsort_grad(*args, **kwargs):
    """
    C++ interface function for argsort_grad in dygraph.
    """
def as_complex(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for as_complex in dygraph.
    """
def as_complex_grad(*args, **kwargs):
    """
    C++ interface function for as_complex_grad in dygraph.
    """
def as_real(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for as_real in dygraph.
    """
def as_real_grad(*args, **kwargs):
    """
    C++ interface function for as_real_grad in dygraph.
    """
def as_strided(input: paddle.Tensor, dims: list[int] = [], stride: list[int] = [], offset: int = 0) -> paddle.Tensor:
    """
    C++ interface function for as_strided in dygraph.
    """
def as_strided_grad(*args, **kwargs):
    """
    C++ interface function for as_strided_grad in dygraph.
    """
def asgd_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, d: paddle.Tensor, y: paddle.Tensor, n: paddle.Tensor, master_param: paddle.Tensor, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for asgd_ in dygraph.
    """
def asin(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for asin in dygraph.
    """
def asin_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for asin_ in dygraph.
    """
def asin_grad(*args, **kwargs):
    """
    C++ interface function for asin_grad in dygraph.
    """
def asin_grad_(*args, **kwargs):
    """
    C++ interface function for asin_grad_ in dygraph.
    """
def asinh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for asinh in dygraph.
    """
def asinh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for asinh_ in dygraph.
    """
def asinh_grad(*args, **kwargs):
    """
    C++ interface function for asinh_grad in dygraph.
    """
def asinh_grad_(*args, **kwargs):
    """
    C++ interface function for asinh_grad_ in dygraph.
    """
def assign(*args, **kwargs):
    """
    C++ interface function for assign in dygraph.
    """
def assign_(*args, **kwargs):
    """
    C++ interface function for assign_ in dygraph.
    """
def assign_grad(*args, **kwargs):
    """
    C++ interface function for assign_grad in dygraph.
    """
def assign_out_(x: paddle.Tensor, output: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for assign_out_ in dygraph.
    """
def assign_out__grad(*args, **kwargs):
    """
    C++ interface function for assign_out__grad in dygraph.
    """
def assign_out__grad_(*args, **kwargs):
    """
    C++ interface function for assign_out__grad_ in dygraph.
    """
def assign_pos(x: paddle.Tensor, cum_count: paddle.Tensor, eff_num_len: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for assign_pos in dygraph.
    """
def assign_value_(output: paddle.Tensor, shape: list[int], dtype: paddle._typing.DTypeLike, values: list[float], place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for assign_value_ in dygraph.
    """
def atan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atan in dygraph.
    """
def atan2(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atan2 in dygraph.
    """
def atan2_grad(*args, **kwargs):
    """
    C++ interface function for atan2_grad in dygraph.
    """
def atan_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atan_ in dygraph.
    """
def atan_grad(*args, **kwargs):
    """
    C++ interface function for atan_grad in dygraph.
    """
def atan_grad_(*args, **kwargs):
    """
    C++ interface function for atan_grad_ in dygraph.
    """
def atanh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atanh in dygraph.
    """
def atanh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atanh_ in dygraph.
    """
def atanh_grad(*args, **kwargs):
    """
    C++ interface function for atanh_grad in dygraph.
    """
def atanh_grad_(*args, **kwargs):
    """
    C++ interface function for atanh_grad_ in dygraph.
    """
def attention_lstm(x: paddle.Tensor, c0: paddle.Tensor, h0: paddle.Tensor, attention_weight: paddle.Tensor, attention_bias: paddle.Tensor, attention_scalar: paddle.Tensor, attention_scalar_bias: paddle.Tensor, lstm_weight: paddle.Tensor, lstm_bias: paddle.Tensor, gate_activation: str = "sigmoid", cell_activation: str = "tanh", candidate_activation: str = "tanh") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for attention_lstm in dygraph.
    """
def auc(x: paddle.Tensor, label: paddle.Tensor, stat_pos: paddle.Tensor, stat_neg: paddle.Tensor, ins_tag_weight: paddle.Tensor, curve: str = "ROC", num_thresholds: int = (2 << 12) - 1, slide_steps: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for auc in dygraph.
    """
def average_accumulates_(param: paddle.Tensor, in_sum_1: paddle.Tensor, in_sum_2: paddle.Tensor, in_sum_3: paddle.Tensor, in_num_accumulates: paddle.Tensor, in_old_num_accumulates: paddle.Tensor, in_num_updates: paddle.Tensor, average_window: float = 0, max_average_window: int = "INT64_MAX", min_average_window: int = 10000) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for average_accumulates_ in dygraph.
    """
def baddbmm(input: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor, beta: float = 1.0, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for baddbmm in dygraph.
    """
def baddbmm_(input: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor, beta: float = 1.0, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for baddbmm_ in dygraph.
    """
def baddbmm_grad(*args, **kwargs):
    """
    C++ interface function for baddbmm_grad in dygraph.
    """
def barrier(x: paddle.Tensor, ring_id: int = 0) -> paddle.Tensor:
    """
    C++ interface function for barrier in dygraph.
    """
def batch_fc(input: paddle.Tensor, w: paddle.Tensor, bias: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for batch_fc in dygraph.
    """
def batch_fc_grad(*args, **kwargs):
    """
    C++ interface function for batch_fc_grad in dygraph.
    """
def batch_norm(*args, **kwargs):
    """
    C++ interface function for batch_norm in dygraph.
    """
def batch_norm_grad(*args, **kwargs):
    """
    C++ interface function for batch_norm_grad in dygraph.
    """
def bce_loss(input: paddle.Tensor, label: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bce_loss in dygraph.
    """
def bce_loss_(input: paddle.Tensor, label: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bce_loss_ in dygraph.
    """
def bce_loss_grad(*args, **kwargs):
    """
    C++ interface function for bce_loss_grad in dygraph.
    """
def bce_loss_grad_(*args, **kwargs):
    """
    C++ interface function for bce_loss_grad_ in dygraph.
    """
def beam_search(pre_ids: paddle.Tensor, pre_scores: paddle.Tensor, ids: paddle.Tensor, scores: paddle.Tensor, level: int, beam_size: int, end_id: int, is_accumulated: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for beam_search in dygraph.
    """
def bernoulli(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bernoulli in dygraph.
    """
def bicubic_interp(x: paddle.Tensor, out_size: paddle.Tensor, size_tensor: list[paddle.Tensor], scale_tensor: paddle.Tensor, data_format: str = "NCHW", out_d: int = 0, out_h: int = 0, out_w: int = 0, scale: list[float] = [], interp_method: str = "bilinear", align_corners: bool = True, align_mode: int = 1) -> paddle.Tensor:
    """
    C++ interface function for bicubic_interp in dygraph.
    """
def bicubic_interp_grad(*args, **kwargs):
    """
    C++ interface function for bicubic_interp_grad in dygraph.
    """
def bilinear(x: paddle.Tensor, y: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bilinear in dygraph.
    """
def bilinear_grad(*args, **kwargs):
    """
    C++ interface function for bilinear_grad in dygraph.
    """
def bilinear_interp(x: paddle.Tensor, out_size: paddle.Tensor, size_tensor: list[paddle.Tensor], scale_tensor: paddle.Tensor, data_format: str = "NCHW", out_d: int = 0, out_h: int = 0, out_w: int = 0, scale: list[float] = [], interp_method: str = "bilinear", align_corners: bool = True, align_mode: int = 1) -> paddle.Tensor:
    """
    C++ interface function for bilinear_interp in dygraph.
    """
def bilinear_interp_grad(*args, **kwargs):
    """
    C++ interface function for bilinear_interp_grad in dygraph.
    """
def bincount(x: paddle.Tensor, weights: paddle.Tensor, minlength: int = 0) -> paddle.Tensor:
    """
    C++ interface function for bincount in dygraph.
    """
def binomial(count: paddle.Tensor, prob: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for binomial in dygraph.
    """
def bipartite_match(dist_mat: paddle.Tensor, match_type: str = "bipartite", dist_threshold: float = 0.5) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for bipartite_match in dygraph.
    """
def bitwise_and(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_and in dygraph.
    """
def bitwise_and_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_and_ in dygraph.
    """
def bitwise_left_shift(x: paddle.Tensor, y: paddle.Tensor, is_arithmetic: bool = True) -> paddle.Tensor:
    """
    C++ interface function for bitwise_left_shift in dygraph.
    """
def bitwise_left_shift_(x: paddle.Tensor, y: paddle.Tensor, is_arithmetic: bool = True) -> paddle.Tensor:
    """
    C++ interface function for bitwise_left_shift_ in dygraph.
    """
def bitwise_not(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_not in dygraph.
    """
def bitwise_not_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_not_ in dygraph.
    """
def bitwise_or(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_or in dygraph.
    """
def bitwise_or_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_or_ in dygraph.
    """
def bitwise_right_shift(x: paddle.Tensor, y: paddle.Tensor, is_arithmetic: bool = True) -> paddle.Tensor:
    """
    C++ interface function for bitwise_right_shift in dygraph.
    """
def bitwise_right_shift_(x: paddle.Tensor, y: paddle.Tensor, is_arithmetic: bool = True) -> paddle.Tensor:
    """
    C++ interface function for bitwise_right_shift_ in dygraph.
    """
def bitwise_xor(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_xor in dygraph.
    """
def bitwise_xor_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bitwise_xor_ in dygraph.
    """
def blha_get_max_len(*args, **kwargs):
    """
    C++ interface function for blha_get_max_len in dygraph.
    """
def block_multihead_attention_(*args, **kwargs):
    """
    C++ interface function for block_multihead_attention_ in dygraph.
    """
def block_multihead_attention_xpu(*args, **kwargs):
    """
    C++ interface function for block_multihead_attention_xpu in dygraph.
    """
def block_multihead_attention_xpu_(*args, **kwargs):
    """
    C++ interface function for block_multihead_attention_xpu_ in dygraph.
    """
def bmm(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for bmm in dygraph.
    """
def bmm_grad(*args, **kwargs):
    """
    C++ interface function for bmm_grad in dygraph.
    """
def box_clip(input: paddle.Tensor, im_info: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for box_clip in dygraph.
    """
def box_coder(prior_box: paddle.Tensor, prior_box_var: paddle.Tensor, target_box: paddle.Tensor, code_type: str = "encode_center_size", box_normalized: bool = True, axis: int = 0, variance: list[float] = []) -> paddle.Tensor:
    """
    C++ interface function for box_coder in dygraph.
    """
def broadcast(x: paddle.Tensor, ring_id: int = 0, root: int = 0) -> paddle.Tensor:
    """
    C++ interface function for broadcast in dygraph.
    """
def broadcast_(x: paddle.Tensor, ring_id: int = 0, root: int = 0) -> paddle.Tensor:
    """
    C++ interface function for broadcast_ in dygraph.
    """
def broadcast_tensors(input: list[paddle.Tensor]) -> list[paddle.Tensor]:
    """
    C++ interface function for broadcast_tensors in dygraph.
    """
def broadcast_tensors_grad(*args, **kwargs):
    """
    C++ interface function for broadcast_tensors_grad in dygraph.
    """
def c_allreduce_sum(x: paddle.Tensor, ring_id: int, use_calc_stream: bool, use_model_parallel: bool) -> paddle.Tensor:
    """
    C++ interface function for c_allreduce_sum in dygraph.
    """
def c_allreduce_sum_(x: paddle.Tensor, ring_id: int, use_calc_stream: bool, use_model_parallel: bool) -> paddle.Tensor:
    """
    C++ interface function for c_allreduce_sum_ in dygraph.
    """
def c_concat(x: paddle.Tensor, rank: int, nranks: int, ring_id: int, use_calc_stream: bool, use_model_parallel: bool) -> paddle.Tensor:
    """
    C++ interface function for c_concat in dygraph.
    """
def c_concat_grad(*args, **kwargs):
    """
    C++ interface function for c_concat_grad in dygraph.
    """
def c_embedding(*args, **kwargs):
    """
    C++ interface function for c_embedding in dygraph.
    """
def c_embedding_grad(*args, **kwargs):
    """
    C++ interface function for c_embedding_grad in dygraph.
    """
def c_identity(x: paddle.Tensor, ring_id: int, use_calc_stream: bool, use_model_parallel: bool) -> paddle.Tensor:
    """
    C++ interface function for c_identity in dygraph.
    """
def c_identity_(x: paddle.Tensor, ring_id: int, use_calc_stream: bool, use_model_parallel: bool) -> paddle.Tensor:
    """
    C++ interface function for c_identity_ in dygraph.
    """
def c_scatter(x: paddle.Tensor, ring_id: int = 0, root: int = 0, nranks: int = 0, use_calc_stream: bool = False) -> paddle.Tensor:
    """
    C++ interface function for c_scatter in dygraph.
    """
def c_softmax_with_cross_entropy(logits: paddle.Tensor, label: paddle.Tensor, ignore_index: int = -100, ring_id: int = 0, rank: int = 0, nranks: int = 0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for c_softmax_with_cross_entropy in dygraph.
    """
def c_softmax_with_cross_entropy_grad(*args, **kwargs):
    """
    C++ interface function for c_softmax_with_cross_entropy_grad in dygraph.
    """
def c_split(x: paddle.Tensor, rank: int = 0, nranks: int = 1, ring_id: int = 0, use_model_parallel: bool = True) -> paddle.Tensor:
    """
    C++ interface function for c_split in dygraph.
    """
def calc_reduced_attn_scores(q: paddle.Tensor, k: paddle.Tensor, softmax_lse: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for calc_reduced_attn_scores in dygraph.
    """
def cast(x: paddle.Tensor, dtype: paddle._typing.DTypeLike) -> paddle.Tensor:
    """
    C++ interface function for cast in dygraph.
    """
def cast_(x: paddle.Tensor, dtype: paddle._typing.DTypeLike) -> paddle.Tensor:
    """
    C++ interface function for cast_ in dygraph.
    """
def cast_grad(*args, **kwargs):
    """
    C++ interface function for cast_grad in dygraph.
    """
def ceil(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for ceil in dygraph.
    """
def ceil_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for ceil_ in dygraph.
    """
def ceil_grad(*args, **kwargs):
    """
    C++ interface function for ceil_grad in dygraph.
    """
def ceil_grad_(*args, **kwargs):
    """
    C++ interface function for ceil_grad_ in dygraph.
    """
def celu(x: paddle.Tensor, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for celu in dygraph.
    """
def celu_grad(*args, **kwargs):
    """
    C++ interface function for celu_grad in dygraph.
    """
def celu_grad_(*args, **kwargs):
    """
    C++ interface function for celu_grad_ in dygraph.
    """
def channel_shuffle(x: paddle.Tensor, groups: int, data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for channel_shuffle in dygraph.
    """
def channel_shuffle_grad(*args, **kwargs):
    """
    C++ interface function for channel_shuffle_grad in dygraph.
    """
def check_finite_and_unscale_(x: list[paddle.Tensor], scale: paddle.Tensor) -> tuple[list[paddle.Tensor], paddle.Tensor]:
    """
    C++ interface function for check_finite_and_unscale_ in dygraph.
    """
def check_numerics(tensor: paddle.Tensor, op_type: str = "", var_name: str = "", check_nan_inf_level: int = 0, stack_height_limit: int = -1, output_dir: str = "") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for check_numerics in dygraph.
    """
def cholesky(x: paddle.Tensor, upper: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cholesky in dygraph.
    """
def cholesky_grad(*args, **kwargs):
    """
    C++ interface function for cholesky_grad in dygraph.
    """
def cholesky_solve(x: paddle.Tensor, y: paddle.Tensor, upper: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cholesky_solve in dygraph.
    """
def cholesky_solve_grad(*args, **kwargs):
    """
    C++ interface function for cholesky_solve_grad in dygraph.
    """
def chunk_eval(inference: paddle.Tensor, label: paddle.Tensor, seq_length: paddle.Tensor, num_chunk_types: int, chunk_scheme: str = "IOB", excluded_chunk_types: list[int] = []) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for chunk_eval in dygraph.
    """
def class_center_sample(label: paddle.Tensor, num_classes: int, num_samples: int, ring_id: int = 0, rank: int = 0, nranks: int = 1, fix_seed: bool = False, seed: int = 0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for class_center_sample in dygraph.
    """
def clip(x: paddle.Tensor, min: float, max: float) -> paddle.Tensor:
    """
    C++ interface function for clip in dygraph.
    """
def clip_(x: paddle.Tensor, min: float, max: float) -> paddle.Tensor:
    """
    C++ interface function for clip_ in dygraph.
    """
def clip_by_norm(x: paddle.Tensor, max_norm: float) -> paddle.Tensor:
    """
    C++ interface function for clip_by_norm in dygraph.
    """
def clip_grad(*args, **kwargs):
    """
    C++ interface function for clip_grad in dygraph.
    """
def clip_grad_(*args, **kwargs):
    """
    C++ interface function for clip_grad_ in dygraph.
    """
def coalesce_tensor(input: list[paddle.Tensor], dtype: paddle._typing.DTypeLike, copy_data: bool = False, set_constant: bool = False, persist_output: bool = False, constant: float = 0.0, use_align: bool = True, align_size: int = -1, size_of_dtype: int = -1, concated_shapes: list[int] = [], concated_ranks: list[int] = []) -> tuple[list[paddle.Tensor], paddle.Tensor]:
    """
    C++ interface function for coalesce_tensor in dygraph.
    """
def collect_fpn_proposals(multi_level_rois: list[paddle.Tensor], multi_level_scores: list[paddle.Tensor], multi_level_rois_num: list[paddle.Tensor], post_nms_topn: int) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for collect_fpn_proposals in dygraph.
    """
def complex(real: paddle.Tensor, imag: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for complex in dygraph.
    """
def complex_grad(*args, **kwargs):
    """
    C++ interface function for complex_grad in dygraph.
    """
def concat(x: list[paddle.Tensor], axis: float = 0) -> paddle.Tensor:
    """
    C++ interface function for concat in dygraph.
    """
def concat_grad(*args, **kwargs):
    """
    C++ interface function for concat_grad in dygraph.
    """
def conj(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for conj in dygraph.
    """
def conj_grad(*args, **kwargs):
    """
    C++ interface function for conj_grad in dygraph.
    """
def conv2d(input: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1], paddings: list[int] = [0, 0], padding_algorithm: str = "EXPLICIT", dilations: list[int] = [1, 1], groups: int = 1, data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for conv2d in dygraph.
    """
def conv2d_transpose(x: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1], paddings: list[int] = [0, 0], output_padding: list[int] = [], output_size: list[int] = [], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1], data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for conv2d_transpose in dygraph.
    """
def conv2d_transpose_bias(x: paddle.Tensor, filter: paddle.Tensor, bias: paddle.Tensor, strides: list[int] = [1, 1], paddings: list[int] = [0, 0], output_padding: list[int] = [], output_size: list[int] = [], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1], data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for conv2d_transpose_bias in dygraph.
    """
def conv2d_transpose_grad(*args, **kwargs):
    """
    C++ interface function for conv2d_transpose_grad in dygraph.
    """
def conv3d(input: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1, 1], paddings: list[int] = [0, 0, 0], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1, 1], data_format: str = "NCDHW") -> paddle.Tensor:
    """
    C++ interface function for conv3d in dygraph.
    """
def conv3d_grad(*args, **kwargs):
    """
    C++ interface function for conv3d_grad in dygraph.
    """
def conv3d_transpose(x: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1, 1], paddings: list[int] = [0, 0, 0], output_padding: list[int] = [], output_size: list[int] = [], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1, 1], data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for conv3d_transpose in dygraph.
    """
def conv3d_transpose_grad(*args, **kwargs):
    """
    C++ interface function for conv3d_transpose_grad in dygraph.
    """
def copy_to(x: paddle.Tensor, place: paddle._typing.PlaceLike, blocking: bool) -> paddle.Tensor:
    """
    C++ interface function for copy_to in dygraph.
    """
def copysign(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for copysign in dygraph.
    """
def copysign_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for copysign_ in dygraph.
    """
def copysign_grad(*args, **kwargs):
    """
    C++ interface function for copysign_grad in dygraph.
    """
def copysign_grad_(*args, **kwargs):
    """
    C++ interface function for copysign_grad_ in dygraph.
    """
def correlation(input1: paddle.Tensor, input2: paddle.Tensor, pad_size: int, kernel_size: int, max_displacement: int, stride1: int, stride2: int, corr_type_multiply: int = 1) -> paddle.Tensor:
    """
    C++ interface function for correlation in dygraph.
    """
def correlation_grad(*args, **kwargs):
    """
    C++ interface function for correlation_grad in dygraph.
    """
def cos(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for cos in dygraph.
    """
def cos_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for cos_ in dygraph.
    """
def cos_grad(*args, **kwargs):
    """
    C++ interface function for cos_grad in dygraph.
    """
def cos_grad_(*args, **kwargs):
    """
    C++ interface function for cos_grad_ in dygraph.
    """
def cosh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for cosh in dygraph.
    """
def cosh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for cosh_ in dygraph.
    """
def cosh_grad(*args, **kwargs):
    """
    C++ interface function for cosh_grad in dygraph.
    """
def cosh_grad_(*args, **kwargs):
    """
    C++ interface function for cosh_grad_ in dygraph.
    """
def crf_decoding(emission: paddle.Tensor, transition: paddle.Tensor, label: paddle.Tensor, length: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for crf_decoding in dygraph.
    """
def crop(x: paddle.Tensor, shape: list[int] = [], offsets: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for crop in dygraph.
    """
def crop_grad(*args, **kwargs):
    """
    C++ interface function for crop_grad in dygraph.
    """
def cross(x: paddle.Tensor, y: paddle.Tensor, axis: int = 9) -> paddle.Tensor:
    """
    C++ interface function for cross in dygraph.
    """
def cross_entropy_with_softmax(input: paddle.Tensor, label: paddle.Tensor, soft_label: bool = False, use_softmax: bool = True, numeric_stable_mode: bool = True, ignore_index: int = -100, axis: int = -1) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for cross_entropy_with_softmax in dygraph.
    """
def cross_entropy_with_softmax_(input: paddle.Tensor, label: paddle.Tensor, soft_label: bool = False, use_softmax: bool = True, numeric_stable_mode: bool = True, ignore_index: int = -100, axis: int = -1) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for cross_entropy_with_softmax_ in dygraph.
    """
def cross_entropy_with_softmax_grad(*args, **kwargs):
    """
    C++ interface function for cross_entropy_with_softmax_grad in dygraph.
    """
def cross_entropy_with_softmax_grad_(*args, **kwargs):
    """
    C++ interface function for cross_entropy_with_softmax_grad_ in dygraph.
    """
def cross_grad(*args, **kwargs):
    """
    C++ interface function for cross_grad in dygraph.
    """
def ctc_align(input: paddle.Tensor, input_length: paddle.Tensor, blank: int = 0, merge_repeated: bool = True, padding_value: int = 0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for ctc_align in dygraph.
    """
def cudnn_lstm(x: paddle.Tensor, init_h: paddle.Tensor, init_c: paddle.Tensor, w: paddle.Tensor, weight_list: list[paddle.Tensor], sequence_length: paddle.Tensor, dropout_prob: float = 0.0, is_bidirec: bool = False, hidden_size: int = 100, num_layers: int = 1, is_test: bool = False, seed: int = 0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for cudnn_lstm in dygraph.
    """
def cudnn_lstm_grad(*args, **kwargs):
    """
    C++ interface function for cudnn_lstm_grad in dygraph.
    """
def cummax(x: paddle.Tensor, axis: int = -1, dtype: paddle._typing.DTypeLike = "DataType::INT64") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for cummax in dygraph.
    """
def cummax_grad(*args, **kwargs):
    """
    C++ interface function for cummax_grad in dygraph.
    """
def cummin(x: paddle.Tensor, axis: int = -1, dtype: paddle._typing.DTypeLike = "DataType::INT64") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for cummin in dygraph.
    """
def cummin_grad(*args, **kwargs):
    """
    C++ interface function for cummin_grad in dygraph.
    """
def cumprod(x: paddle.Tensor, dim: int, exclusive: bool = False, reverse: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cumprod in dygraph.
    """
def cumprod_(x: paddle.Tensor, dim: int, exclusive: bool = False, reverse: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cumprod_ in dygraph.
    """
def cumprod_grad(*args, **kwargs):
    """
    C++ interface function for cumprod_grad in dygraph.
    """
def cumsum(x: paddle.Tensor, axis: float = -1, flatten: bool = False, exclusive: bool = False, reverse: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cumsum in dygraph.
    """
def cumsum_(x: paddle.Tensor, axis: float = -1, flatten: bool = False, exclusive: bool = False, reverse: bool = False) -> paddle.Tensor:
    """
    C++ interface function for cumsum_ in dygraph.
    """
def cumsum_grad(*args, **kwargs):
    """
    C++ interface function for cumsum_grad in dygraph.
    """
def cvm(x: paddle.Tensor, cvm: paddle.Tensor, use_cvm: bool = True) -> paddle.Tensor:
    """
    C++ interface function for cvm in dygraph.
    """
def cvm_grad(*args, **kwargs):
    """
    C++ interface function for cvm_grad in dygraph.
    """
def data(name: str, shape: list[int], dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike) -> paddle.Tensor:
    """
    C++ interface function for data in dygraph.
    """
def decayed_adagrad(param: paddle.Tensor, grad: paddle.Tensor, moment: paddle.Tensor, learning_rate: paddle.Tensor, decay: float = 0.95, epsilon: float = 1.0e-6) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for decayed_adagrad in dygraph.
    """
def decode_jpeg(x: paddle.Tensor, mode: str, place: paddle._typing.PlaceLike) -> paddle.Tensor:
    """
    C++ interface function for decode_jpeg in dygraph.
    """
def deformable_conv(x: paddle.Tensor, offset: paddle.Tensor, filter: paddle.Tensor, mask: paddle.Tensor, strides: list[int], paddings: list[int], dilations: list[int], deformable_groups: int, groups: int, im2col_step: int) -> paddle.Tensor:
    """
    C++ interface function for deformable_conv in dygraph.
    """
def deformable_conv_grad(*args, **kwargs):
    """
    C++ interface function for deformable_conv_grad in dygraph.
    """
def depend(x: paddle.Tensor, dep: list[paddle.Tensor]) -> paddle.Tensor:
    """
    C++ interface function for depend in dygraph.
    """
def depthwise_conv2d(input: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1], paddings: list[int] = [0, 0], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1], data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for depthwise_conv2d in dygraph.
    """
def depthwise_conv2d_grad(*args, **kwargs):
    """
    C++ interface function for depthwise_conv2d_grad in dygraph.
    """
def depthwise_conv2d_transpose(x: paddle.Tensor, filter: paddle.Tensor, strides: list[int] = [1, 1], paddings: list[int] = [0, 0], output_padding: list[int] = [], output_size: list[int] = [], padding_algorithm: str = "EXPLICIT", groups: int = 1, dilations: list[int] = [1, 1], data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for depthwise_conv2d_transpose in dygraph.
    """
def depthwise_conv2d_transpose_grad(*args, **kwargs):
    """
    C++ interface function for depthwise_conv2d_transpose_grad in dygraph.
    """
def dequantize_abs_max(x: paddle.Tensor, scale: paddle.Tensor, max_range: float) -> paddle.Tensor:
    """
    C++ interface function for dequantize_abs_max in dygraph.
    """
def dequantize_log(x: paddle.Tensor, dict: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for dequantize_log in dygraph.
    """
def det(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for det in dygraph.
    """
def det_grad(*args, **kwargs):
    """
    C++ interface function for det_grad in dygraph.
    """
def dgc(u: paddle.Tensor, v: paddle.Tensor, grad: paddle.Tensor, param: paddle.Tensor, current_step: paddle.Tensor, nranks: paddle.Tensor, m: float = 0.9, use_nesterov: bool = True, sparsity: list[float] = [], rampup_begin_step: float = 0.0, rampup_step: float = 0.0, regular_coeff: float = 0.0, regular_type: int = 0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for dgc in dygraph.
    """
def dgc_clip_by_norm(x: paddle.Tensor, current_step: paddle.Tensor, max_norm: float, rampup_begin_step: float = -1.0) -> paddle.Tensor:
    """
    C++ interface function for dgc_clip_by_norm in dygraph.
    """
def dgc_momentum(param: paddle.Tensor, grad: paddle.Tensor, velocity: paddle.Tensor, learning_rate: paddle.Tensor, master_param: paddle.Tensor, current_step_tensor: paddle.Tensor, nranks_tensor: paddle.Tensor, mu: float, use_nesterov: bool = False, regularization_method: str = "", regularization_coeff: float = 0.0, multi_precision: bool = False, rescale_grad: float = 1.0, rampup_begin_step: float = -1.0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for dgc_momentum in dygraph.
    """
def diag(x: paddle.Tensor, offset: int = 0, padding_value: float = 0.0) -> paddle.Tensor:
    """
    C++ interface function for diag in dygraph.
    """
def diag_embed(input: paddle.Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> paddle.Tensor:
    """
    C++ interface function for diag_embed in dygraph.
    """
def diag_grad(*args, **kwargs):
    """
    C++ interface function for diag_grad in dygraph.
    """
def diagonal(x: paddle.Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> paddle.Tensor:
    """
    C++ interface function for diagonal in dygraph.
    """
def diagonal_grad(*args, **kwargs):
    """
    C++ interface function for diagonal_grad in dygraph.
    """
def digamma(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for digamma in dygraph.
    """
def digamma_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for digamma_ in dygraph.
    """
def digamma_grad(*args, **kwargs):
    """
    C++ interface function for digamma_grad in dygraph.
    """
def dirichlet(alpha: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for dirichlet in dygraph.
    """
def disable_check_model_nan_inf(x: paddle.Tensor, flag: int = 0) -> paddle.Tensor:
    """
    C++ interface function for disable_check_model_nan_inf in dygraph.
    """
def disable_check_model_nan_inf_grad(*args, **kwargs):
    """
    C++ interface function for disable_check_model_nan_inf_grad in dygraph.
    """
def dist(x: paddle.Tensor, y: paddle.Tensor, p: float = 2.0) -> paddle.Tensor:
    """
    C++ interface function for dist in dygraph.
    """
def dist_grad(*args, **kwargs):
    """
    C++ interface function for dist_grad in dygraph.
    """
def distribute_fpn_proposals(*args, **kwargs):
    """
    C++ interface function for distribute_fpn_proposals in dygraph.
    """
def divide(*args, **kwargs):
    """
    C++ interface function for divide in dygraph.
    """
def divide_(*args, **kwargs):
    """
    C++ interface function for divide_ in dygraph.
    """
def divide_grad(*args, **kwargs):
    """
    C++ interface function for divide_grad in dygraph.
    """
def dot(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for dot in dygraph.
    """
def dot_grad(*args, **kwargs):
    """
    C++ interface function for dot_grad in dygraph.
    """
def dpsgd(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, clip: float = 10.0, batch_size: float = 16.0, sigma: float = 1.0, seed: int = 0) -> paddle.Tensor:
    """
    C++ interface function for dpsgd in dygraph.
    """
def dropout(x: paddle.Tensor, seed_tensor: paddle.Tensor, p: float, is_test: bool, mode: str, seed: int, fix_seed: bool) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for dropout in dygraph.
    """
def dropout_grad(*args, **kwargs):
    """
    C++ interface function for dropout_grad in dygraph.
    """
def edit_distance(hyps: paddle.Tensor, refs: paddle.Tensor, hypslength: paddle.Tensor, refslength: paddle.Tensor, normalized: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for edit_distance in dygraph.
    """
def eig(x: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for eig in dygraph.
    """
def eig_grad(*args, **kwargs):
    """
    C++ interface function for eig_grad in dygraph.
    """
def eigh(x: paddle.Tensor, UPLO: str = "L") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for eigh in dygraph.
    """
def eigh_grad(*args, **kwargs):
    """
    C++ interface function for eigh_grad in dygraph.
    """
def eigvals(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for eigvals in dygraph.
    """
def eigvalsh(x: paddle.Tensor, uplo: str = "L", is_test: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for eigvalsh in dygraph.
    """
def eigvalsh_grad(*args, **kwargs):
    """
    C++ interface function for eigvalsh_grad in dygraph.
    """
def einsum(*args, **kwargs):
    """
    C++ interface function for einsum in dygraph.
    """
def einsum_grad(*args, **kwargs):
    """
    C++ interface function for einsum_grad in dygraph.
    """
def elementwise_pow(*args, **kwargs):
    """
    C++ interface function for elementwise_pow in dygraph.
    """
def elementwise_pow_grad(*args, **kwargs):
    """
    C++ interface function for elementwise_pow_grad in dygraph.
    """
def elu(x: paddle.Tensor, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for elu in dygraph.
    """
def elu_(x: paddle.Tensor, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for elu_ in dygraph.
    """
def elu_grad(*args, **kwargs):
    """
    C++ interface function for elu_grad in dygraph.
    """
def elu_grad_(*args, **kwargs):
    """
    C++ interface function for elu_grad_ in dygraph.
    """
def embedding(*args, **kwargs):
    """
    C++ interface function for embedding in dygraph.
    """
def embedding_grad(*args, **kwargs):
    """
    C++ interface function for embedding_grad in dygraph.
    """
def embedding_grad_dense(*args, **kwargs):
    """
    C++ interface function for embedding_grad_dense in dygraph.
    """
def embedding_with_scaled_gradient(x: paddle.Tensor, weight: paddle.Tensor, padding_idx: int = -1) -> paddle.Tensor:
    """
    C++ interface function for embedding_with_scaled_gradient in dygraph.
    """
def embedding_with_scaled_gradient_grad(*args, **kwargs):
    """
    C++ interface function for embedding_with_scaled_gradient_grad in dygraph.
    """
def empty(shape: list[int], dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for empty in dygraph.
    """
def empty_like(x: paddle.Tensor, dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for empty_like in dygraph.
    """
def enable_check_model_nan_inf(x: paddle.Tensor, flag: int = 1) -> paddle.Tensor:
    """
    C++ interface function for enable_check_model_nan_inf in dygraph.
    """
def enable_check_model_nan_inf_grad(*args, **kwargs):
    """
    C++ interface function for enable_check_model_nan_inf_grad in dygraph.
    """
def equal(*args, **kwargs):
    """
    C++ interface function for equal in dygraph.
    """
def equal_(*args, **kwargs):
    """
    C++ interface function for equal_ in dygraph.
    """
def equal_all(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for equal_all in dygraph.
    """
def erf(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for erf in dygraph.
    """
def erf_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for erf_ in dygraph.
    """
def erf_grad(*args, **kwargs):
    """
    C++ interface function for erf_grad in dygraph.
    """
def erfinv(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for erfinv in dygraph.
    """
def erfinv_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for erfinv_ in dygraph.
    """
def erfinv_grad(*args, **kwargs):
    """
    C++ interface function for erfinv_grad in dygraph.
    """
def exp(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for exp in dygraph.
    """
def exp_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for exp_ in dygraph.
    """
def exp_grad(*args, **kwargs):
    """
    C++ interface function for exp_grad in dygraph.
    """
def exp_grad_(*args, **kwargs):
    """
    C++ interface function for exp_grad_ in dygraph.
    """
def expand(x: paddle.Tensor, shape: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for expand in dygraph.
    """
def expand_as(x: paddle.Tensor, y: paddle.Tensor, target_shape: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for expand_as in dygraph.
    """
def expand_as_grad(*args, **kwargs):
    """
    C++ interface function for expand_as_grad in dygraph.
    """
def expand_grad(*args, **kwargs):
    """
    C++ interface function for expand_grad in dygraph.
    """
def expm1(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for expm1 in dygraph.
    """
def expm1_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for expm1_ in dygraph.
    """
def expm1_grad(*args, **kwargs):
    """
    C++ interface function for expm1_grad in dygraph.
    """
def expm1_grad_(*args, **kwargs):
    """
    C++ interface function for expm1_grad_ in dygraph.
    """
def exponential_(x: paddle.Tensor, lam: float) -> paddle.Tensor:
    """
    C++ interface function for exponential_ in dygraph.
    """
def exponential__grad(*args, **kwargs):
    """
    C++ interface function for exponential__grad in dygraph.
    """
def eye(num_rows: float, num_columns: float, dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for eye in dygraph.
    """
def fake_channel_wise_dequantize_max_abs(x: paddle.Tensor, scales: list[paddle.Tensor], quant_bits: list[int] = [8], quant_axis: int = 0, x_num_col_dims: int = 1) -> paddle.Tensor:
    """
    C++ interface function for fake_channel_wise_dequantize_max_abs in dygraph.
    """
def fake_channel_wise_quantize_abs_max(x: paddle.Tensor, bit_length: int = 8, round_type: int = 1, quant_axis: int = 0, is_test: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_channel_wise_quantize_abs_max in dygraph.
    """
def fake_channel_wise_quantize_dequantize_abs_max(x: paddle.Tensor, bit_length: int = 8, round_type: int = 1, quant_axis: int = 0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_channel_wise_quantize_dequantize_abs_max in dygraph.
    """
def fake_channel_wise_quantize_dequantize_abs_max_grad(*args, **kwargs):
    """
    C++ interface function for fake_channel_wise_quantize_dequantize_abs_max_grad in dygraph.
    """
def fake_dequantize_max_abs(x: paddle.Tensor, scale: paddle.Tensor, max_range: float) -> paddle.Tensor:
    """
    C++ interface function for fake_dequantize_max_abs in dygraph.
    """
def fake_quantize_abs_max(x: paddle.Tensor, bit_length: int = 8, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_abs_max in dygraph.
    """
def fake_quantize_dequantize_abs_max(x: paddle.Tensor, bit_length: int = 8, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_dequantize_abs_max in dygraph.
    """
def fake_quantize_dequantize_abs_max_grad(*args, **kwargs):
    """
    C++ interface function for fake_quantize_dequantize_abs_max_grad in dygraph.
    """
def fake_quantize_dequantize_moving_average_abs_max(x: paddle.Tensor, in_scale: paddle.Tensor, in_accum: paddle.Tensor, in_state: paddle.Tensor, moving_rate: float = 0.9, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_dequantize_moving_average_abs_max in dygraph.
    """
def fake_quantize_dequantize_moving_average_abs_max_(x: paddle.Tensor, in_scale: paddle.Tensor, in_accum: paddle.Tensor, in_state: paddle.Tensor, moving_rate: float = 0.9, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_dequantize_moving_average_abs_max_ in dygraph.
    """
def fake_quantize_dequantize_moving_average_abs_max_grad(*args, **kwargs):
    """
    C++ interface function for fake_quantize_dequantize_moving_average_abs_max_grad in dygraph.
    """
def fake_quantize_moving_average_abs_max(x: paddle.Tensor, in_scale: paddle.Tensor, in_accum: paddle.Tensor, in_state: paddle.Tensor, moving_rate: float = 0.9, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_moving_average_abs_max in dygraph.
    """
def fake_quantize_moving_average_abs_max_(x: paddle.Tensor, in_scale: paddle.Tensor, in_accum: paddle.Tensor, in_state: paddle.Tensor, moving_rate: float = 0.9, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_moving_average_abs_max_ in dygraph.
    """
def fake_quantize_range_abs_max(x: paddle.Tensor, in_scale: paddle.Tensor, iter: paddle.Tensor, window_size: int = 10000, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_range_abs_max in dygraph.
    """
def fake_quantize_range_abs_max_(x: paddle.Tensor, in_scale: paddle.Tensor, iter: paddle.Tensor, window_size: int = 10000, bit_length: int = 8, is_test: bool = False, round_type: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fake_quantize_range_abs_max_ in dygraph.
    """
def fft_c2c(x: paddle.Tensor, axes: list[int], normalization: str, forward: bool) -> paddle.Tensor:
    """
    C++ interface function for fft_c2c in dygraph.
    """
def fft_c2c_grad(*args, **kwargs):
    """
    C++ interface function for fft_c2c_grad in dygraph.
    """
def fft_c2r(x: paddle.Tensor, axes: list[int], normalization: str, forward: bool, last_dim_size: int = 0) -> paddle.Tensor:
    """
    C++ interface function for fft_c2r in dygraph.
    """
def fft_c2r_grad(*args, **kwargs):
    """
    C++ interface function for fft_c2r_grad in dygraph.
    """
def fft_r2c(x: paddle.Tensor, axes: list[int], normalization: str, forward: bool, onesided: bool) -> paddle.Tensor:
    """
    C++ interface function for fft_r2c in dygraph.
    """
def fft_r2c_grad(*args, **kwargs):
    """
    C++ interface function for fft_r2c_grad in dygraph.
    """
def fill(x: paddle.Tensor, value: float = 0) -> paddle.Tensor:
    """
    C++ interface function for fill in dygraph.
    """
def fill_(x: paddle.Tensor, value: float = 0) -> paddle.Tensor:
    """
    C++ interface function for fill_ in dygraph.
    """
def fill_diagonal(x: paddle.Tensor, value: float = 0, offset: int = 0, wrap: bool = False) -> paddle.Tensor:
    """
    C++ interface function for fill_diagonal in dygraph.
    """
def fill_diagonal_(x: paddle.Tensor, value: float = 0, offset: int = 0, wrap: bool = False) -> paddle.Tensor:
    """
    C++ interface function for fill_diagonal_ in dygraph.
    """
def fill_diagonal_grad(*args, **kwargs):
    """
    C++ interface function for fill_diagonal_grad in dygraph.
    """
def fill_diagonal_tensor(x: paddle.Tensor, y: paddle.Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> paddle.Tensor:
    """
    C++ interface function for fill_diagonal_tensor in dygraph.
    """
def fill_diagonal_tensor_(x: paddle.Tensor, y: paddle.Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> paddle.Tensor:
    """
    C++ interface function for fill_diagonal_tensor_ in dygraph.
    """
def fill_diagonal_tensor_grad(*args, **kwargs):
    """
    C++ interface function for fill_diagonal_tensor_grad in dygraph.
    """
def fill_diagonal_tensor_grad_(*args, **kwargs):
    """
    C++ interface function for fill_diagonal_tensor_grad_ in dygraph.
    """
def fill_grad(*args, **kwargs):
    """
    C++ interface function for fill_grad in dygraph.
    """
def fill_grad_(*args, **kwargs):
    """
    C++ interface function for fill_grad_ in dygraph.
    """
def flash_attn(q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, fixed_seed_offset: paddle.Tensor, attn_mask: paddle.Tensor, dropout: float = 0.0, causal: bool = False, return_softmax: bool = False, is_test: bool = False, rng_name: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for flash_attn in dygraph.
    """
def flash_attn_grad(*args, **kwargs):
    """
    C++ interface function for flash_attn_grad in dygraph.
    """
def flash_attn_qkvpacked(qkv: paddle.Tensor, fixed_seed_offset: paddle.Tensor, attn_mask: paddle.Tensor, dropout: float = 0.0, causal: bool = False, return_softmax: bool = False, is_test: bool = False, rng_name: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for flash_attn_qkvpacked in dygraph.
    """
def flash_attn_qkvpacked_grad(*args, **kwargs):
    """
    C++ interface function for flash_attn_qkvpacked_grad in dygraph.
    """
def flash_attn_unpadded(q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, cu_seqlens_q: paddle.Tensor, cu_seqlens_k: paddle.Tensor, fixed_seed_offset: paddle.Tensor, attn_mask: paddle.Tensor, max_seqlen_q: float, max_seqlen_k: float, scale: float, dropout: float = 0.0, causal: bool = False, return_softmax: bool = False, is_test: bool = False, rng_name: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for flash_attn_unpadded in dygraph.
    """
def flash_attn_unpadded_grad(*args, **kwargs):
    """
    C++ interface function for flash_attn_unpadded_grad in dygraph.
    """
def flash_attn_varlen_qkvpacked(qkv: paddle.Tensor, cu_seqlens_q: paddle.Tensor, cu_seqlens_k: paddle.Tensor, fixed_seed_offset: paddle.Tensor, attn_mask: paddle.Tensor, max_seqlen_q: float, max_seqlen_k: float, scale: float, dropout: float = 0.0, causal: bool = False, return_softmax: bool = False, is_test: bool = False, rng_name: str = "", varlen_padded: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for flash_attn_varlen_qkvpacked in dygraph.
    """
def flash_attn_varlen_qkvpacked_grad(*args, **kwargs):
    """
    C++ interface function for flash_attn_varlen_qkvpacked_grad in dygraph.
    """
def flashmask_attention(q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, startend_row_indices: paddle.Tensor, fixed_seed_offset: paddle.Tensor, dropout: float = 0.0, causal: bool = False, return_softmax: bool = False, is_test: bool = False, rng_name: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for flashmask_attention in dygraph.
    """
def flashmask_attention_grad(*args, **kwargs):
    """
    C++ interface function for flashmask_attention_grad in dygraph.
    """
def flatten(x: paddle.Tensor, start_axis: int = 1, stop_axis: int = 1) -> paddle.Tensor:
    """
    C++ interface function for flatten in dygraph.
    """
def flatten_(x: paddle.Tensor, start_axis: int = 1, stop_axis: int = 1) -> paddle.Tensor:
    """
    C++ interface function for flatten_ in dygraph.
    """
def flatten_grad(*args, **kwargs):
    """
    C++ interface function for flatten_grad in dygraph.
    """
def flatten_grad_(*args, **kwargs):
    """
    C++ interface function for flatten_grad_ in dygraph.
    """
def flip(x: paddle.Tensor, axis: list[int]) -> paddle.Tensor:
    """
    C++ interface function for flip in dygraph.
    """
def flip_grad(*args, **kwargs):
    """
    C++ interface function for flip_grad in dygraph.
    """
def floor(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for floor in dygraph.
    """
def floor_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for floor_ in dygraph.
    """
def floor_divide(*args, **kwargs):
    """
    C++ interface function for floor_divide in dygraph.
    """
def floor_divide_(*args, **kwargs):
    """
    C++ interface function for floor_divide_ in dygraph.
    """
def floor_grad(*args, **kwargs):
    """
    C++ interface function for floor_grad in dygraph.
    """
def floor_grad_(*args, **kwargs):
    """
    C++ interface function for floor_grad_ in dygraph.
    """
def fmax(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for fmax in dygraph.
    """
def fmax_grad(*args, **kwargs):
    """
    C++ interface function for fmax_grad in dygraph.
    """
def fmin(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for fmin in dygraph.
    """
def fmin_grad(*args, **kwargs):
    """
    C++ interface function for fmin_grad in dygraph.
    """
def fold(x: paddle.Tensor, output_sizes: list[int], kernel_sizes: list[int], strides: list[int], paddings: list[int], dilations: list[int]) -> paddle.Tensor:
    """
    C++ interface function for fold in dygraph.
    """
def fold_grad(*args, **kwargs):
    """
    C++ interface function for fold_grad in dygraph.
    """
def fp8_fp8_half_gemm_fused(*args, **kwargs):
    """
    C++ interface function for fp8_fp8_half_gemm_fused in dygraph.
    """
def fractional_max_pool2d(x: paddle.Tensor, output_size: list[int], kernel_size: list[int] = [0, 0], random_u: float = 0.0, return_mask: bool = True) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fractional_max_pool2d in dygraph.
    """
def fractional_max_pool2d_grad(*args, **kwargs):
    """
    C++ interface function for fractional_max_pool2d_grad in dygraph.
    """
def fractional_max_pool3d(x: paddle.Tensor, output_size: list[int], kernel_size: list[int] = [0, 0, 0], random_u: float = 0.0, return_mask: bool = True) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fractional_max_pool3d in dygraph.
    """
def fractional_max_pool3d_grad(*args, **kwargs):
    """
    C++ interface function for fractional_max_pool3d_grad in dygraph.
    """
def frame(x: paddle.Tensor, frame_length: int, hop_length: int, axis: int = -1) -> paddle.Tensor:
    """
    C++ interface function for frame in dygraph.
    """
def frame_grad(*args, **kwargs):
    """
    C++ interface function for frame_grad in dygraph.
    """
def frobenius_norm(x: paddle.Tensor, axis: list[int], keep_dim: bool, reduce_all: bool) -> paddle.Tensor:
    """
    C++ interface function for frobenius_norm in dygraph.
    """
def frobenius_norm_grad(*args, **kwargs):
    """
    C++ interface function for frobenius_norm_grad in dygraph.
    """
def ftrl(param: paddle.Tensor, squared_accumulator: paddle.Tensor, linear_accumulator: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, l1: float = 0.0, l2: float = 0.0, lr_power: float = -0.5) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for ftrl in dygraph.
    """
def full(shape: list[int], value: float, dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for full in dygraph.
    """
def full_(output: paddle.Tensor, shape: list[int], value: float, dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for full_ in dygraph.
    """
def full_batch_size_like(input: paddle.Tensor, shape: list[int], dtype: paddle._typing.DTypeLike, value: float, input_dim_idx: int, output_dim_idx: int, place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for full_batch_size_like in dygraph.
    """
def full_int_array(value: list[int], dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for full_int_array in dygraph.
    """
def full_like(x: paddle.Tensor, value: float, dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for full_like in dygraph.
    """
def full_with_tensor(value: paddle.Tensor, shape: list[int], dtype: paddle._typing.DTypeLike = "DataType::FLOAT32") -> paddle.Tensor:
    """
    C++ interface function for full_with_tensor in dygraph.
    """
def fused_adam_(*args, **kwargs):
    """
    C++ interface function for fused_adam_ in dygraph.
    """
def fused_batch_norm_act(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, mean: paddle.Tensor, variance: paddle.Tensor, momentum: float, epsilon: float, act_type: str) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fused_batch_norm_act in dygraph.
    """
def fused_batch_norm_act_grad(*args, **kwargs):
    """
    C++ interface function for fused_batch_norm_act_grad in dygraph.
    """
def fused_bias_act(*args, **kwargs):
    """
    C++ interface function for fused_bias_act in dygraph.
    """
def fused_bias_dropout_residual_layer_norm(*args, **kwargs):
    """
    C++ interface function for fused_bias_dropout_residual_layer_norm in dygraph.
    """
def fused_bias_dropout_residual_layer_norm_grad(*args, **kwargs):
    """
    C++ interface function for fused_bias_dropout_residual_layer_norm_grad in dygraph.
    """
def fused_bias_residual_layernorm(*args, **kwargs):
    """
    C++ interface function for fused_bias_residual_layernorm in dygraph.
    """
def fused_bn_add_activation(x: paddle.Tensor, z: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, mean: paddle.Tensor, variance: paddle.Tensor, momentum: float, epsilon: float, act_type: str) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fused_bn_add_activation in dygraph.
    """
def fused_bn_add_activation_grad(*args, **kwargs):
    """
    C++ interface function for fused_bn_add_activation_grad in dygraph.
    """
def fused_dot_product_attention(*args, **kwargs):
    """
    C++ interface function for fused_dot_product_attention in dygraph.
    """
def fused_dot_product_attention_grad(*args, **kwargs):
    """
    C++ interface function for fused_dot_product_attention_grad in dygraph.
    """
def fused_dropout_add(*args, **kwargs):
    """
    C++ interface function for fused_dropout_add in dygraph.
    """
def fused_dropout_add_grad(*args, **kwargs):
    """
    C++ interface function for fused_dropout_add_grad in dygraph.
    """
def fused_elementwise_add(*args, **kwargs):
    """
    C++ interface function for fused_elementwise_add in dygraph.
    """
def fused_elementwise_div(*args, **kwargs):
    """
    C++ interface function for fused_elementwise_div in dygraph.
    """
def fused_elementwise_mul(*args, **kwargs):
    """
    C++ interface function for fused_elementwise_mul in dygraph.
    """
def fused_elementwise_sub(*args, **kwargs):
    """
    C++ interface function for fused_elementwise_sub in dygraph.
    """
def fused_gemm_epilogue(*args, **kwargs):
    """
    C++ interface function for fused_gemm_epilogue in dygraph.
    """
def fused_gemm_epilogue_grad(*args, **kwargs):
    """
    C++ interface function for fused_gemm_epilogue_grad in dygraph.
    """
def fused_linear_param_grad_add(*args, **kwargs):
    """
    C++ interface function for fused_linear_param_grad_add in dygraph.
    """
def fused_moe(*args, **kwargs):
    """
    C++ interface function for fused_moe in dygraph.
    """
def fused_multi_transformer_(*args, **kwargs):
    """
    C++ interface function for fused_multi_transformer_ in dygraph.
    """
def fused_rotary_position_embedding(*args, **kwargs):
    """
    C++ interface function for fused_rotary_position_embedding in dygraph.
    """
def fused_rotary_position_embedding_grad(*args, **kwargs):
    """
    C++ interface function for fused_rotary_position_embedding_grad in dygraph.
    """
def fused_softmax_mask(x: paddle.Tensor, mask: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for fused_softmax_mask in dygraph.
    """
def fused_softmax_mask_grad(*args, **kwargs):
    """
    C++ interface function for fused_softmax_mask_grad in dygraph.
    """
def fused_softmax_mask_upper_triangle(X: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for fused_softmax_mask_upper_triangle in dygraph.
    """
def fused_softmax_mask_upper_triangle_grad(*args, **kwargs):
    """
    C++ interface function for fused_softmax_mask_upper_triangle_grad in dygraph.
    """
def fused_token_prune(*args, **kwargs):
    """
    C++ interface function for fused_token_prune in dygraph.
    """
def fusion_seqpool_concat(*args, **kwargs):
    """
    C++ interface function for fusion_seqpool_concat in dygraph.
    """
def fusion_seqpool_cvm_concat(*args, **kwargs):
    """
    C++ interface function for fusion_seqpool_cvm_concat in dygraph.
    """
def gammaincc(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gammaincc in dygraph.
    """
def gammaincc_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gammaincc_ in dygraph.
    """
def gammaincc_grad(*args, **kwargs):
    """
    C++ interface function for gammaincc_grad in dygraph.
    """
def gammaln(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gammaln in dygraph.
    """
def gammaln_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gammaln_ in dygraph.
    """
def gammaln_grad(*args, **kwargs):
    """
    C++ interface function for gammaln_grad in dygraph.
    """
def gather(x: paddle.Tensor, index: paddle.Tensor, axis: float = 0) -> paddle.Tensor:
    """
    C++ interface function for gather in dygraph.
    """
def gather_grad(*args, **kwargs):
    """
    C++ interface function for gather_grad in dygraph.
    """
def gather_nd(x: paddle.Tensor, index: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gather_nd in dygraph.
    """
def gather_nd_grad(*args, **kwargs):
    """
    C++ interface function for gather_nd_grad in dygraph.
    """
def gather_tree(ids: paddle.Tensor, parents: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for gather_tree in dygraph.
    """
def gaussian(shape: list[int], mean: float, std: float, seed: int, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for gaussian in dygraph.
    """
def gaussian_inplace(x: paddle.Tensor, mean: float = 0, std: float = 1.0, seed: int = 0) -> paddle.Tensor:
    """
    C++ interface function for gaussian_inplace in dygraph.
    """
def gaussian_inplace_(x: paddle.Tensor, mean: float = 0, std: float = 1.0, seed: int = 0) -> paddle.Tensor:
    """
    C++ interface function for gaussian_inplace_ in dygraph.
    """
def gaussian_inplace_grad(*args, **kwargs):
    """
    C++ interface function for gaussian_inplace_grad in dygraph.
    """
def gaussian_inplace_grad_(*args, **kwargs):
    """
    C++ interface function for gaussian_inplace_grad_ in dygraph.
    """
def gelu(x: paddle.Tensor, approximate: bool = False) -> paddle.Tensor:
    """
    C++ interface function for gelu in dygraph.
    """
def gelu_grad(*args, **kwargs):
    """
    C++ interface function for gelu_grad in dygraph.
    """
def gemm_epilogue(*args, **kwargs):
    """
    C++ interface function for gemm_epilogue in dygraph.
    """
def generate_proposals(scores: paddle.Tensor, bbox_deltas: paddle.Tensor, im_shape: paddle.Tensor, anchors: paddle.Tensor, variances: paddle.Tensor, pre_nms_top_n: int, post_nms_top_n: int, nms_thresh: float, min_size: float, eta: float, pixel_offset: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for generate_proposals in dygraph.
    """
def get_core_ops_args_info(*args, **kwargs):
    """
    C++ interface function for eager_get_core_ops_args_info.
    """
def get_core_ops_args_type_info(*args, **kwargs):
    """
    C++ interface function for eager_get_core_ops_args_type_info.
    """
def get_core_ops_returns_info(*args, **kwargs):
    """
    C++ interface function for eager_get_core_ops_returns_info.
    """
def global_gather(x: paddle.Tensor, local_count: paddle.Tensor, global_count: paddle.Tensor, ring_id: int = 0) -> paddle.Tensor:
    """
    C++ interface function for global_gather in dygraph.
    """
def global_gather_grad(*args, **kwargs):
    """
    C++ interface function for global_gather_grad in dygraph.
    """
def global_scatter(x: paddle.Tensor, local_count: paddle.Tensor, global_count: paddle.Tensor, ring_id: int = 0) -> paddle.Tensor:
    """
    C++ interface function for global_scatter in dygraph.
    """
def global_scatter_grad(*args, **kwargs):
    """
    C++ interface function for global_scatter_grad in dygraph.
    """
def graph_khop_sampler(row: paddle.Tensor, colptr: paddle.Tensor, x: paddle.Tensor, eids: paddle.Tensor, sample_sizes: list[int], return_eids: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for graph_khop_sampler in dygraph.
    """
def graph_sample_neighbors(row: paddle.Tensor, colptr: paddle.Tensor, x: paddle.Tensor, eids: paddle.Tensor, perm_buffer: paddle.Tensor, sample_size: int, return_eids: bool, flag_perm_buffer: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for graph_sample_neighbors in dygraph.
    """
def greater_equal(*args, **kwargs):
    """
    C++ interface function for greater_equal in dygraph.
    """
def greater_equal_(*args, **kwargs):
    """
    C++ interface function for greater_equal_ in dygraph.
    """
def greater_than(*args, **kwargs):
    """
    C++ interface function for greater_than in dygraph.
    """
def greater_than_(*args, **kwargs):
    """
    C++ interface function for greater_than_ in dygraph.
    """
def grid_sample(x: paddle.Tensor, grid: paddle.Tensor, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True) -> paddle.Tensor:
    """
    C++ interface function for grid_sample in dygraph.
    """
def grid_sample_grad(*args, **kwargs):
    """
    C++ interface function for grid_sample_grad in dygraph.
    """
def group_norm(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, epsilon: float = 1e-5, groups: int = -1, data_format: str = "NCHW") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for group_norm in dygraph.
    """
def group_norm_grad(*args, **kwargs):
    """
    C++ interface function for group_norm_grad in dygraph.
    """
def group_norm_grad_(*args, **kwargs):
    """
    C++ interface function for group_norm_grad_ in dygraph.
    """
def gru(input: paddle.Tensor, h0: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, activation: str = "tanh", gate_activation: str = "sigmoid", is_reverse: bool = False, origin_mode: bool = False, is_test: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for gru in dygraph.
    """
def gru_grad(*args, **kwargs):
    """
    C++ interface function for gru_grad in dygraph.
    """
def gru_unit(input: paddle.Tensor, hidden_prev: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, activation: int = 2, gate_activation: int = 1, origin_mode: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for gru_unit in dygraph.
    """
def gru_unit_grad(*args, **kwargs):
    """
    C++ interface function for gru_unit_grad in dygraph.
    """
def gumbel_softmax(x: paddle.Tensor, temperature: float = 1.0, hard: bool = False, axis: int = -1) -> paddle.Tensor:
    """
    C++ interface function for gumbel_softmax in dygraph.
    """
def gumbel_softmax_grad(*args, **kwargs):
    """
    C++ interface function for gumbel_softmax_grad in dygraph.
    """
def hardshrink(x: paddle.Tensor, threshold: float = 0.5) -> paddle.Tensor:
    """
    C++ interface function for hardshrink in dygraph.
    """
def hardshrink_grad(*args, **kwargs):
    """
    C++ interface function for hardshrink_grad in dygraph.
    """
def hardshrink_grad_(*args, **kwargs):
    """
    C++ interface function for hardshrink_grad_ in dygraph.
    """
def hardsigmoid(x: paddle.Tensor, slope: float = 0.2, offset: float = 0.5) -> paddle.Tensor:
    """
    C++ interface function for hardsigmoid in dygraph.
    """
def hardsigmoid_grad(*args, **kwargs):
    """
    C++ interface function for hardsigmoid_grad in dygraph.
    """
def hardsigmoid_grad_(*args, **kwargs):
    """
    C++ interface function for hardsigmoid_grad_ in dygraph.
    """
def hardswish(*args, **kwargs):
    """
    C++ interface function for hardswish in dygraph.
    """
def hardswish_grad(*args, **kwargs):
    """
    C++ interface function for hardswish_grad in dygraph.
    """
def hardswish_grad_(*args, **kwargs):
    """
    C++ interface function for hardswish_grad_ in dygraph.
    """
def hardtanh(x: paddle.Tensor, t_min: float = 0, t_max: float = 24) -> paddle.Tensor:
    """
    C++ interface function for hardtanh in dygraph.
    """
def hardtanh_(x: paddle.Tensor, t_min: float = 0, t_max: float = 24) -> paddle.Tensor:
    """
    C++ interface function for hardtanh_ in dygraph.
    """
def hardtanh_grad(*args, **kwargs):
    """
    C++ interface function for hardtanh_grad in dygraph.
    """
def hardtanh_grad_(*args, **kwargs):
    """
    C++ interface function for hardtanh_grad_ in dygraph.
    """
def heaviside(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for heaviside in dygraph.
    """
def heaviside_grad(*args, **kwargs):
    """
    C++ interface function for heaviside_grad in dygraph.
    """
def hinge_loss(logits: paddle.Tensor, labels: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for hinge_loss in dygraph.
    """
def hinge_loss_grad(*args, **kwargs):
    """
    C++ interface function for hinge_loss_grad in dygraph.
    """
def histogram(input: paddle.Tensor, weight: paddle.Tensor, bins: int = 100, min: float = 0.0, max: float = 0.0, density: bool = False) -> paddle.Tensor:
    """
    C++ interface function for histogram in dygraph.
    """
def hsigmoid_loss(x: paddle.Tensor, label: paddle.Tensor, w: paddle.Tensor, bias: paddle.Tensor, path: paddle.Tensor, code: paddle.Tensor, num_classes: int, is_sparse: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for hsigmoid_loss in dygraph.
    """
def hsigmoid_loss_grad(*args, **kwargs):
    """
    C++ interface function for hsigmoid_loss_grad in dygraph.
    """
def huber_loss(input: paddle.Tensor, label: paddle.Tensor, delta: float) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for huber_loss in dygraph.
    """
def huber_loss_grad(*args, **kwargs):
    """
    C++ interface function for huber_loss_grad in dygraph.
    """
def i0(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for i0 in dygraph.
    """
def i0_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for i0_ in dygraph.
    """
def i0_grad(*args, **kwargs):
    """
    C++ interface function for i0_grad in dygraph.
    """
def i0e(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for i0e in dygraph.
    """
def i0e_grad(*args, **kwargs):
    """
    C++ interface function for i0e_grad in dygraph.
    """
def i1(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for i1 in dygraph.
    """
def i1_grad(*args, **kwargs):
    """
    C++ interface function for i1_grad in dygraph.
    """
def i1e(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for i1e in dygraph.
    """
def i1e_grad(*args, **kwargs):
    """
    C++ interface function for i1e_grad in dygraph.
    """
def identity_loss(x: paddle.Tensor, reduction: int = 1) -> paddle.Tensor:
    """
    C++ interface function for identity_loss in dygraph.
    """
def identity_loss_(x: paddle.Tensor, reduction: int = 1) -> paddle.Tensor:
    """
    C++ interface function for identity_loss_ in dygraph.
    """
def identity_loss_grad(*args, **kwargs):
    """
    C++ interface function for identity_loss_grad in dygraph.
    """
def identity_loss_grad_(*args, **kwargs):
    """
    C++ interface function for identity_loss_grad_ in dygraph.
    """
def im2sequence(x: paddle.Tensor, y: paddle.Tensor, kernels: list[int], strides: list[int] = [1, 1], paddings: list[int] = [0, 0, 0, 0], out_stride: list[int] = [1, 1]) -> paddle.Tensor:
    """
    C++ interface function for im2sequence in dygraph.
    """
def im2sequence_grad(*args, **kwargs):
    """
    C++ interface function for im2sequence_grad in dygraph.
    """
def imag(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for imag in dygraph.
    """
def imag_grad(*args, **kwargs):
    """
    C++ interface function for imag_grad in dygraph.
    """
def increment(x: paddle.Tensor, value: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for increment in dygraph.
    """
def increment_(x: paddle.Tensor, value: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for increment_ in dygraph.
    """
def index_add(x: paddle.Tensor, index: paddle.Tensor, add_value: paddle.Tensor, axis: int = 0) -> paddle.Tensor:
    """
    C++ interface function for index_add in dygraph.
    """
def index_add_(x: paddle.Tensor, index: paddle.Tensor, add_value: paddle.Tensor, axis: int = 0) -> paddle.Tensor:
    """
    C++ interface function for index_add_ in dygraph.
    """
def index_add_grad(*args, **kwargs):
    """
    C++ interface function for index_add_grad in dygraph.
    """
def index_add_grad_(*args, **kwargs):
    """
    C++ interface function for index_add_grad_ in dygraph.
    """
def index_put(x: paddle.Tensor, indices: list[paddle.Tensor], value: paddle.Tensor, accumulate: bool = False) -> paddle.Tensor:
    """
    C++ interface function for index_put in dygraph.
    """
def index_put_(x: paddle.Tensor, indices: list[paddle.Tensor], value: paddle.Tensor, accumulate: bool = False) -> paddle.Tensor:
    """
    C++ interface function for index_put_ in dygraph.
    """
def index_put_grad(*args, **kwargs):
    """
    C++ interface function for index_put_grad in dygraph.
    """
def index_sample(x: paddle.Tensor, index: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for index_sample in dygraph.
    """
def index_sample_grad(*args, **kwargs):
    """
    C++ interface function for index_sample_grad in dygraph.
    """
def index_select(x: paddle.Tensor, index: paddle.Tensor, axis: int = 0) -> paddle.Tensor:
    """
    C++ interface function for index_select in dygraph.
    """
def index_select_grad(*args, **kwargs):
    """
    C++ interface function for index_select_grad in dygraph.
    """
def index_select_strided(x: paddle.Tensor, index: int, axis: int = 0) -> paddle.Tensor:
    """
    C++ interface function for index_select_strided in dygraph.
    """
def index_select_strided_grad(*args, **kwargs):
    """
    C++ interface function for index_select_strided_grad in dygraph.
    """
def instance_norm(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, epsilon: float = 1e-5) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for instance_norm in dygraph.
    """
def instance_norm_grad(*args, **kwargs):
    """
    C++ interface function for instance_norm_grad in dygraph.
    """
def inverse(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for inverse in dygraph.
    """
def inverse_grad(*args, **kwargs):
    """
    C++ interface function for inverse_grad in dygraph.
    """
def is_empty(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for is_empty in dygraph.
    """
def isclose(x: paddle.Tensor, y: paddle.Tensor, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> paddle.Tensor:
    """
    C++ interface function for isclose in dygraph.
    """
def isfinite(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for isfinite in dygraph.
    """
def isinf(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for isinf in dygraph.
    """
def isnan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for isnan in dygraph.
    """
def kldiv_loss(x: paddle.Tensor, label: paddle.Tensor, reduction: str = "mean", log_target: bool = False) -> paddle.Tensor:
    """
    C++ interface function for kldiv_loss in dygraph.
    """
def kldiv_loss_grad(*args, **kwargs):
    """
    C++ interface function for kldiv_loss_grad in dygraph.
    """
def kron(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for kron in dygraph.
    """
def kron_grad(*args, **kwargs):
    """
    C++ interface function for kron_grad in dygraph.
    """
def kthvalue(x: paddle.Tensor, k: int = 1, axis: int = -1, keepdim: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for kthvalue in dygraph.
    """
def kthvalue_grad(*args, **kwargs):
    """
    C++ interface function for kthvalue_grad in dygraph.
    """
def l1_norm(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for l1_norm in dygraph.
    """
def l1_norm_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for l1_norm_ in dygraph.
    """
def l1_norm_grad(*args, **kwargs):
    """
    C++ interface function for l1_norm_grad in dygraph.
    """
def label_smooth(label: paddle.Tensor, prior_dist: paddle.Tensor, epsilon: float = 0.0) -> paddle.Tensor:
    """
    C++ interface function for label_smooth in dygraph.
    """
def label_smooth_grad(*args, **kwargs):
    """
    C++ interface function for label_smooth_grad in dygraph.
    """
def lamb_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, moment1: paddle.Tensor, moment2: paddle.Tensor, beta1_pow: paddle.Tensor, beta2_pow: paddle.Tensor, master_param: paddle.Tensor, skip_update: paddle.Tensor, weight_decay: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-6, always_adapt: bool = False, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lamb_ in dygraph.
    """
def layer_norm(x: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, epsilon: float = 1e-5, begin_norm_axis: int = 1) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for layer_norm in dygraph.
    """
def layer_norm_grad(*args, **kwargs):
    """
    C++ interface function for layer_norm_grad in dygraph.
    """
def leaky_relu(x: paddle.Tensor, negative_slope: float = 0.02) -> paddle.Tensor:
    """
    C++ interface function for leaky_relu in dygraph.
    """
def leaky_relu_(x: paddle.Tensor, negative_slope: float = 0.02) -> paddle.Tensor:
    """
    C++ interface function for leaky_relu_ in dygraph.
    """
def leaky_relu_grad(*args, **kwargs):
    """
    C++ interface function for leaky_relu_grad in dygraph.
    """
def leaky_relu_grad_(*args, **kwargs):
    """
    C++ interface function for leaky_relu_grad_ in dygraph.
    """
def lerp(x: paddle.Tensor, y: paddle.Tensor, weight: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for lerp in dygraph.
    """
def lerp_(x: paddle.Tensor, y: paddle.Tensor, weight: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for lerp_ in dygraph.
    """
def lerp_grad(*args, **kwargs):
    """
    C++ interface function for lerp_grad in dygraph.
    """
def less_equal(*args, **kwargs):
    """
    C++ interface function for less_equal in dygraph.
    """
def less_equal_(*args, **kwargs):
    """
    C++ interface function for less_equal_ in dygraph.
    """
def less_than(*args, **kwargs):
    """
    C++ interface function for less_than in dygraph.
    """
def less_than_(*args, **kwargs):
    """
    C++ interface function for less_than_ in dygraph.
    """
def lgamma(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for lgamma in dygraph.
    """
def lgamma_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for lgamma_ in dygraph.
    """
def lgamma_grad(*args, **kwargs):
    """
    C++ interface function for lgamma_grad in dygraph.
    """
def limit_by_capacity(expert_count: paddle.Tensor, capacity: paddle.Tensor, n_worker: int) -> paddle.Tensor:
    """
    C++ interface function for limit_by_capacity in dygraph.
    """
def linear(*args, **kwargs):
    """
    C++ interface function for run_program in dygraph.
    """
def linear_interp(x: paddle.Tensor, out_size: paddle.Tensor, size_tensor: list[paddle.Tensor], scale_tensor: paddle.Tensor, data_format: str = "NCHW", out_d: int = 0, out_h: int = 0, out_w: int = 0, scale: list[float] = [], interp_method: str = "bilinear", align_corners: bool = True, align_mode: int = 1) -> paddle.Tensor:
    """
    C++ interface function for linear_interp in dygraph.
    """
def linear_interp_grad(*args, **kwargs):
    """
    C++ interface function for linear_interp_grad in dygraph.
    """
def linspace(start: paddle.Tensor, stop: paddle.Tensor, number: paddle.Tensor, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike) -> paddle.Tensor:
    """
    C++ interface function for linspace in dygraph.
    """
def llm_int8_linear(x: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, weight_scale: paddle.Tensor, threshold: float = 6.0) -> paddle.Tensor:
    """
    C++ interface function for llm_int8_linear in dygraph.
    """
def log(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log in dygraph.
    """
def log10(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log10 in dygraph.
    """
def log10_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log10_ in dygraph.
    """
def log10_grad(*args, **kwargs):
    """
    C++ interface function for log10_grad in dygraph.
    """
def log10_grad_(*args, **kwargs):
    """
    C++ interface function for log10_grad_ in dygraph.
    """
def log1p(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log1p in dygraph.
    """
def log1p_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log1p_ in dygraph.
    """
def log1p_grad(*args, **kwargs):
    """
    C++ interface function for log1p_grad in dygraph.
    """
def log1p_grad_(*args, **kwargs):
    """
    C++ interface function for log1p_grad_ in dygraph.
    """
def log2(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log2 in dygraph.
    """
def log2_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log2_ in dygraph.
    """
def log2_grad(*args, **kwargs):
    """
    C++ interface function for log2_grad in dygraph.
    """
def log2_grad_(*args, **kwargs):
    """
    C++ interface function for log2_grad_ in dygraph.
    """
def log_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log_ in dygraph.
    """
def log_grad(*args, **kwargs):
    """
    C++ interface function for log_grad in dygraph.
    """
def log_grad_(*args, **kwargs):
    """
    C++ interface function for log_grad_ in dygraph.
    """
def log_loss(input: paddle.Tensor, label: paddle.Tensor, epsilon: float) -> paddle.Tensor:
    """
    C++ interface function for log_loss in dygraph.
    """
def log_loss_grad(*args, **kwargs):
    """
    C++ interface function for log_loss_grad in dygraph.
    """
def log_softmax(x: paddle.Tensor, axis: int = -1) -> paddle.Tensor:
    """
    C++ interface function for log_softmax in dygraph.
    """
def log_softmax_grad(*args, **kwargs):
    """
    C++ interface function for log_softmax_grad in dygraph.
    """
def logcumsumexp(x: paddle.Tensor, axis: int = -1, flatten: bool = False, exclusive: bool = False, reverse: bool = False) -> paddle.Tensor:
    """
    C++ interface function for logcumsumexp in dygraph.
    """
def logcumsumexp_grad(*args, **kwargs):
    """
    C++ interface function for logcumsumexp_grad in dygraph.
    """
def logical_and(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_and in dygraph.
    """
def logical_and_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_and_ in dygraph.
    """
def logical_not(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_not in dygraph.
    """
def logical_not_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_not_ in dygraph.
    """
def logical_or(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_or in dygraph.
    """
def logical_or_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_or_ in dygraph.
    """
def logical_xor(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_xor in dygraph.
    """
def logical_xor_(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logical_xor_ in dygraph.
    """
def logit(x: paddle.Tensor, eps: float = 1e-6) -> paddle.Tensor:
    """
    C++ interface function for logit in dygraph.
    """
def logit_(x: paddle.Tensor, eps: float = 1e-6) -> paddle.Tensor:
    """
    C++ interface function for logit_ in dygraph.
    """
def logit_grad(*args, **kwargs):
    """
    C++ interface function for logit_grad in dygraph.
    """
def logsigmoid(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for logsigmoid in dygraph.
    """
def logsigmoid_grad(*args, **kwargs):
    """
    C++ interface function for logsigmoid_grad in dygraph.
    """
def logsigmoid_grad_(*args, **kwargs):
    """
    C++ interface function for logsigmoid_grad_ in dygraph.
    """
def logspace(start: paddle.Tensor, stop: paddle.Tensor, num: paddle.Tensor, base: paddle.Tensor, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for logspace in dygraph.
    """
def logsumexp(x: paddle.Tensor, axis: list[int] = [0], keepdim: bool = False, reduce_all: bool = False) -> paddle.Tensor:
    """
    C++ interface function for logsumexp in dygraph.
    """
def logsumexp_grad(*args, **kwargs):
    """
    C++ interface function for logsumexp_grad in dygraph.
    """
def lookup_table_dequant(w: paddle.Tensor, ids: paddle.Tensor, padding_idx: int = -1) -> paddle.Tensor:
    """
    C++ interface function for lookup_table_dequant in dygraph.
    """
def lp_pool2d(x: paddle.Tensor, kernel_size: list[int], strides: list[int] = [1,1], paddings: list[int] = [0,0], ceil_mode: bool = False, exclusive: bool = True, data_format: str = "NCHW", pooling_type: str = "", global_pooling: bool = False, adaptive: bool = False, padding_algorithm: str = "EXPLICIT", norm_type: float = 0.0) -> paddle.Tensor:
    """
    C++ interface function for lp_pool2d in dygraph.
    """
def lp_pool2d_grad(*args, **kwargs):
    """
    C++ interface function for lp_pool2d_grad in dygraph.
    """
def lstm(input: paddle.Tensor, h0: paddle.Tensor, c0: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, use_peepholes: bool = True, is_reverse: bool = False, is_test: bool = False, gate_activation: str = "sigmoid", cell_activation: str = "tanh", candidate_activation: str = "tanh") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lstm in dygraph.
    """
def lstm_grad(*args, **kwargs):
    """
    C++ interface function for lstm_grad in dygraph.
    """
def lstsq(x: paddle.Tensor, y: paddle.Tensor, rcond: float = 0.0, driver: str = "gels") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lstsq in dygraph.
    """
def lu(x: paddle.Tensor, pivot: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lu in dygraph.
    """
def lu_(x: paddle.Tensor, pivot: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lu_ in dygraph.
    """
def lu_grad(*args, **kwargs):
    """
    C++ interface function for lu_grad in dygraph.
    """
def lu_grad_(*args, **kwargs):
    """
    C++ interface function for lu_grad_ in dygraph.
    """
def lu_solve(b: paddle.Tensor, lu: paddle.Tensor, pivots: paddle.Tensor, trans: str) -> paddle.Tensor:
    """
    C++ interface function for lu_solve in dygraph.
    """
def lu_solve_grad(*args, **kwargs):
    """
    C++ interface function for lu_solve_grad in dygraph.
    """
def lu_unpack(x: paddle.Tensor, y: paddle.Tensor, unpack_ludata: bool = True, unpack_pivots: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for lu_unpack in dygraph.
    """
def lu_unpack_grad(*args, **kwargs):
    """
    C++ interface function for lu_unpack_grad in dygraph.
    """
def margin_cross_entropy(logits: paddle.Tensor, label: paddle.Tensor, return_softmax: bool = False, ring_id: int = 0, rank: int = 0, nranks: int = 1, margin1: float = 1.0, margin2: float = 0.5, margin3: float = 0.0, scale: float = 64.0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for margin_cross_entropy in dygraph.
    """
def margin_cross_entropy_grad(*args, **kwargs):
    """
    C++ interface function for margin_cross_entropy_grad in dygraph.
    """
def margin_cross_entropy_grad_(*args, **kwargs):
    """
    C++ interface function for margin_cross_entropy_grad_ in dygraph.
    """
def masked_multihead_attention_(x: paddle.Tensor, cache_kv: paddle.Tensor, bias: paddle.Tensor, src_mask: paddle.Tensor, cum_offsets: paddle.Tensor, sequence_lengths: paddle.Tensor, rotary_tensor: paddle.Tensor, beam_cache_offset: paddle.Tensor, qkv_out_scale: paddle.Tensor, out_shift: paddle.Tensor, out_smooth: paddle.Tensor, seq_len: int, rotary_emb_dims: int, use_neox_rotary_style: bool = False, compute_dtype: str = "default", out_scale: float = -1, quant_round_type: int = 1, quant_max_bound: float = 127.0, quant_min_bound: float = -127.0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for masked_multihead_attention_ in dygraph.
    """
def masked_select(x: paddle.Tensor, mask: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for masked_select in dygraph.
    """
def masked_select_grad(*args, **kwargs):
    """
    C++ interface function for masked_select_grad in dygraph.
    """
def match_matrix_tensor(x: paddle.Tensor, y: paddle.Tensor, w: paddle.Tensor, dim_t: int = 1) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for match_matrix_tensor in dygraph.
    """
def match_matrix_tensor_grad(*args, **kwargs):
    """
    C++ interface function for match_matrix_tensor_grad in dygraph.
    """
def matmul(*args, **kwargs):
    """
    C++ interface function for matmul in dygraph.
    """
def matmul_grad(*args, **kwargs):
    """
    C++ interface function for matmul_grad in dygraph.
    """
def matrix_nms(bboxes: paddle.Tensor, scores: paddle.Tensor, score_threshold: float, nms_top_k: int, keep_top_k: int, post_threshold: float = 0., use_gaussian: bool = False, gaussian_sigma: float = 2., background_label: int = 0, normalized: bool = True) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for matrix_nms in dygraph.
    """
def matrix_power(x: paddle.Tensor, n: int) -> paddle.Tensor:
    """
    C++ interface function for matrix_power in dygraph.
    """
def matrix_power_grad(*args, **kwargs):
    """
    C++ interface function for matrix_power_grad in dygraph.
    """
def matrix_rank(x: paddle.Tensor, tol: float, use_default_tol: bool = True, hermitian: bool = False) -> paddle.Tensor:
    """
    C++ interface function for matrix_rank in dygraph.
    """
def matrix_rank_atol_rtol(x: paddle.Tensor, atol: paddle.Tensor, rtol: paddle.Tensor, hermitian: bool = False) -> paddle.Tensor:
    """
    C++ interface function for matrix_rank_atol_rtol in dygraph.
    """
def matrix_rank_tol(x: paddle.Tensor, atol_tensor: paddle.Tensor, use_default_tol: bool = True, hermitian: bool = False) -> paddle.Tensor:
    """
    C++ interface function for matrix_rank_tol in dygraph.
    """
def max(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for max in dygraph.
    """
def max_grad(*args, **kwargs):
    """
    C++ interface function for max_grad in dygraph.
    """
def max_pool2d_with_index(x: paddle.Tensor, kernel_size: list[int], strides: list[int] = [1, 1], paddings: list[int] = [0, 0], global_pooling: bool = False, adaptive: bool = False, ceil_mode: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for max_pool2d_with_index in dygraph.
    """
def max_pool2d_with_index_grad(*args, **kwargs):
    """
    C++ interface function for max_pool2d_with_index_grad in dygraph.
    """
def max_pool3d_with_index(x: paddle.Tensor, kernel_size: list[int], strides: list[int] = [1, 1, 1], paddings: list[int] = [0, 0, 0], global_pooling: bool = False, adaptive: bool = False, ceil_mode: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for max_pool3d_with_index in dygraph.
    """
def max_pool3d_with_index_grad(*args, **kwargs):
    """
    C++ interface function for max_pool3d_with_index_grad in dygraph.
    """
def maximum(*args, **kwargs):
    """
    C++ interface function for maximum in dygraph.
    """
def maximum_grad(*args, **kwargs):
    """
    C++ interface function for maximum_grad in dygraph.
    """
def maxout(x: paddle.Tensor, groups: int, axis: int = 1) -> paddle.Tensor:
    """
    C++ interface function for maxout in dygraph.
    """
def maxout_grad(*args, **kwargs):
    """
    C++ interface function for maxout_grad in dygraph.
    """
def mean(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for mean in dygraph.
    """
def mean_all(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for mean_all in dygraph.
    """
def mean_all_grad(*args, **kwargs):
    """
    C++ interface function for mean_all_grad in dygraph.
    """
def mean_grad(*args, **kwargs):
    """
    C++ interface function for mean_grad in dygraph.
    """
def memcpy_d2h(x: paddle.Tensor, dst_place_type: int) -> paddle.Tensor:
    """
    C++ interface function for memcpy_d2h in dygraph.
    """
def memcpy_h2d(x: paddle.Tensor, dst_place_type: int) -> paddle.Tensor:
    """
    C++ interface function for memcpy_h2d in dygraph.
    """
def memory_efficient_attention(query: paddle.Tensor, key: paddle.Tensor, value: paddle.Tensor, bias: paddle.Tensor, cu_seqlens_q: paddle.Tensor, cu_seqlens_k: paddle.Tensor, causal_diagonal: paddle.Tensor, seqlen_k: paddle.Tensor, max_seqlen_q: float, max_seqlen_k: float, causal: bool, dropout_p: float, scale: float, is_test: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for memory_efficient_attention in dygraph.
    """
def memory_efficient_attention_grad(*args, **kwargs):
    """
    C++ interface function for memory_efficient_attention_grad in dygraph.
    """
def merge_selected_rows(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for merge_selected_rows in dygraph.
    """
def merged_adam_(param: list[paddle.Tensor], grad: list[paddle.Tensor], learning_rate: list[paddle.Tensor], moment1: list[paddle.Tensor], moment2: list[paddle.Tensor], moment2_max: list[paddle.Tensor], beta1_pow: list[paddle.Tensor], beta2_pow: list[paddle.Tensor], master_param: list[paddle.Tensor], beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, multi_precision: bool = False, use_global_beta_pow: bool = False, amsgrad: bool = False) -> tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]:
    """
    C++ interface function for merged_adam_ in dygraph.
    """
def merged_momentum_(param: list[paddle.Tensor], grad: list[paddle.Tensor], velocity: list[paddle.Tensor], learning_rate: list[paddle.Tensor], master_param: list[paddle.Tensor], mu: float, use_nesterov: bool = False, regularization_method: list[str] = [], regularization_coeff: list[float] = [], multi_precision: bool = False, rescale_grad: float = 1.0) -> tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]:
    """
    C++ interface function for merged_momentum_ in dygraph.
    """
def meshgrid(inputs: list[paddle.Tensor]) -> list[paddle.Tensor]:
    """
    C++ interface function for meshgrid in dygraph.
    """
def meshgrid_grad(*args, **kwargs):
    """
    C++ interface function for meshgrid_grad in dygraph.
    """
def min(*args, **kwargs):
    """
    C++ interface function for min in dygraph.
    """
def min_grad(*args, **kwargs):
    """
    C++ interface function for min_grad in dygraph.
    """
def minimum(*args, **kwargs):
    """
    C++ interface function for minimum in dygraph.
    """
def minimum_grad(*args, **kwargs):
    """
    C++ interface function for minimum_grad in dygraph.
    """
def mish(x: paddle.Tensor, lambda_: float) -> paddle.Tensor:
    """
    C++ interface function for mish in dygraph.
    """
def mish_grad(*args, **kwargs):
    """
    C++ interface function for mish_grad in dygraph.
    """
def mish_grad_(*args, **kwargs):
    """
    C++ interface function for mish_grad_ in dygraph.
    """
def mode(x: paddle.Tensor, axis: int = -1, keepdim: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for mode in dygraph.
    """
def mode_grad(*args, **kwargs):
    """
    C++ interface function for mode_grad in dygraph.
    """
def moe_dispatch(X: paddle.Tensor, gating_output: paddle.Tensor, moe_topk: int = 2, group_moe: bool = False, topk_only_mode: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for moe_dispatch in dygraph.
    """
def moe_ffn(permute_input: paddle.Tensor, token_nums_per_expert: paddle.Tensor, ffn1_weight: paddle.Tensor, ffn2_weight: paddle.Tensor, ffn1_bias: paddle.Tensor, ffn1_scale: paddle.Tensor, ffn2_scale: paddle.Tensor, quant_method: str) -> paddle.Tensor:
    """
    C++ interface function for moe_ffn in dygraph.
    """
def moe_reduce(ffn_out: paddle.Tensor, expert_scales_float: paddle.Tensor, permute_indices_per_token: paddle.Tensor, top_k_indices: paddle.Tensor, ffn2_bias: paddle.Tensor, norm_topk_prob: bool = False, routed_scaling_factor: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for moe_reduce in dygraph.
    """
def momentum_(param: paddle.Tensor, grad: paddle.Tensor, velocity: paddle.Tensor, learning_rate: paddle.Tensor, master_param: paddle.Tensor, mu: float, use_nesterov: bool = False, regularization_method: str = "", regularization_coeff: float = 0.0, multi_precision: bool = False, rescale_grad: float = 1.0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for momentum_ in dygraph.
    """
def mp_allreduce_sum(x: paddle.Tensor, ring_id: int = 0) -> paddle.Tensor:
    """
    C++ interface function for mp_allreduce_sum in dygraph.
    """
def mp_allreduce_sum_(x: paddle.Tensor, ring_id: int = 0) -> paddle.Tensor:
    """
    C++ interface function for mp_allreduce_sum_ in dygraph.
    """
def mp_allreduce_sum_grad(*args, **kwargs):
    """
    C++ interface function for mp_allreduce_sum_grad in dygraph.
    """
def multi_dot(x: list[paddle.Tensor]) -> paddle.Tensor:
    """
    C++ interface function for multi_dot in dygraph.
    """
def multi_dot_grad(*args, **kwargs):
    """
    C++ interface function for multi_dot_grad in dygraph.
    """
def multiclass_nms3(bboxes: paddle.Tensor, scores: paddle.Tensor, rois_num: paddle.Tensor, score_threshold: float, nms_top_k: int, keep_top_k: int, nms_threshold: float = 0.3, normalized: bool = True, nms_eta: float = 1.0, background_label: int = 0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for multiclass_nms3 in dygraph.
    """
def multinomial(x: paddle.Tensor, num_samples: int = 1, replacement: bool = False) -> paddle.Tensor:
    """
    C++ interface function for multinomial in dygraph.
    """
def multiplex(inputs: list[paddle.Tensor], index: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for multiplex in dygraph.
    """
def multiplex_grad(*args, **kwargs):
    """
    C++ interface function for multiplex_grad in dygraph.
    """
def multiply(*args, **kwargs):
    """
    C++ interface function for multiply in dygraph.
    """
def multiply_(*args, **kwargs):
    """
    C++ interface function for multiply_ in dygraph.
    """
def mv(x: paddle.Tensor, vec: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for mv in dygraph.
    """
def mv_grad(*args, **kwargs):
    """
    C++ interface function for mv_grad in dygraph.
    """
def nadam_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, momentum_decay_pow: paddle.Tensor, beta2_pow: paddle.Tensor, mu_product: paddle.Tensor, moment1: paddle.Tensor, moment2: paddle.Tensor, master_param: paddle.Tensor, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, momentum_decay: float = 0.004, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for nadam_ in dygraph.
    """
def nanmedian(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = True, mode: str = "avg") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for nanmedian in dygraph.
    """
def nanmedian_grad(*args, **kwargs):
    """
    C++ interface function for nanmedian_grad in dygraph.
    """
def nearest_interp(x: paddle.Tensor, out_size: paddle.Tensor, size_tensor: list[paddle.Tensor], scale_tensor: paddle.Tensor, data_format: str = "NCHW", out_d: int = 0, out_h: int = 0, out_w: int = 0, scale: list[float] = [], interp_method: str = "bilinear", align_corners: bool = True, align_mode: int = 1) -> paddle.Tensor:
    """
    C++ interface function for nearest_interp in dygraph.
    """
def nearest_interp_grad(*args, **kwargs):
    """
    C++ interface function for nearest_interp_grad in dygraph.
    """
def nextafter(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for nextafter in dygraph.
    """
def nll_loss(input: paddle.Tensor, label: paddle.Tensor, weight: paddle.Tensor, ignore_index: int = -100, reduction: str = "mean") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for nll_loss in dygraph.
    """
def nll_loss_grad(*args, **kwargs):
    """
    C++ interface function for nll_loss_grad in dygraph.
    """
def nms(x: paddle.Tensor, threshold: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for nms in dygraph.
    """
def nonzero(condition: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for nonzero in dygraph.
    """
def norm(x: paddle.Tensor, axis: int, epsilon: float, is_test: bool) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for norm in dygraph.
    """
def norm_grad(*args, **kwargs):
    """
    C++ interface function for norm_grad in dygraph.
    """
def not_equal(*args, **kwargs):
    """
    C++ interface function for not_equal in dygraph.
    """
def not_equal_(*args, **kwargs):
    """
    C++ interface function for not_equal_ in dygraph.
    """
def npu_identity(x: paddle.Tensor, format: int = -1) -> paddle.Tensor:
    """
    C++ interface function for npu_identity in dygraph.
    """
def number_count(numbers: paddle.Tensor, upper_range: int) -> paddle.Tensor:
    """
    C++ interface function for number_count in dygraph.
    """
def numel(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for numel in dygraph.
    """
def one_hot(x: paddle.Tensor, num_classes: int) -> paddle.Tensor:
    """
    C++ interface function for one_hot in dygraph.
    """
def ones(shape: list[int], dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for ones in dygraph.
    """
def ones_like(x: paddle.Tensor, dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for ones_like in dygraph.
    """
def overlap_add(x: paddle.Tensor, hop_length: int, axis: int = -1) -> paddle.Tensor:
    """
    C++ interface function for overlap_add in dygraph.
    """
def overlap_add_grad(*args, **kwargs):
    """
    C++ interface function for overlap_add_grad in dygraph.
    """
def p_norm(x: paddle.Tensor, porder: float = 2, axis: int = -1, epsilon: float = 1.0e-12, keepdim: bool = False, asvector: bool = False) -> paddle.Tensor:
    """
    C++ interface function for p_norm in dygraph.
    """
def p_norm_grad(*args, **kwargs):
    """
    C++ interface function for p_norm_grad in dygraph.
    """
def pad(x: paddle.Tensor, paddings: list[int], pad_value: float) -> paddle.Tensor:
    """
    C++ interface function for pad in dygraph.
    """
def pad3d(x: paddle.Tensor, paddings: list[int], mode: str = "constant", pad_value: float = 0.0, data_format: str = "NCDHW") -> paddle.Tensor:
    """
    C++ interface function for pad3d in dygraph.
    """
def pad3d_grad(*args, **kwargs):
    """
    C++ interface function for pad3d_grad in dygraph.
    """
def pad_grad(*args, **kwargs):
    """
    C++ interface function for pad_grad in dygraph.
    """
def partial_allgather(x: paddle.Tensor, nranks: int, rank: int, ring_id: int = 0) -> paddle.Tensor:
    """
    C++ interface function for partial_allgather in dygraph.
    """
def partial_allgather_(x: paddle.Tensor, nranks: int, rank: int, ring_id: int = 0) -> paddle.Tensor:
    """
    C++ interface function for partial_allgather_ in dygraph.
    """
def partial_concat(x: list[paddle.Tensor], start_index: int = 0, length: int = -1) -> paddle.Tensor:
    """
    C++ interface function for partial_concat in dygraph.
    """
def partial_concat_grad(*args, **kwargs):
    """
    C++ interface function for partial_concat_grad in dygraph.
    """
def partial_sum(x: list[paddle.Tensor], start_index: int = 0, length: int = -1) -> paddle.Tensor:
    """
    C++ interface function for partial_sum in dygraph.
    """
def partial_sum_grad(*args, **kwargs):
    """
    C++ interface function for partial_sum_grad in dygraph.
    """
def pixel_shuffle(x: paddle.Tensor, upscale_factor: int = 1, data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for pixel_shuffle in dygraph.
    """
def pixel_shuffle_grad(*args, **kwargs):
    """
    C++ interface function for pixel_shuffle_grad in dygraph.
    """
def pixel_unshuffle(x: paddle.Tensor, downscale_factor: int = 1, data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for pixel_unshuffle in dygraph.
    """
def pixel_unshuffle_grad(*args, **kwargs):
    """
    C++ interface function for pixel_unshuffle_grad in dygraph.
    """
def poisson(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for poisson in dygraph.
    """
def poisson_grad(*args, **kwargs):
    """
    C++ interface function for poisson_grad in dygraph.
    """
def polygamma(x: paddle.Tensor, n: int) -> paddle.Tensor:
    """
    C++ interface function for polygamma in dygraph.
    """
def polygamma_(x: paddle.Tensor, n: int) -> paddle.Tensor:
    """
    C++ interface function for polygamma_ in dygraph.
    """
def polygamma_grad(*args, **kwargs):
    """
    C++ interface function for polygamma_grad in dygraph.
    """
def pool2d(x: paddle.Tensor, kernel_size: list[int], strides: list[int], paddings: list[int], ceil_mode: bool, exclusive: bool, data_format: str, pooling_type: str, global_pooling: bool, adaptive: bool, padding_algorithm: str) -> paddle.Tensor:
    """
    C++ interface function for pool2d in dygraph.
    """
def pool2d_grad(*args, **kwargs):
    """
    C++ interface function for pool2d_grad in dygraph.
    """
def pool3d(x: paddle.Tensor, kernel_size: list[int], strides: list[int], paddings: list[int], ceil_mode: bool, exclusive: bool, data_format: str, pooling_type: str, global_pooling: bool, adaptive: bool, padding_algorithm: str) -> paddle.Tensor:
    """
    C++ interface function for pool3d in dygraph.
    """
def pool3d_grad(*args, **kwargs):
    """
    C++ interface function for pool3d_grad in dygraph.
    """
def pow(x: paddle.Tensor, y: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for pow in dygraph.
    """
def pow_(x: paddle.Tensor, y: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for pow_ in dygraph.
    """
def pow_grad(*args, **kwargs):
    """
    C++ interface function for pow_grad in dygraph.
    """
def pow_grad_(*args, **kwargs):
    """
    C++ interface function for pow_grad_ in dygraph.
    """
def prelu(x: paddle.Tensor, alpha: paddle.Tensor, data_format: str = "NCHW", mode: str = "all") -> paddle.Tensor:
    """
    C++ interface function for prelu in dygraph.
    """
def prelu_grad(*args, **kwargs):
    """
    C++ interface function for prelu_grad in dygraph.
    """
def prior_box(input: paddle.Tensor, image: paddle.Tensor, min_sizes: list[float], max_sizes: list[float] = [], aspect_ratios: list[float] = [], variances: list[float] = [], flip: bool = True, clip: bool = True, step_w: float = 0.0, step_h: float = 0.0, offset: float = 0.5, min_max_aspect_ratios_order: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for prior_box in dygraph.
    """
def prod(x: paddle.Tensor, axis: list[int], keepdim: bool, reduce_all: bool) -> paddle.Tensor:
    """
    C++ interface function for prod in dygraph.
    """
def prod_grad(*args, **kwargs):
    """
    C++ interface function for prod_grad in dygraph.
    """
def prune_gate_by_capacity(gate_idx: paddle.Tensor, expert_count: paddle.Tensor, n_expert: int = 0, n_worker: int = 0) -> paddle.Tensor:
    """
    C++ interface function for prune_gate_by_capacity in dygraph.
    """
def psroi_pool(x: paddle.Tensor, boxes: paddle.Tensor, boxes_num: paddle.Tensor, pooled_height: int = 1, pooled_width: int = 1, output_channels: int = 1, spatial_scale: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for psroi_pool in dygraph.
    """
def psroi_pool_grad(*args, **kwargs):
    """
    C++ interface function for psroi_pool_grad in dygraph.
    """
def put_along_axis(arr: paddle.Tensor, indices: paddle.Tensor, values: paddle.Tensor, axis: int, reduce: str = "assign", include_self: bool = True) -> paddle.Tensor:
    """
    C++ interface function for put_along_axis in dygraph.
    """
def put_along_axis_(arr: paddle.Tensor, indices: paddle.Tensor, values: paddle.Tensor, axis: int, reduce: str = "assign", include_self: bool = True) -> paddle.Tensor:
    """
    C++ interface function for put_along_axis_ in dygraph.
    """
def put_along_axis_grad(*args, **kwargs):
    """
    C++ interface function for put_along_axis_grad in dygraph.
    """
def pyramid_hash(x: paddle.Tensor, w: paddle.Tensor, white_list: paddle.Tensor, black_list: paddle.Tensor, num_emb: int = 0, space_len: int = 0, pyramid_layer: int = 2, rand_len: int = 0, drop_out_percent: float = 0, is_training: int = 0, use_filter: bool = True, white_list_len: int = 0, black_list_len: int = 0, seed: int = 0, lr: float = 0.0, distribute_update_vars: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for pyramid_hash in dygraph.
    """
def pyramid_hash_grad(*args, **kwargs):
    """
    C++ interface function for pyramid_hash_grad in dygraph.
    """
def qr(x: paddle.Tensor, mode: str = "reduced") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for qr in dygraph.
    """
def qr_grad(*args, **kwargs):
    """
    C++ interface function for qr_grad in dygraph.
    """
def radam_(param: paddle.Tensor, grad: paddle.Tensor, learning_rate: paddle.Tensor, beta1_pow: paddle.Tensor, beta2_pow: paddle.Tensor, rho: paddle.Tensor, moment1: paddle.Tensor, moment2: paddle.Tensor, master_param: paddle.Tensor, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1.0e-8, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for radam_ in dygraph.
    """
def randint(low: int, high: int, shape: list[int], dtype: paddle._typing.DTypeLike = "DataType::INT64", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for randint in dygraph.
    """
def random_routing(prob: paddle.Tensor, topk_value: paddle.Tensor, topk_idx: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for random_routing in dygraph.
    """
def random_routing_(prob: paddle.Tensor, topk_value: paddle.Tensor, topk_idx: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for random_routing_ in dygraph.
    """
def randperm(n: int, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for randperm in dygraph.
    """
def rank_attention(x: paddle.Tensor, rank_offset: paddle.Tensor, rank_param: paddle.Tensor, max_rank: int = 3, max_size: int = 0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for rank_attention in dygraph.
    """
def rank_attention_grad(*args, **kwargs):
    """
    C++ interface function for rank_attention_grad in dygraph.
    """
def read_file(filename: str = "", dtype: paddle._typing.DTypeLike = "DataType::UINT8", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for read_file in dygraph.
    """
def real(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for real in dygraph.
    """
def real_grad(*args, **kwargs):
    """
    C++ interface function for real_grad in dygraph.
    """
def reciprocal(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for reciprocal in dygraph.
    """
def reciprocal_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for reciprocal_ in dygraph.
    """
def reciprocal_grad(*args, **kwargs):
    """
    C++ interface function for reciprocal_grad in dygraph.
    """
def reciprocal_grad_(*args, **kwargs):
    """
    C++ interface function for reciprocal_grad_ in dygraph.
    """
def reduce(x: paddle.Tensor, ring_id: int = 0, root_id: int = 0, reduce_type: int = 0) -> paddle.Tensor:
    """
    C++ interface function for reduce in dygraph.
    """
def reduce_(x: paddle.Tensor, ring_id: int = 0, root_id: int = 0, reduce_type: int = 0) -> paddle.Tensor:
    """
    C++ interface function for reduce_ in dygraph.
    """
def reduce_as(x: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for reduce_as in dygraph.
    """
def reduce_as_grad(*args, **kwargs):
    """
    C++ interface function for reduce_as_grad in dygraph.
    """
def reduce_scatter(x: paddle.Tensor, ring_id: int = 0, nranks: int = 1) -> paddle.Tensor:
    """
    C++ interface function for reduce_scatter in dygraph.
    """
def reindex_graph(x: paddle.Tensor, neighbors: paddle.Tensor, count: paddle.Tensor, hashtable_value: paddle.Tensor, hashtable_index: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for reindex_graph in dygraph.
    """
def relu(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for relu in dygraph.
    """
def relu6(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for relu6 in dygraph.
    """
def relu6_grad(*args, **kwargs):
    """
    C++ interface function for relu6_grad in dygraph.
    """
def relu6_grad_(*args, **kwargs):
    """
    C++ interface function for relu6_grad_ in dygraph.
    """
def relu_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for relu_ in dygraph.
    """
def relu_grad(*args, **kwargs):
    """
    C++ interface function for relu_grad in dygraph.
    """
def relu_grad_(*args, **kwargs):
    """
    C++ interface function for relu_grad_ in dygraph.
    """
def remainder(*args, **kwargs):
    """
    C++ interface function for remainder in dygraph.
    """
def remainder_(*args, **kwargs):
    """
    C++ interface function for remainder_ in dygraph.
    """
def remainder_grad(*args, **kwargs):
    """
    C++ interface function for remainder_grad in dygraph.
    """
def renorm(x: paddle.Tensor, p: float, axis: int, max_norm: float) -> paddle.Tensor:
    """
    C++ interface function for renorm in dygraph.
    """
def renorm_(x: paddle.Tensor, p: float, axis: int, max_norm: float) -> paddle.Tensor:
    """
    C++ interface function for renorm_ in dygraph.
    """
def renorm_grad(*args, **kwargs):
    """
    C++ interface function for renorm_grad in dygraph.
    """
def repeat_interleave(x: paddle.Tensor, repeats: int, axis: int) -> paddle.Tensor:
    """
    C++ interface function for repeat_interleave in dygraph.
    """
def repeat_interleave_grad(*args, **kwargs):
    """
    C++ interface function for repeat_interleave_grad in dygraph.
    """
def repeat_interleave_with_tensor_index(x: paddle.Tensor, repeats: paddle.Tensor, axis: int) -> paddle.Tensor:
    """
    C++ interface function for repeat_interleave_with_tensor_index in dygraph.
    """
def repeat_interleave_with_tensor_index_grad(*args, **kwargs):
    """
    C++ interface function for repeat_interleave_with_tensor_index_grad in dygraph.
    """
def reshape(x: paddle.Tensor, shape: list[int]) -> paddle.Tensor:
    """
    C++ interface function for reshape in dygraph.
    """
def reshape_(x: paddle.Tensor, shape: list[int]) -> paddle.Tensor:
    """
    C++ interface function for reshape_ in dygraph.
    """
def reshape_grad(*args, **kwargs):
    """
    C++ interface function for reshape_grad in dygraph.
    """
def reshape_grad_(*args, **kwargs):
    """
    C++ interface function for reshape_grad_ in dygraph.
    """
def resnet_basic_block(*args, **kwargs):
    """
    C++ interface function for resnet_basic_block in dygraph.
    """
def resnet_basic_block_grad(*args, **kwargs):
    """
    C++ interface function for resnet_basic_block_grad in dygraph.
    """
def resnet_unit(*args, **kwargs):
    """
    C++ interface function for resnet_unit in dygraph.
    """
def resnet_unit_grad(*args, **kwargs):
    """
    C++ interface function for resnet_unit_grad in dygraph.
    """
def reverse(x: paddle.Tensor, axis: list[int]) -> paddle.Tensor:
    """
    C++ interface function for reverse in dygraph.
    """
def reverse_grad(*args, **kwargs):
    """
    C++ interface function for reverse_grad in dygraph.
    """
def rms_norm(x: paddle.Tensor, bias: paddle.Tensor, residual: paddle.Tensor, norm_weight: paddle.Tensor, norm_bias: paddle.Tensor, epsilon: float, begin_norm_axis: int, quant_scale: float, quant_round_type: int, quant_max_bound: float, quant_min_bound: float) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for rms_norm in dygraph.
    """
def rms_norm_grad(*args, **kwargs):
    """
    C++ interface function for rms_norm_grad in dygraph.
    """
def rmsprop_(param: paddle.Tensor, mean_square: paddle.Tensor, grad: paddle.Tensor, moment: paddle.Tensor, learning_rate: paddle.Tensor, mean_grad: paddle.Tensor, master_param: paddle.Tensor, epsilon: float = 1.0e-10, decay: float = 0.9, momentum: float = 0.0, centered: bool = False, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for rmsprop_ in dygraph.
    """
def rnn(x: paddle.Tensor, pre_state: list[paddle.Tensor], weight_list: list[paddle.Tensor], sequence_length: paddle.Tensor, dropout_state_in: paddle.Tensor, dropout_prob: float = 0.0, is_bidirec: bool = False, input_size: int = 10, hidden_size: int = 100, num_layers: int = 1, mode: str = "RNN_TANH", seed: int = 0, is_test: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, list[paddle.Tensor], paddle.Tensor]:
    """
    C++ interface function for rnn in dygraph.
    """
def rnn_grad(*args, **kwargs):
    """
    C++ interface function for rnn_grad in dygraph.
    """
def roi_align(x: paddle.Tensor, boxes: paddle.Tensor, boxes_num: paddle.Tensor, pooled_height: int = 1, pooled_width: int = 1, spatial_scale: float = 1.0, sampling_ratio: int = -1, aligned: bool = False) -> paddle.Tensor:
    """
    C++ interface function for roi_align in dygraph.
    """
def roi_align_grad(*args, **kwargs):
    """
    C++ interface function for roi_align_grad in dygraph.
    """
def roi_pool(x: paddle.Tensor, boxes: paddle.Tensor, boxes_num: paddle.Tensor, pooled_height: int = 1, pooled_width: int = 1, spatial_scale: float = 1.0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for roi_pool in dygraph.
    """
def roi_pool_grad(*args, **kwargs):
    """
    C++ interface function for roi_pool_grad in dygraph.
    """
def roll(x: paddle.Tensor, shifts: list[int] = [], axis: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for roll in dygraph.
    """
def roll_grad(*args, **kwargs):
    """
    C++ interface function for roll_grad in dygraph.
    """
def round(x: paddle.Tensor, decimals: int = 0) -> paddle.Tensor:
    """
    C++ interface function for round in dygraph.
    """
def round_(x: paddle.Tensor, decimals: int = 0) -> paddle.Tensor:
    """
    C++ interface function for round_ in dygraph.
    """
def round_grad(*args, **kwargs):
    """
    C++ interface function for round_grad in dygraph.
    """
def round_grad_(*args, **kwargs):
    """
    C++ interface function for round_grad_ in dygraph.
    """
def rprop_(param: paddle.Tensor, grad: paddle.Tensor, prev: paddle.Tensor, learning_rate: paddle.Tensor, master_param: paddle.Tensor, learning_rate_range: paddle.Tensor, etas: paddle.Tensor, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for rprop_ in dygraph.
    """
def rrelu(x: paddle.Tensor, lower: float = "1.0f/8", upper: float = "1.0f/3", is_test: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for rrelu in dygraph.
    """
def rrelu_grad(*args, **kwargs):
    """
    C++ interface function for rrelu_grad in dygraph.
    """
def rsqrt(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for rsqrt in dygraph.
    """
def rsqrt_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for rsqrt_ in dygraph.
    """
def rsqrt_grad(*args, **kwargs):
    """
    C++ interface function for rsqrt_grad in dygraph.
    """
def rsqrt_grad_(*args, **kwargs):
    """
    C++ interface function for rsqrt_grad_ in dygraph.
    """
def scale(x: paddle.Tensor, scale: float = 1.0, bias: float = 0.0, bias_after_scale: bool = True) -> paddle.Tensor:
    """
    C++ interface function for scale in dygraph.
    """
def scale_(x: paddle.Tensor, scale: float = 1.0, bias: float = 0.0, bias_after_scale: bool = True) -> paddle.Tensor:
    """
    C++ interface function for scale_ in dygraph.
    """
def scatter(x: paddle.Tensor, index: paddle.Tensor, updates: paddle.Tensor, overwrite: bool = True) -> paddle.Tensor:
    """
    C++ interface function for scatter in dygraph.
    """
def scatter_(x: paddle.Tensor, index: paddle.Tensor, updates: paddle.Tensor, overwrite: bool = True) -> paddle.Tensor:
    """
    C++ interface function for scatter_ in dygraph.
    """
def scatter_grad(*args, **kwargs):
    """
    C++ interface function for scatter_grad in dygraph.
    """
def scatter_nd_add(x: paddle.Tensor, index: paddle.Tensor, updates: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for scatter_nd_add in dygraph.
    """
def scatter_nd_add_grad(*args, **kwargs):
    """
    C++ interface function for scatter_nd_add_grad in dygraph.
    """
def searchsorted(sorted_sequence: paddle.Tensor, values: paddle.Tensor, out_int32: bool = False, right: bool = False) -> paddle.Tensor:
    """
    C++ interface function for searchsorted in dygraph.
    """
def segment_pool(x: paddle.Tensor, segment_ids: paddle.Tensor, pooltype: str = "SUM") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for segment_pool in dygraph.
    """
def segment_pool_grad(*args, **kwargs):
    """
    C++ interface function for segment_pool_grad in dygraph.
    """
def selu(x: paddle.Tensor, scale: float = 1.0507009873554804934193349852946, alpha: float = 1.6732632423543772848170429916717) -> paddle.Tensor:
    """
    C++ interface function for selu in dygraph.
    """
def selu_grad(*args, **kwargs):
    """
    C++ interface function for selu_grad in dygraph.
    """
def send_u_recv(x: paddle.Tensor, src_index: paddle.Tensor, dst_index: paddle.Tensor, reduce_op: str = "SUM", out_size: list[int] = [0]) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for send_u_recv in dygraph.
    """
def send_u_recv_grad(*args, **kwargs):
    """
    C++ interface function for send_u_recv_grad in dygraph.
    """
def send_ue_recv(x: paddle.Tensor, y: paddle.Tensor, src_index: paddle.Tensor, dst_index: paddle.Tensor, message_op: str = "ADD", reduce_op: str = "SUM", out_size: list[int] = [0]) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for send_ue_recv in dygraph.
    """
def send_ue_recv_grad(*args, **kwargs):
    """
    C++ interface function for send_ue_recv_grad in dygraph.
    """
def send_uv(x: paddle.Tensor, y: paddle.Tensor, src_index: paddle.Tensor, dst_index: paddle.Tensor, message_op: str = "ADD") -> paddle.Tensor:
    """
    C++ interface function for send_uv in dygraph.
    """
def send_uv_grad(*args, **kwargs):
    """
    C++ interface function for send_uv_grad in dygraph.
    """
def sequence_conv(x: paddle.Tensor, padding_data: paddle.Tensor, filter: paddle.Tensor, context_length: int, padding_trainable: bool = False, context_start: int = 0, context_stride: int = 1) -> paddle.Tensor:
    """
    C++ interface function for sequence_conv in dygraph.
    """
def sequence_conv_grad(*args, **kwargs):
    """
    C++ interface function for sequence_conv_grad in dygraph.
    """
def sequence_mask(x: paddle.Tensor, max_len: int, out_dtype: paddle._typing.DTypeLike) -> paddle.Tensor:
    """
    C++ interface function for sequence_mask in dygraph.
    """
def sequence_pool(x: paddle.Tensor, is_test: bool = False, pooltype: str = "AVERAGE", pad_value: float = 0.0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sequence_pool in dygraph.
    """
def sequence_pool_grad(*args, **kwargs):
    """
    C++ interface function for sequence_pool_grad in dygraph.
    """
def set(x: paddle.Tensor, source: paddle.Tensor, dims: list[int] = [], stride: list[int] = [], offset: int = 0) -> paddle.Tensor:
    """
    C++ interface function for set in dygraph.
    """
def set_(x: paddle.Tensor, source: paddle.Tensor, dims: list[int] = [], stride: list[int] = [], offset: int = 0) -> paddle.Tensor:
    """
    C++ interface function for set_ in dygraph.
    """
def set_value(*args, **kwargs):
    """
    C++ interface function for set_value in dygraph.
    """
def set_value_(*args, **kwargs):
    """
    C++ interface function for set_value_ in dygraph.
    """
def set_value_grad(*args, **kwargs):
    """
    C++ interface function for set_value_grad in dygraph.
    """
def set_value_with_tensor(x: paddle.Tensor, values: paddle.Tensor, starts: list[int], ends: list[int], steps: list[int], axes: list[int], decrease_axes: list[int], none_axes: list[int]) -> paddle.Tensor:
    """
    C++ interface function for set_value_with_tensor in dygraph.
    """
def set_value_with_tensor_(x: paddle.Tensor, values: paddle.Tensor, starts: list[int], ends: list[int], steps: list[int], axes: list[int], decrease_axes: list[int], none_axes: list[int]) -> paddle.Tensor:
    """
    C++ interface function for set_value_with_tensor_ in dygraph.
    """
def set_value_with_tensor_grad(*args, **kwargs):
    """
    C++ interface function for set_value_with_tensor_grad in dygraph.
    """
def sgd_(param: paddle.Tensor, learning_rate: paddle.Tensor, grad: paddle.Tensor, master_param: paddle.Tensor, multi_precision: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sgd_ in dygraph.
    """
def shape(input: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for shape in dygraph.
    """
def shape64(input: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for shape64 in dygraph.
    """
def shard_index(input: paddle.Tensor, index_num: int, nshards: int, shard_id: int, ignore_value: int = -1) -> paddle.Tensor:
    """
    C++ interface function for shard_index in dygraph.
    """
def share_data(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for share_data in dygraph.
    """
def shuffle_batch(x: paddle.Tensor, seed: paddle.Tensor, startup_seed: int = 0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for shuffle_batch in dygraph.
    """
def shuffle_batch_grad(*args, **kwargs):
    """
    C++ interface function for shuffle_batch_grad in dygraph.
    """
def shuffle_channel(x: paddle.Tensor, group: int = 1) -> paddle.Tensor:
    """
    C++ interface function for shuffle_channel in dygraph.
    """
def shuffle_channel_grad(*args, **kwargs):
    """
    C++ interface function for shuffle_channel_grad in dygraph.
    """
def sigmoid(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sigmoid in dygraph.
    """
def sigmoid_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sigmoid_ in dygraph.
    """
def sigmoid_cross_entropy_with_logits(x: paddle.Tensor, label: paddle.Tensor, pos_weight: paddle.Tensor, normalize: bool = False, ignore_index: int = -100) -> paddle.Tensor:
    """
    C++ interface function for sigmoid_cross_entropy_with_logits in dygraph.
    """
def sigmoid_cross_entropy_with_logits_(x: paddle.Tensor, label: paddle.Tensor, pos_weight: paddle.Tensor, normalize: bool = False, ignore_index: int = -100) -> paddle.Tensor:
    """
    C++ interface function for sigmoid_cross_entropy_with_logits_ in dygraph.
    """
def sigmoid_cross_entropy_with_logits_grad(*args, **kwargs):
    """
    C++ interface function for sigmoid_cross_entropy_with_logits_grad in dygraph.
    """
def sigmoid_cross_entropy_with_logits_grad_(*args, **kwargs):
    """
    C++ interface function for sigmoid_cross_entropy_with_logits_grad_ in dygraph.
    """
def sigmoid_grad(*args, **kwargs):
    """
    C++ interface function for sigmoid_grad in dygraph.
    """
def sigmoid_grad_(*args, **kwargs):
    """
    C++ interface function for sigmoid_grad_ in dygraph.
    """
def sign(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sign in dygraph.
    """
def sign_grad(*args, **kwargs):
    """
    C++ interface function for sign_grad in dygraph.
    """
def silu(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for silu in dygraph.
    """
def silu_grad(*args, **kwargs):
    """
    C++ interface function for silu_grad in dygraph.
    """
def silu_grad_(*args, **kwargs):
    """
    C++ interface function for silu_grad_ in dygraph.
    """
def sin(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sin in dygraph.
    """
def sin_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sin_ in dygraph.
    """
def sin_grad(*args, **kwargs):
    """
    C++ interface function for sin_grad in dygraph.
    """
def sin_grad_(*args, **kwargs):
    """
    C++ interface function for sin_grad_ in dygraph.
    """
def sinh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sinh in dygraph.
    """
def sinh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sinh_ in dygraph.
    """
def sinh_grad(*args, **kwargs):
    """
    C++ interface function for sinh_grad in dygraph.
    """
def sinh_grad_(*args, **kwargs):
    """
    C++ interface function for sinh_grad_ in dygraph.
    """
def slice(input: paddle.Tensor, axes: list[int], starts: list[int], ends: list[int], infer_flags: list[int], decrease_axis: list[int]) -> paddle.Tensor:
    """
    C++ interface function for slice in dygraph.
    """
def slice_grad(*args, **kwargs):
    """
    C++ interface function for slice_grad in dygraph.
    """
def slogdet(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for slogdet in dygraph.
    """
def slogdet_grad(*args, **kwargs):
    """
    C++ interface function for slogdet_grad in dygraph.
    """
def softmax(*args, **kwargs):
    """
    C++ interface function for softmax in dygraph.
    """
def softmax_(*args, **kwargs):
    """
    C++ interface function for softmax_ in dygraph.
    """
def softmax_grad(*args, **kwargs):
    """
    C++ interface function for softmax_grad in dygraph.
    """
def softplus(x: paddle.Tensor, beta: float = 1.0, threshold: float = 20.0) -> paddle.Tensor:
    """
    C++ interface function for softplus in dygraph.
    """
def softplus_grad(*args, **kwargs):
    """
    C++ interface function for softplus_grad in dygraph.
    """
def softplus_grad_(*args, **kwargs):
    """
    C++ interface function for softplus_grad_ in dygraph.
    """
def softshrink(x: paddle.Tensor, threshold: float = 0.5) -> paddle.Tensor:
    """
    C++ interface function for softshrink in dygraph.
    """
def softshrink_grad(*args, **kwargs):
    """
    C++ interface function for softshrink_grad in dygraph.
    """
def softshrink_grad_(*args, **kwargs):
    """
    C++ interface function for softshrink_grad_ in dygraph.
    """
def softsign(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for softsign in dygraph.
    """
def softsign_grad(*args, **kwargs):
    """
    C++ interface function for softsign_grad in dygraph.
    """
def softsign_grad_(*args, **kwargs):
    """
    C++ interface function for softsign_grad_ in dygraph.
    """
def solve(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for solve in dygraph.
    """
def solve_grad(*args, **kwargs):
    """
    C++ interface function for solve_grad in dygraph.
    """
def sparse_abs(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for abs in dygraph.
    """
def sparse_abs_grad(*args, **kwargs):
    """
    C++ interface function for abs_grad in dygraph.
    """
def sparse_acos(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for acos in dygraph.
    """
def sparse_acos_grad(*args, **kwargs):
    """
    C++ interface function for acos_grad in dygraph.
    """
def sparse_acosh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for acosh in dygraph.
    """
def sparse_acosh_grad(*args, **kwargs):
    """
    C++ interface function for acosh_grad in dygraph.
    """
def sparse_add(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for add in dygraph.
    """
def sparse_add_grad(*args, **kwargs):
    """
    C++ interface function for add_grad in dygraph.
    """
def sparse_addmm(input: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor, beta: float = 1.0, alpha: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for addmm in dygraph.
    """
def sparse_addmm_grad(*args, **kwargs):
    """
    C++ interface function for addmm_grad in dygraph.
    """
def sparse_asin(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for asin in dygraph.
    """
def sparse_asin_grad(*args, **kwargs):
    """
    C++ interface function for asin_grad in dygraph.
    """
def sparse_asinh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for asinh in dygraph.
    """
def sparse_asinh_grad(*args, **kwargs):
    """
    C++ interface function for asinh_grad in dygraph.
    """
def sparse_atan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atan in dygraph.
    """
def sparse_atan_grad(*args, **kwargs):
    """
    C++ interface function for atan_grad in dygraph.
    """
def sparse_atanh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for atanh in dygraph.
    """
def sparse_atanh_grad(*args, **kwargs):
    """
    C++ interface function for atanh_grad in dygraph.
    """
def sparse_attention(q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, offset: paddle.Tensor, columns: paddle.Tensor, key_padding_mask: paddle.Tensor, attn_mask: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sparse_attention in dygraph.
    """
def sparse_attention_grad(*args, **kwargs):
    """
    C++ interface function for sparse_attention_grad in dygraph.
    """
def sparse_batch_norm_(x: paddle.Tensor, mean: paddle.Tensor, variance: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, is_test: bool, momentum: float, epsilon: float, data_format: str, use_global_stats: bool, trainable_statistics: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for batch_norm_ in dygraph.
    """
def sparse_batch_norm_grad(*args, **kwargs):
    """
    C++ interface function for batch_norm_grad in dygraph.
    """
def sparse_cast(x: paddle.Tensor, index_dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", value_dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED") -> paddle.Tensor:
    """
    C++ interface function for cast in dygraph.
    """
def sparse_cast_grad(*args, **kwargs):
    """
    C++ interface function for cast_grad in dygraph.
    """
def sparse_coalesce(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for coalesce in dygraph.
    """
def sparse_conv3d(x: paddle.Tensor, kernel: paddle.Tensor, paddings: list[int], dilations: list[int], strides: list[int], groups: int, subm: bool, key: str = "") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for conv3d in dygraph.
    """
def sparse_conv3d_grad(*args, **kwargs):
    """
    C++ interface function for conv3d_grad in dygraph.
    """
def sparse_conv3d_implicit_gemm(x: paddle.Tensor, kernel: paddle.Tensor, paddings: list[int], dilations: list[int], strides: list[int], groups: int, subm: bool, key: str = "") -> paddle.Tensor:
    """
    C++ interface function for conv3d_implicit_gemm in dygraph.
    """
def sparse_divide(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for divide in dygraph.
    """
def sparse_divide_grad(*args, **kwargs):
    """
    C++ interface function for divide_grad in dygraph.
    """
def sparse_divide_scalar(x: paddle.Tensor, scalar: float) -> paddle.Tensor:
    """
    C++ interface function for divide_scalar in dygraph.
    """
def sparse_divide_scalar_grad(*args, **kwargs):
    """
    C++ interface function for divide_scalar_grad in dygraph.
    """
def sparse_expm1(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for expm1 in dygraph.
    """
def sparse_expm1_grad(*args, **kwargs):
    """
    C++ interface function for expm1_grad in dygraph.
    """
def sparse_full_like(x: paddle.Tensor, value: float, dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED") -> paddle.Tensor:
    """
    C++ interface function for full_like in dygraph.
    """
def sparse_fused_attention(query: paddle.Tensor, key: paddle.Tensor, value: paddle.Tensor, sparse_mask: paddle.Tensor, key_padding_mask: paddle.Tensor, attn_mask: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for fused_attention in dygraph.
    """
def sparse_fused_attention_grad(*args, **kwargs):
    """
    C++ interface function for fused_attention_grad in dygraph.
    """
def sparse_indices(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for indices in dygraph.
    """
def sparse_isnan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for isnan in dygraph.
    """
def sparse_leaky_relu(x: paddle.Tensor, alpha: float) -> paddle.Tensor:
    """
    C++ interface function for leaky_relu in dygraph.
    """
def sparse_leaky_relu_grad(*args, **kwargs):
    """
    C++ interface function for leaky_relu_grad in dygraph.
    """
def sparse_log1p(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for log1p in dygraph.
    """
def sparse_log1p_grad(*args, **kwargs):
    """
    C++ interface function for log1p_grad in dygraph.
    """
def sparse_mask_as(x: paddle.Tensor, mask: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for mask_as in dygraph.
    """
def sparse_mask_as_grad(*args, **kwargs):
    """
    C++ interface function for mask_as_grad in dygraph.
    """
def sparse_masked_matmul(x: paddle.Tensor, y: paddle.Tensor, mask: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for masked_matmul in dygraph.
    """
def sparse_masked_matmul_grad(*args, **kwargs):
    """
    C++ interface function for masked_matmul_grad in dygraph.
    """
def sparse_matmul(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for matmul in dygraph.
    """
def sparse_matmul_grad(*args, **kwargs):
    """
    C++ interface function for matmul_grad in dygraph.
    """
def sparse_maxpool(x: paddle.Tensor, kernel_sizes: list[int], paddings: list[int], dilations: list[int], strides: list[int]) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for maxpool in dygraph.
    """
def sparse_maxpool_grad(*args, **kwargs):
    """
    C++ interface function for maxpool_grad in dygraph.
    """
def sparse_multiply(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for multiply in dygraph.
    """
def sparse_mv(x: paddle.Tensor, vec: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for mv in dygraph.
    """
def sparse_mv_grad(*args, **kwargs):
    """
    C++ interface function for mv_grad in dygraph.
    """
def sparse_pow(x: paddle.Tensor, factor: float) -> paddle.Tensor:
    """
    C++ interface function for pow in dygraph.
    """
def sparse_pow_grad(*args, **kwargs):
    """
    C++ interface function for pow_grad in dygraph.
    """
def sparse_relu(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for relu in dygraph.
    """
def sparse_relu6(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for relu6 in dygraph.
    """
def sparse_relu6_grad(*args, **kwargs):
    """
    C++ interface function for relu6_grad in dygraph.
    """
def sparse_relu_grad(*args, **kwargs):
    """
    C++ interface function for relu_grad in dygraph.
    """
def sparse_reshape(x: paddle.Tensor, shape: list[int]) -> paddle.Tensor:
    """
    C++ interface function for reshape in dygraph.
    """
def sparse_reshape_grad(*args, **kwargs):
    """
    C++ interface function for reshape_grad in dygraph.
    """
def sparse_scale(x: paddle.Tensor, scale: float, bias: float, bias_after_scale: bool) -> paddle.Tensor:
    """
    C++ interface function for scale in dygraph.
    """
def sparse_sin(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sin in dygraph.
    """
def sparse_sin_grad(*args, **kwargs):
    """
    C++ interface function for sin_grad in dygraph.
    """
def sparse_sinh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sinh in dygraph.
    """
def sparse_sinh_grad(*args, **kwargs):
    """
    C++ interface function for sinh_grad in dygraph.
    """
def sparse_slice(x: paddle.Tensor, axes: list[int], starts: list[int], ends: list[int]) -> paddle.Tensor:
    """
    C++ interface function for slice in dygraph.
    """
def sparse_slice_grad(*args, **kwargs):
    """
    C++ interface function for slice_grad in dygraph.
    """
def sparse_softmax(x: paddle.Tensor, axis: int = -1) -> paddle.Tensor:
    """
    C++ interface function for softmax in dygraph.
    """
def sparse_softmax_grad(*args, **kwargs):
    """
    C++ interface function for softmax_grad in dygraph.
    """
def sparse_sparse_coo_tensor(values: paddle.Tensor, indices: paddle.Tensor, shape: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for sparse_coo_tensor in dygraph.
    """
def sparse_sparse_coo_tensor_grad(*args, **kwargs):
    """
    C++ interface function for sparse_coo_tensor_grad in dygraph.
    """
def sparse_sqrt(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sqrt in dygraph.
    """
def sparse_sqrt_grad(*args, **kwargs):
    """
    C++ interface function for sqrt_grad in dygraph.
    """
def sparse_square(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for square in dygraph.
    """
def sparse_square_grad(*args, **kwargs):
    """
    C++ interface function for square_grad in dygraph.
    """
def sparse_subtract(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for subtract in dygraph.
    """
def sparse_subtract_grad(*args, **kwargs):
    """
    C++ interface function for subtract_grad in dygraph.
    """
def sparse_sum(x: paddle.Tensor, axis: list[int] = [], dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for sum in dygraph.
    """
def sparse_sum_grad(*args, **kwargs):
    """
    C++ interface function for sum_grad in dygraph.
    """
def sparse_sync_batch_norm_(x: paddle.Tensor, mean: paddle.Tensor, variance: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, is_test: bool, momentum: float, epsilon: float, data_format: str, use_global_stats: bool, trainable_statistics: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sync_batch_norm_ in dygraph.
    """
def sparse_sync_batch_norm_grad(*args, **kwargs):
    """
    C++ interface function for sync_batch_norm_grad in dygraph.
    """
def sparse_tan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tan in dygraph.
    """
def sparse_tan_grad(*args, **kwargs):
    """
    C++ interface function for tan_grad in dygraph.
    """
def sparse_tanh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tanh in dygraph.
    """
def sparse_tanh_grad(*args, **kwargs):
    """
    C++ interface function for tanh_grad in dygraph.
    """
def sparse_to_dense(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for to_dense in dygraph.
    """
def sparse_to_dense_grad(*args, **kwargs):
    """
    C++ interface function for to_dense_grad in dygraph.
    """
def sparse_to_sparse_coo(x: paddle.Tensor, sparse_dim: int) -> paddle.Tensor:
    """
    C++ interface function for to_sparse_coo in dygraph.
    """
def sparse_to_sparse_coo_grad(*args, **kwargs):
    """
    C++ interface function for to_sparse_coo_grad in dygraph.
    """
def sparse_to_sparse_csr(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for to_sparse_csr in dygraph.
    """
def sparse_transpose(x: paddle.Tensor, perm: list[int]) -> paddle.Tensor:
    """
    C++ interface function for transpose in dygraph.
    """
def sparse_transpose_grad(*args, **kwargs):
    """
    C++ interface function for transpose_grad in dygraph.
    """
def sparse_values(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for values in dygraph.
    """
def sparse_values_grad(*args, **kwargs):
    """
    C++ interface function for values_grad in dygraph.
    """
def spectral_norm(weight: paddle.Tensor, u: paddle.Tensor, v: paddle.Tensor, dim: int = 0, power_iters: int = 1, eps: float = 1e-12) -> paddle.Tensor:
    """
    C++ interface function for spectral_norm in dygraph.
    """
def spectral_norm_grad(*args, **kwargs):
    """
    C++ interface function for spectral_norm_grad in dygraph.
    """
def split(x: paddle.Tensor, sections: list[int], axis: int) -> list[paddle.Tensor]:
    """
    C++ interface function for split in dygraph.
    """
def split_grad(*args, **kwargs):
    """
    C++ interface function for split_grad in dygraph.
    """
def split_with_num(x: paddle.Tensor, num: int, axis: int) -> list[paddle.Tensor]:
    """
    C++ interface function for split_with_num in dygraph.
    """
def split_with_num_grad(*args, **kwargs):
    """
    C++ interface function for split_with_num_grad in dygraph.
    """
def sqrt(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sqrt in dygraph.
    """
def sqrt_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sqrt_ in dygraph.
    """
def sqrt_grad(*args, **kwargs):
    """
    C++ interface function for sqrt_grad in dygraph.
    """
def sqrt_grad_(*args, **kwargs):
    """
    C++ interface function for sqrt_grad_ in dygraph.
    """
def square(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for square in dygraph.
    """
def square_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for square_ in dygraph.
    """
def square_grad(*args, **kwargs):
    """
    C++ interface function for square_grad in dygraph.
    """
def square_grad_(*args, **kwargs):
    """
    C++ interface function for square_grad_ in dygraph.
    """
def squared_l2_norm(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for squared_l2_norm in dygraph.
    """
def squared_l2_norm_grad(*args, **kwargs):
    """
    C++ interface function for squared_l2_norm_grad in dygraph.
    """
def squeeze(x: paddle.Tensor, axis: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for squeeze in dygraph.
    """
def squeeze_(x: paddle.Tensor, axis: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for squeeze_ in dygraph.
    """
def squeeze_grad(*args, **kwargs):
    """
    C++ interface function for squeeze_grad in dygraph.
    """
def squeeze_grad_(*args, **kwargs):
    """
    C++ interface function for squeeze_grad_ in dygraph.
    """
def stack(x: list[paddle.Tensor], axis: int = 0) -> paddle.Tensor:
    """
    C++ interface function for stack in dygraph.
    """
def stack_grad(*args, **kwargs):
    """
    C++ interface function for stack_grad in dygraph.
    """
def standard_gamma(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for standard_gamma in dygraph.
    """
def stanh(x: paddle.Tensor, scale_a: float = 0.67, scale_b: float = 1.7159) -> paddle.Tensor:
    """
    C++ interface function for stanh in dygraph.
    """
def stanh_grad(*args, **kwargs):
    """
    C++ interface function for stanh_grad in dygraph.
    """
def stft(x: paddle.Tensor, window: paddle.Tensor, n_fft: int, hop_length: int, normalized: bool, onesided: bool) -> paddle.Tensor:
    """
    C++ interface function for stft in dygraph.
    """
def stft_grad(*args, **kwargs):
    """
    C++ interface function for stft_grad in dygraph.
    """
def strided_slice(x: paddle.Tensor, axes: list[int], starts: list[int], ends: list[int], strides: list[int]) -> paddle.Tensor:
    """
    C++ interface function for strided_slice in dygraph.
    """
def strided_slice_grad(*args, **kwargs):
    """
    C++ interface function for strided_slice_grad in dygraph.
    """
def strings_empty(shape: list[int], place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for empty in dygraph.
    """
def strings_empty_like(x: paddle.Tensor, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for empty_like in dygraph.
    """
def strings_lower(x: paddle.Tensor, use_utf8_encoding: bool) -> paddle.Tensor:
    """
    C++ interface function for lower in dygraph.
    """
def strings_upper(x: paddle.Tensor, use_utf8_encoding: bool) -> paddle.Tensor:
    """
    C++ interface function for upper in dygraph.
    """
def subtract(*args, **kwargs):
    """
    C++ interface function for subtract in dygraph.
    """
def subtract_(*args, **kwargs):
    """
    C++ interface function for subtract_ in dygraph.
    """
def subtract_grad(*args, **kwargs):
    """
    C++ interface function for subtract_grad in dygraph.
    """
def subtract_grad_(*args, **kwargs):
    """
    C++ interface function for subtract_grad_ in dygraph.
    """
def sum(x: paddle.Tensor, axis: list[int] = [], dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for sum in dygraph.
    """
def sum_grad(*args, **kwargs):
    """
    C++ interface function for sum_grad in dygraph.
    """
def svd(x: paddle.Tensor, full_matrices: bool = False) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for svd in dygraph.
    """
def svd_grad(*args, **kwargs):
    """
    C++ interface function for svd_grad in dygraph.
    """
def svdvals(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for svdvals in dygraph.
    """
def svdvals_grad(*args, **kwargs):
    """
    C++ interface function for svdvals_grad in dygraph.
    """
def swiglu(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for swiglu in dygraph.
    """
def swiglu_grad(*args, **kwargs):
    """
    C++ interface function for swiglu_grad in dygraph.
    """
def swish(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for swish in dygraph.
    """
def swish_grad(*args, **kwargs):
    """
    C++ interface function for swish_grad in dygraph.
    """
def swish_grad_(*args, **kwargs):
    """
    C++ interface function for swish_grad_ in dygraph.
    """
def sync_batch_norm_(x: paddle.Tensor, mean: paddle.Tensor, variance: paddle.Tensor, scale: paddle.Tensor, bias: paddle.Tensor, is_test: bool, momentum: float, epsilon: float, data_format: str, use_global_stats: bool, trainable_statistics: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for sync_batch_norm_ in dygraph.
    """
def sync_batch_norm_grad(*args, **kwargs):
    """
    C++ interface function for sync_batch_norm_grad in dygraph.
    """
def sync_calc_stream(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sync_calc_stream in dygraph.
    """
def sync_calc_stream_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for sync_calc_stream_ in dygraph.
    """
def sync_comm_stream(*args, **kwargs):
    """
    C++ interface function for sync_comm_stream in dygraph.
    """
def take_along_axis(arr: paddle.Tensor, indices: paddle.Tensor, axis: int) -> paddle.Tensor:
    """
    C++ interface function for take_along_axis in dygraph.
    """
def take_along_axis_grad(*args, **kwargs):
    """
    C++ interface function for take_along_axis_grad in dygraph.
    """
def tan(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tan in dygraph.
    """
def tan_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tan_ in dygraph.
    """
def tan_grad(*args, **kwargs):
    """
    C++ interface function for tan_grad in dygraph.
    """
def tan_grad_(*args, **kwargs):
    """
    C++ interface function for tan_grad_ in dygraph.
    """
def tanh(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tanh in dygraph.
    """
def tanh_(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tanh_ in dygraph.
    """
def tanh_grad(*args, **kwargs):
    """
    C++ interface function for tanh_grad in dygraph.
    """
def tanh_grad_(*args, **kwargs):
    """
    C++ interface function for tanh_grad_ in dygraph.
    """
def tanh_shrink(x: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for tanh_shrink in dygraph.
    """
def tanh_shrink_grad(*args, **kwargs):
    """
    C++ interface function for tanh_shrink_grad in dygraph.
    """
def tanh_shrink_grad_(*args, **kwargs):
    """
    C++ interface function for tanh_shrink_grad_ in dygraph.
    """
def tdm_child(x: paddle.Tensor, tree_info: paddle.Tensor, child_nums: int, dtype: paddle._typing.DTypeLike = "DataType::INT32") -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for tdm_child in dygraph.
    """
def tdm_sampler(x: paddle.Tensor, travel: paddle.Tensor, layer: paddle.Tensor, output_positive: bool = True, neg_samples_num_list: list[int] = [], layer_offset: list[int] = [], seed: int = 0, dtype: int = 2) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for tdm_sampler in dygraph.
    """
def temporal_shift(x: paddle.Tensor, seg_num: int, shift_ratio: float = 0.25, data_format: str = "NCHW") -> paddle.Tensor:
    """
    C++ interface function for temporal_shift in dygraph.
    """
def temporal_shift_grad(*args, **kwargs):
    """
    C++ interface function for temporal_shift_grad in dygraph.
    """
def tensor_unfold(*args, **kwargs):
    """
    C++ interface function for tensor_unfold in dygraph.
    """
def tensor_unfold_grad(*args, **kwargs):
    """
    C++ interface function for tensor_unfold_grad in dygraph.
    """
def thresholded_relu(x: paddle.Tensor, threshold: float = 1.0, value: float = 0.0) -> paddle.Tensor:
    """
    C++ interface function for thresholded_relu in dygraph.
    """
def thresholded_relu_(x: paddle.Tensor, threshold: float = 1.0, value: float = 0.0) -> paddle.Tensor:
    """
    C++ interface function for thresholded_relu_ in dygraph.
    """
def thresholded_relu_grad(*args, **kwargs):
    """
    C++ interface function for thresholded_relu_grad in dygraph.
    """
def thresholded_relu_grad_(*args, **kwargs):
    """
    C++ interface function for thresholded_relu_grad_ in dygraph.
    """
def tile(*args, **kwargs):
    """
    C++ interface function for tile in dygraph.
    """
def tile_grad(*args, **kwargs):
    """
    C++ interface function for tile_grad in dygraph.
    """
def top_p_sampling(x: paddle.Tensor, ps: paddle.Tensor, threshold: paddle.Tensor, topp_seed: paddle.Tensor, seed: int = -1, k: int = 0, mode: str = "truncate") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for top_p_sampling in dygraph.
    """
def topk(x: paddle.Tensor, k: int = 1, axis: int = -1, largest: bool = True, sorted: bool = True) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for topk in dygraph.
    """
def topk_grad(*args, **kwargs):
    """
    C++ interface function for topk_grad in dygraph.
    """
def trace(x: paddle.Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> paddle.Tensor:
    """
    C++ interface function for trace in dygraph.
    """
def trace_grad(*args, **kwargs):
    """
    C++ interface function for trace_grad in dygraph.
    """
def trans_layout(x: paddle.Tensor, perm: list[int]) -> paddle.Tensor:
    """
    C++ interface function for trans_layout in dygraph.
    """
def trans_layout_grad(*args, **kwargs):
    """
    C++ interface function for trans_layout_grad in dygraph.
    """
def transpose(x: paddle.Tensor, perm: list[int]) -> paddle.Tensor:
    """
    C++ interface function for transpose in dygraph.
    """
def transpose_(x: paddle.Tensor, perm: list[int]) -> paddle.Tensor:
    """
    C++ interface function for transpose_ in dygraph.
    """
def transpose_grad(*args, **kwargs):
    """
    C++ interface function for transpose_grad in dygraph.
    """
def triangular_solve(x: paddle.Tensor, y: paddle.Tensor, upper: bool = True, transpose: bool = False, unitriangular: bool = False) -> paddle.Tensor:
    """
    C++ interface function for triangular_solve in dygraph.
    """
def triangular_solve_grad(*args, **kwargs):
    """
    C++ interface function for triangular_solve_grad in dygraph.
    """
def tril(x: paddle.Tensor, diagonal: int) -> paddle.Tensor:
    """
    C++ interface function for tril in dygraph.
    """
def tril_(x: paddle.Tensor, diagonal: int) -> paddle.Tensor:
    """
    C++ interface function for tril_ in dygraph.
    """
def tril_grad(*args, **kwargs):
    """
    C++ interface function for tril_grad in dygraph.
    """
def tril_indices(rows: int, cols: int, offset: int, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for tril_indices in dygraph.
    """
def trilinear_interp(x: paddle.Tensor, out_size: paddle.Tensor, size_tensor: list[paddle.Tensor], scale_tensor: paddle.Tensor, data_format: str = "NCHW", out_d: int = 0, out_h: int = 0, out_w: int = 0, scale: list[float] = [], interp_method: str = "bilinear", align_corners: bool = True, align_mode: int = 1) -> paddle.Tensor:
    """
    C++ interface function for trilinear_interp in dygraph.
    """
def trilinear_interp_grad(*args, **kwargs):
    """
    C++ interface function for trilinear_interp_grad in dygraph.
    """
def triu(x: paddle.Tensor, diagonal: int) -> paddle.Tensor:
    """
    C++ interface function for triu in dygraph.
    """
def triu_(x: paddle.Tensor, diagonal: int) -> paddle.Tensor:
    """
    C++ interface function for triu_ in dygraph.
    """
def triu_grad(*args, **kwargs):
    """
    C++ interface function for triu_grad in dygraph.
    """
def triu_indices(row: int, col: int, offset: int, dtype: paddle._typing.DTypeLike, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for triu_indices in dygraph.
    """
def trunc(input: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for trunc in dygraph.
    """
def trunc_(input: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for trunc_ in dygraph.
    """
def trunc_grad(*args, **kwargs):
    """
    C++ interface function for trunc_grad in dygraph.
    """
def truncated_gaussian_random(shape: list[int], mean: float, std: float, seed: int, a: float, b: float, dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for truncated_gaussian_random in dygraph.
    """
def unbind(input: paddle.Tensor, axis: int = 0) -> list[paddle.Tensor]:
    """
    C++ interface function for unbind in dygraph.
    """
def unbind_grad(*args, **kwargs):
    """
    C++ interface function for unbind_grad in dygraph.
    """
def unfold(x: paddle.Tensor, kernel_sizes: list[int], strides: list[int], paddings: list[int], dilations: list[int]) -> paddle.Tensor:
    """
    C++ interface function for unfold in dygraph.
    """
def unfold_grad(*args, **kwargs):
    """
    C++ interface function for unfold_grad in dygraph.
    """
def uniform(shape: list[int], dtype: paddle._typing.DTypeLike, min: float, max: float, seed: int, place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for uniform in dygraph.
    """
def uniform_inplace(x: paddle.Tensor, min: float = -1.0, max: float = 1.0, seed: int = 0, diag_num: int = 0, diag_step: int = 0, diag_val: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for uniform_inplace in dygraph.
    """
def uniform_inplace_(x: paddle.Tensor, min: float = -1.0, max: float = 1.0, seed: int = 0, diag_num: int = 0, diag_step: int = 0, diag_val: float = 1.0) -> paddle.Tensor:
    """
    C++ interface function for uniform_inplace_ in dygraph.
    """
def uniform_inplace_grad(*args, **kwargs):
    """
    C++ interface function for uniform_inplace_grad in dygraph.
    """
def uniform_inplace_grad_(*args, **kwargs):
    """
    C++ interface function for uniform_inplace_grad_ in dygraph.
    """
def uniform_random_batch_size_like(input: paddle.Tensor, shape: list[int], input_dim_idx: int = 0, output_dim_idx: int = 0, min: float = -1.0, max: float = 1.0, seed: int = 0, diag_num: int = 0, diag_step: int = 0, diag_val: float = 1.0, dtype: paddle._typing.DTypeLike = "DataType::FLOAT32") -> paddle.Tensor:
    """
    C++ interface function for uniform_random_batch_size_like in dygraph.
    """
def unique(*args, **kwargs):
    """
    C++ interface function for unique in dygraph.
    """
def unique_consecutive(x: paddle.Tensor, return_inverse: bool = False, return_counts: bool = False, axis: list[int] = [], dtype: paddle._typing.DTypeLike = "DataType::FLOAT32") -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for unique_consecutive in dygraph.
    """
def unpool(x: paddle.Tensor, indices: paddle.Tensor, ksize: list[int], strides: list[int], padding: list[int], output_size: list[int], data_format: str) -> paddle.Tensor:
    """
    C++ interface function for unpool in dygraph.
    """
def unpool3d(x: paddle.Tensor, indices: paddle.Tensor, ksize: list[int], strides: list[int] = [1,1,1], paddings: list[int] = [0,0,0], output_size: list[int] = [0,0,0], data_format: str = "NCDHW") -> paddle.Tensor:
    """
    C++ interface function for unpool3d in dygraph.
    """
def unpool3d_grad(*args, **kwargs):
    """
    C++ interface function for unpool3d_grad in dygraph.
    """
def unpool_grad(*args, **kwargs):
    """
    C++ interface function for unpool_grad in dygraph.
    """
def unsqueeze(x: paddle.Tensor, axis: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for unsqueeze in dygraph.
    """
def unsqueeze_(x: paddle.Tensor, axis: list[int] = []) -> paddle.Tensor:
    """
    C++ interface function for unsqueeze_ in dygraph.
    """
def unsqueeze_grad(*args, **kwargs):
    """
    C++ interface function for unsqueeze_grad in dygraph.
    """
def unsqueeze_grad_(*args, **kwargs):
    """
    C++ interface function for unsqueeze_grad_ in dygraph.
    """
def unstack(x: paddle.Tensor, axis: int = 0, num: int = 0) -> list[paddle.Tensor]:
    """
    C++ interface function for unstack in dygraph.
    """
def unstack_grad(*args, **kwargs):
    """
    C++ interface function for unstack_grad in dygraph.
    """
def update_loss_scaling_(x: list[paddle.Tensor], found_infinite: paddle.Tensor, prev_loss_scaling: paddle.Tensor, in_good_steps: paddle.Tensor, in_bad_steps: paddle.Tensor, incr_every_n_steps: int, decr_every_n_nan_or_inf: int, incr_ratio: float, decr_ratio: float, stop_update: float = False) -> tuple[list[paddle.Tensor], paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for update_loss_scaling_ in dygraph.
    """
def variable_length_memory_efficient_attention(*args, **kwargs):
    """
    C++ interface function for variable_length_memory_efficient_attention in dygraph.
    """
def variance(x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False) -> paddle.Tensor:
    """
    C++ interface function for variance in dygraph.
    """
def view_dtype(input: paddle.Tensor, dtype: paddle._typing.DTypeLike) -> paddle.Tensor:
    """
    C++ interface function for view_dtype in dygraph.
    """
def view_dtype_grad(*args, **kwargs):
    """
    C++ interface function for view_dtype_grad in dygraph.
    """
def view_shape(*args, **kwargs):
    """
    C++ interface function for view_shape in dygraph.
    """
def view_shape_grad(*args, **kwargs):
    """
    C++ interface function for view_shape_grad in dygraph.
    """
def view_slice(input: paddle.Tensor, begin_idx: int, end_idx: int) -> paddle.Tensor:
    """
    C++ interface function for view_slice in dygraph.
    """
def viterbi_decode(potentials: paddle.Tensor, transition_params: paddle.Tensor, lengths: paddle.Tensor, include_bos_eos_tag: bool = True) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for viterbi_decode in dygraph.
    """
def warpctc(logits: paddle.Tensor, label: paddle.Tensor, logits_length: paddle.Tensor, labels_length: paddle.Tensor, blank: int = 0, norm_by_times: bool = False) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for warpctc in dygraph.
    """
def warpctc_grad(*args, **kwargs):
    """
    C++ interface function for warpctc_grad in dygraph.
    """
def warprnnt(input: paddle.Tensor, label: paddle.Tensor, input_lengths: paddle.Tensor, label_lengths: paddle.Tensor, blank: int = 0, fastemit_lambda: float = 0.0) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for warprnnt in dygraph.
    """
def warprnnt_grad(*args, **kwargs):
    """
    C++ interface function for warprnnt_grad in dygraph.
    """
def weight_dequantize(x: paddle.Tensor, scale: paddle.Tensor, algo: str = "weight_only_int8", out_dtype: paddle._typing.DTypeLike = "DataType::FLOAT16", group_size: int = -1) -> paddle.Tensor:
    """
    C++ interface function for weight_dequantize in dygraph.
    """
def weight_only_linear(x: paddle.Tensor, weight: paddle.Tensor, bias: paddle.Tensor, weight_scale: paddle.Tensor, weight_dtype: str, arch: int = 80, group_size: int = -1) -> paddle.Tensor:
    """
    C++ interface function for weight_only_linear in dygraph.
    """
def weight_only_linear_grad(*args, **kwargs):
    """
    C++ interface function for weight_only_linear_grad in dygraph.
    """
def weight_quantize(x: paddle.Tensor, algo: str = "weight_only_int8", arch: int = 80, group_size: int = -1) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for weight_quantize in dygraph.
    """
def weighted_sample_neighbors(row: paddle.Tensor, colptr: paddle.Tensor, edge_weight: paddle.Tensor, input_nodes: paddle.Tensor, eids: paddle.Tensor, sample_size: int, return_eids: bool) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for weighted_sample_neighbors in dygraph.
    """
def where(condition: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for where in dygraph.
    """
def where_(condition: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    C++ interface function for where_ in dygraph.
    """
def where_grad(*args, **kwargs):
    """
    C++ interface function for where_grad in dygraph.
    """
def yolo_box(x: paddle.Tensor, img_size: paddle.Tensor, anchors: list[int] = [], class_num: int = 1, conf_thresh: float = 0.01, downsample_ratio: int = 32, clip_bbox: bool = True, scale_x_y: float = 1.0, iou_aware: bool = False, iou_aware_factor: float = 0.5) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for yolo_box in dygraph.
    """
def yolo_box_head(x: paddle.Tensor, anchors: list[int], class_num: int) -> paddle.Tensor:
    """
    C++ interface function for yolo_box_head in dygraph.
    """
def yolo_box_post(boxes0: paddle.Tensor, boxes1: paddle.Tensor, boxes2: paddle.Tensor, image_shape: paddle.Tensor, image_scale: paddle.Tensor, anchors0: list[int], anchors1: list[int], anchors2: list[int], class_num: int, conf_thresh: float, downsample_ratio0: int, downsample_ratio1: int, downsample_ratio2: int, clip_bbox: bool, scale_x_y: float, nms_threshold: float) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for yolo_box_post in dygraph.
    """
def yolo_loss(x: paddle.Tensor, gt_box: paddle.Tensor, gt_label: paddle.Tensor, gt_score: paddle.Tensor, anchors: list[int] = [], anchor_mask: list[int] = [], class_num: int = 1, ignore_thresh: float = 0.7, downsample_ratio: int = 32, use_label_smooth: bool = True, scale_x_y: float = 1.0) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    C++ interface function for yolo_loss in dygraph.
    """
def yolo_loss_grad(*args, **kwargs):
    """
    C++ interface function for yolo_loss_grad in dygraph.
    """
def zeros(shape: list[int], dtype: paddle._typing.DTypeLike = "DataType::FLOAT32", place: paddle._typing.PlaceLike = "CPUPlace()") -> paddle.Tensor:
    """
    C++ interface function for zeros in dygraph.
    """
def zeros_like(x: paddle.Tensor, dtype: paddle._typing.DTypeLike = "DataType::UNDEFINED", place: paddle._typing.PlaceLike = {}) -> paddle.Tensor:
    """
    C++ interface function for zeros_like in dygraph.
    """
