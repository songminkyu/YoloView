"""
C++ core of PaddlePaddle
"""
from __future__ import annotations
import numpy.typing as npt
import typing_extensions
import datetime
import numpy
import paddle
from paddle import dtype as DataType
import pybind11_stubgen.typing_ext
import typing
from . import eager
from . import op_proto_and_checker_maker
from . import pir
from . import var_names
__all__ = ['AESCipher', 'ALL_DTYPE', 'AllreduceOptions', 'AmpAttrs', 'AmpLevel', 'AnalysisConfig', 'AnalysisPredictor', 'AttrType', 'AttributeMatchGuard', 'BFLOAT16', 'BOOL', 'BarrierOptions', 'BlockDesc', 'BoxPS', 'BroadcastOptions', 'COMPLEX128', 'COMPLEX64', 'CPUPlace', 'CUDAEvent', 'CUDAPinnedPlace', 'CUDAPlace', 'CUDAStream', 'Cipher', 'CipherFactory', 'CipherUtils', 'CommContextManager', 'CompiledProgram', 'CostData', 'CostInfo', 'CostModel', 'CpuPassStrategy', 'CustomDeviceEvent', 'CustomDeviceStream', 'CustomPlace', 'DataType', 'Dataset', 'DenseTensor', 'DenseTensorArray', 'DenseTensorBlockingQueue', 'Device', 'DeviceCapability', 'DeviceContext', 'DeviceMesh', 'DevicePythonNode', 'DeviceType', 'DistTensorSpec', 'DtypeMatchGuard', 'EOFException', 'EagerReducer', 'EnforceNotMet', 'EventSortingKey', 'Executor', 'ExecutorPrepareContext', 'FLOAT16', 'FLOAT32', 'FLOAT64', 'FLOAT8_E4M3FN', 'FLOAT8_E5M2', 'FetchList', 'FetchUnmergedList', 'Fleet', 'Function', 'FunctionInfo', 'GatherOptions', 'Generator', 'GeneratorState', 'GlobalVarGetterSetterRegistry', 'Gloo', 'GpuPassStrategy', 'GradNodeBase', 'Graph', 'GuardBase', 'GuardGroup', 'HostPythonNode', 'INT16', 'INT32', 'INT64', 'INT8', 'IPUPlace', 'IdMatchGuard', 'InstanceCheckGuard', 'InternalUtils', 'IterableDatasetWrapper', 'Job', 'LambdaGuard', 'Layer', 'LayerMatchGuard', 'LengthMatchGuard', 'Link', 'LinkCapability', 'LoDTensor', 'Load', 'Machine', 'MemPythonNode', 'MultiDeviceFeedReader', 'NativeConfig', 'NativePaddlePredictor', 'Node', 'NodePostHookRemoveHelper', 'O0', 'O1', 'O2', 'O3', 'OD', 'OpAttrInfo', 'OpBugfixInfo', 'OpCheckpoint', 'OpDesc', 'OpInputOutputInfo', 'OpUpdateBase', 'OpUpdateInfo', 'OpUpdateType', 'OpVersion', 'OpVersionDesc', 'Operator', 'OperatorDistAttr', 'OrderedMultiDeviceDenseTensorBlockingQueue', 'OrderedMultiDeviceFeedReader', 'P2POption', 'PSTRING', 'PToRReshardFunction', 'PToRReshardFunctionCrossMesh', 'PToSReshardFunction', 'PaddleBuf', 'PaddleDType', 'PaddleDataLayout', 'PaddleInferPredictor', 'PaddleInferTensor', 'PaddlePassBuilder', 'PaddlePlace', 'PaddlePredictor', 'PaddleTensor', 'ParallelStrategy', 'Partial', 'Pass', 'PassBuilder', 'PassStrategy', 'PassVersionChecker', 'Place', 'Placement', 'Plan', 'PredictorPool', 'ProcessGroup', 'ProcessGroupIdMap', 'ProcessMesh', 'ProfilerOptions', 'ProfilerState', 'ProgramDesc', 'Property', 'RToPReshardFunction', 'RToPReshardFunctionCrossMesh', 'RToSReshardFunction', 'RToSReshardFunctionCrossMesh', 'RangeMatchGuard', 'Reader', 'ReduceOp', 'ReduceOptions', 'ReduceType', 'Replicate', 'ReshardFunction', 'SToPReshardFunction', 'SToRReshardFunction', 'SToRReshardFunctionCrossMesh', 'SToSReshardFunction', 'SameNdMeshReshardFunction', 'SameStatusReshardFunction', 'Scalar', 'Scope', 'SelectedRows', 'ShapeMatchGuard', 'ShapeMode', 'Shard', 'SparseCooTensor', 'SpmdRule', 'StandaloneExecutor', 'Store', 'TCPStore', 'TRTEngineParams', 'Tensor', 'TensorDistAttr', 'Tracer', 'TracerEventType', 'TracerMemEventType', 'TracerOption', 'TrainerBase', 'TypeMatchGuard', 'UINT16', 'UINT32', 'UINT64', 'UINT8', 'UNDEFINED', 'ValueMatchGuard', 'VarDesc', 'Variable', 'XPUPlace', 'XToRShrinkReshardFunction', 'XpuConfig', 'ZeroCopyTensor', 'apply_pass', 'autotune_status', 'broadcast_shape', 'build_adjacency_list', 'calc_place_hash', 'call_decomp_rule', 'call_decomp_vjp', 'call_vjp', 'clear_device_manager', 'clear_executor_cache', 'clear_gradients', 'clear_kernel_factory', 'clear_low_precision_op_list', 'clear_shape_info', 'contains_spmd_rule', 'convert_to_mixed_precision_bind', 'copy_tensor', 'create_empty_tensors_with_values', 'create_empty_tensors_with_var_descs', 'create_or_get_global_tcp_store', 'create_paddle_predictor', 'create_predictor', 'create_py_reader', 'default_cpu_generator', 'default_cuda_generator', 'default_custom_device_generator', 'default_xpu_generator', 'deserialize_pir_program', 'device_memory_stat_current_value', 'device_memory_stat_peak_value', 'device_memory_stat_reset_peak_value', 'diff_tensor_shape', 'disable_autotune', 'disable_layout_autotune', 'disable_memory_recorder', 'disable_op_info_recorder', 'disable_profiler', 'disable_signal_handler', 'dtensor_from_local', 'dtensor_to_local', 'dygraph_partial_grad', 'dygraph_run_backward', 'eager', 'eager_assign_group_by_size', 'empty_var_name', 'enable_autotune', 'enable_layout_autotune', 'enable_memory_recorder', 'enable_op_info_recorder', 'enable_profiler', 'eval_frame_no_skip_codes', 'eval_frame_skip_file_prefix', 'finfo', 'from_dlpack', 'get_all_custom_device_type', 'get_all_device_type', 'get_all_op_names', 'get_all_op_protos', 'get_attribute_type', 'get_available_custom_device', 'get_available_device', 'get_custom_device_count', 'get_fetch_variable', 'get_float_stats', 'get_grad_op_desc', 'get_int_stats', 'get_low_precision_op_list', 'get_no_need_buffer_values', 'get_num_bytes_of_data_type', 'get_op_attrs_default_value', 'get_op_extra_attrs', 'get_op_version_map', 'get_pass', 'get_phi_spmd_rule', 'get_promote_dtype_old_ir', 'get_random_seed_generator', 'get_trt_compile_version', 'get_trt_runtime_version', 'get_value_shape_range_info', 'get_variable_tensor', 'get_version', 'globals', 'grad_var_suffix', 'graph_num', 'graph_safe_remove_nodes', 'has_circle', 'has_comp_grad_op_maker', 'has_custom_getattro', 'has_custom_vjp', 'has_decomp_rule', 'has_decomp_vjp', 'has_empty_grad_op_maker', 'has_grad_op_maker', 'has_infer_inplace', 'has_non_empty_grad_op_maker', 'has_vjp', 'host_memory_stat_current_value', 'host_memory_stat_peak_value', 'host_memory_stat_reset_peak_value', 'iinfo', 'infer_no_need_buffer_slots', 'init_default_kernel_signatures', 'init_dense_tensor_blocking_queue', 'init_devices', 'init_gflags', 'init_glog', 'init_memory_method', 'init_tensor_operants', 'is_bfloat16_supported', 'is_common_dtype_for_scalar', 'is_compiled_with_avx', 'is_compiled_with_brpc', 'is_compiled_with_cinn', 'is_compiled_with_cuda', 'is_compiled_with_cudnn_frontend', 'is_compiled_with_custom_device', 'is_compiled_with_dist', 'is_compiled_with_distribute', 'is_compiled_with_flagcx', 'is_compiled_with_ipu', 'is_compiled_with_mkldnn', 'is_compiled_with_mpi', 'is_compiled_with_mpi_aware', 'is_compiled_with_nccl', 'is_compiled_with_rocm', 'is_compiled_with_xpu', 'is_cuda_graph_capturing', 'is_float16_supported', 'is_forward_only', 'is_profiler_enabled', 'kAll', 'kAllOpDetail', 'kAutoParallelSuffix', 'kAve', 'kCPU', 'kCUDA', 'kCalls', 'kControlDepVarName', 'kDefault', 'kDisabled', 'kEmptyVarName', 'kGradVarSuffix', 'kMAX', 'kMIN', 'kMax', 'kMin', 'kNewGradSuffix', 'kNoneProcessMeshIndex', 'kOPT', 'kOpDetail', 'kTempVarName', 'kTotal', 'kZeroVarSuffix', 'load_combine_func', 'load_dense_tensor', 'load_dense_tensor_from_memory', 'load_func', 'load_op_meta_info_and_register_op', 'load_profiler_result', 'load_selected_rows', 'load_selected_rows_from_memory', 'merge_guard', 'mt19937_64', 'need_type_promotion_old_ir', 'op_proto_and_checker_maker', 'op_support_gpu', 'op_supported_infos', 'paddle_dtype_size', 'paddle_tensor_to_bytes', 'parse_safe_eager_deletion_skip_vars', 'pir', 'prune', 'prune_backward', 'register_pass', 'register_subgraph_pass', 'reset_profiler', 'reshard', 'run_cmd', 'save_combine_func', 'save_dense_tensor', 'save_dense_tensor_to_memory', 'save_func', 'save_op_version_info', 'save_selected_rows', 'save_selected_rows_to_memory', 'serialize_pir_program', 'set_autotune_range', 'set_checked_op_list', 'set_current_thread_name', 'set_eval_frame', 'set_feed_variable', 'set_nan_inf_debug_path', 'set_nan_inf_stack_limit', 'set_num_threads', 'set_printoptions', 'set_random_seed_generator', 'set_skipped_op_list', 'set_static_op_arg_pre_cast_hook', 'set_tracer_option', 'set_variable', 'shell_execute_cmd', 'sinking_decomp', 'size_of_dtype', 'sot_set_with_graph', 'sot_setup_codes_with_graph', 'start_imperative_gperf_profiler', 'stop_imperative_gperf_profiler', 'sub_mesh_dim', 'supports_avx512f', 'supports_bfloat16', 'supports_bfloat16_fast_performance', 'supports_int8', 'supports_vnni', 'task', 'tensor_from_cuda_array_interface', 'topology_sort', 'touch_dist_mapper', 'tracer_event_type_to_string', 'tracer_mem_event_type_to_string', 'update_autotune_status', 'use_layout_autotune', 'var_names', 'varbase_copy', 'wait_device']
class AESCipher(Cipher):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class AllreduceOptions:
    reduce_op: ReduceOp
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class AmpAttrs:
    """
    """
    _amp_dtype: str
    _amp_level: AmpLevel
    _use_promote: bool
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
class AmpLevel:
    """
    Members:
    
      O0
    
      OD
    
      O1
    
      O2
    
      O3
    """
    O0: typing.ClassVar[AmpLevel]  # value = <AmpLevel.O0: 0>
    O1: typing.ClassVar[AmpLevel]  # value = <AmpLevel.O1: 1>
    O2: typing.ClassVar[AmpLevel]  # value = <AmpLevel.O2: 2>
    O3: typing.ClassVar[AmpLevel]  # value = <AmpLevel.O3: 3>
    OD: typing.ClassVar[AmpLevel]  # value = <AmpLevel.OD: 4>
    __members__: typing.ClassVar[dict[str, AmpLevel]]  # value = {'O0': <AmpLevel.O0: 0>, 'OD': <AmpLevel.OD: 4>, 'O1': <AmpLevel.O1: 1>, 'O2': <AmpLevel.O2: 2>, 'O3': <AmpLevel.O3: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class AnalysisConfig:
    class Precision:
        """
        Members:
        
          Float32
        
          Int8
        
          Half
        
          Bfloat16
        """
        Bfloat16: typing.ClassVar[AnalysisConfig.Precision]  # value = <Precision.Bfloat16: 3>
        Float32: typing.ClassVar[AnalysisConfig.Precision]  # value = <Precision.Float32: 0>
        Half: typing.ClassVar[AnalysisConfig.Precision]  # value = <Precision.Half: 2>
        Int8: typing.ClassVar[AnalysisConfig.Precision]  # value = <Precision.Int8: 1>
        __members__: typing.ClassVar[dict[str, AnalysisConfig.Precision]]  # value = {'Float32': <Precision.Float32: 0>, 'Int8': <Precision.Int8: 1>, 'Half': <Precision.Half: 2>, 'Bfloat16': <Precision.Bfloat16: 3>}
        @staticmethod
        def _pybind11_conduit_v1_(*args, **kwargs):
            ...
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: int) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    Bfloat16: typing.ClassVar[AnalysisConfig.Precision]  # value = <Precision.Bfloat16: 3>
    Float32: typing.ClassVar[AnalysisConfig.Precision]  # value = <Precision.Float32: 0>
    Half: typing.ClassVar[AnalysisConfig.Precision]  # value = <Precision.Half: 2>
    Int8: typing.ClassVar[AnalysisConfig.Precision]  # value = <Precision.Int8: 1>
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: AnalysisConfig) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: str) -> None:
        ...
    def collect_shape_range_info(self, arg0: str) -> None:
        ...
    def cpu_math_library_num_threads(self) -> int:
        ...
    def delete_pass(self, arg0: str) -> None:
        ...
    def disable_glog_info(self) -> None:
        ...
    def disable_gpu(self) -> None:
        ...
    def disable_mkldnn(self) -> None:
        ...
    def disable_mkldnn_fc_passes(self) -> None:
        """
                    Disable Mkldnn FC
                    Returns:
                        None.
        
                    Examples:
                        .. code-block:: python
        
                            >>> from paddle.inference import Config
        
                            >>> config = Config("")
                            >>> config.enable_mkldnn()
                            >>> config.disable_mkldnn_fc_passes()
        """
    def disable_onnxruntime(self) -> None:
        ...
    def enable_cinn(self) -> None:
        ...
    def enable_custom_device(self, device_type: str, device_id: int = 0, precision: AnalysisConfig.Precision = typing.Any) -> None:
        ...
    def enable_custom_passes(self, passes: list[str] = [], custom_pass_only: bool = False) -> None:
        ...
    def enable_ipu(self, ipu_device_num: int = 1, ipu_micro_batch_size: int = 1, ipu_enable_pipelining: bool = False, ipu_batches_per_step: int = 1) -> None:
        ...
    def enable_low_precision_io(self, x: bool = True) -> None:
        ...
    def enable_memory_optim(self, x: bool = True) -> None:
        ...
    def enable_mkldnn(self) -> None:
        ...
    def enable_mkldnn_bfloat16(self) -> None:
        ...
    def enable_mkldnn_int8(self, mkldnn_int8_enabled_op_types: set[str] = set()) -> None:
        ...
    def enable_new_executor(self, x: bool = True) -> None:
        ...
    def enable_new_ir(self, x: bool = True) -> None:
        ...
    def enable_onnxruntime(self) -> None:
        ...
    def enable_openvino_engine(self, inference_precision: AnalysisConfig.Precision = typing.Any) -> None:
        ...
    def enable_ort_optimization(self) -> None:
        ...
    def enable_profile(self) -> None:
        ...
    def enable_save_optim_model(self, save_optimized_model: bool = False) -> None:
        ...
    def enable_tensorrt_dla(self, dla_core: int = 0) -> None:
        ...
    def enable_tensorrt_engine(self, workspace_size: int = 1073741824, max_batch_size: int = 1, min_subgraph_size: int = 3, precision_mode: AnalysisConfig.Precision = typing.Any, use_static: bool = False, use_calib_mode: bool = True, use_cuda_graph: bool = False) -> None:
        ...
    def enable_tensorrt_explicit_quantization(self) -> None:
        ...
    def enable_tensorrt_inspector(self, inspector_serialize: bool = False) -> None:
        ...
    def enable_tensorrt_memory_optim(self, engine_memory_sharing: bool = True, sharing_identifier: int = 0) -> None:
        ...
    def enable_tensorrt_varseqlen(self) -> None:
        ...
    def enable_tuned_tensorrt_dynamic_shape(self, shape_range_info_path: str = '', allow_build_at_runtime: bool = True) -> None:
        ...
    def enable_use_gpu(self, memory_pool_init_size_mb: int, device_id: int = 0, precision_mode: AnalysisConfig.Precision = typing.Any) -> None:
        ...
    def enable_xpu(self, l3_size: int = 16777216, l3_locked: bool = False, conv_autotune: bool = False, conv_autotune_file: str = '', transformer_encoder_precision: str = 'int16', transformer_encoder_adaptive_seqlen: bool = False, enable_multi_stream: bool = False) -> None:
        ...
    def exp_disable_mixed_precision_ops(self, arg0: set[str]) -> None:
        ...
    def exp_disable_tensorrt_dynamic_shape_ops(self, arg0: bool) -> None:
        ...
    def exp_disable_tensorrt_ops(self, arg0: list[str]) -> None:
        ...
    def exp_disable_tensorrt_subgraph(self, arg0: list[str]) -> None:
        ...
    def exp_enable_mixed_precision_ops(self, arg0: set[str]) -> None:
        ...
    def exp_enable_use_cutlass(self) -> None:
        ...
    def exp_specify_tensorrt_subgraph_precision(self, arg0: list[str], arg1: list[str], arg2: list[str]) -> None:
        ...
    def fraction_of_gpu_memory_for_pool(self) -> float:
        ...
    def glog_info_disabled(self) -> bool:
        ...
    def gpu_device_id(self) -> int:
        ...
    def ir_optim(self) -> bool:
        ...
    def load_ipu_config(self, config_path: str) -> None:
        ...
    def mark_trt_engine_outputs(self, output_tensor_names: list[str] = []) -> None:
        ...
    def memory_pool_init_size_mb(self) -> int:
        ...
    def mkldnn_enabled(self) -> bool:
        ...
    def mkldnn_int8_enabled(self) -> bool:
        ...
    def model_dir(self) -> str:
        ...
    def model_from_memory(self) -> bool:
        ...
    def new_ir_enabled(self) -> bool:
        ...
    def onnxruntime_enabled(self) -> bool:
        ...
    def openvino_engine_enabled(self) -> bool:
        ...
    def params_file(self) -> str:
        ...
    def pass_builder(self) -> typing.Any:
        ...
    def prog_file(self) -> str:
        ...
    def set_bfloat16_op(self, arg0: set[str]) -> None:
        ...
    def set_cpu_math_library_num_threads(self, arg0: int) -> None:
        ...
    def set_ipu_config(self, ipu_enable_fp16: bool = False, ipu_replica_num: int = 1, ipu_available_memory_proportion: float = 1.0, ipu_enable_half_partial: bool = False, ipu_enable_model_runtime_executor: bool = False) -> None:
        ...
    def set_ipu_custom_info(self, ipu_custom_ops_info: list[list[str]] = [], ipu_custom_patterns: dict[str, bool] = {}) -> None:
        ...
    def set_mkldnn_cache_capacity(self, capacity: int = 0) -> None:
        ...
    def set_mkldnn_op(self, arg0: set[str]) -> None:
        ...
    @typing.overload
    def set_model(self, arg0: str) -> None:
        ...
    @typing.overload
    def set_model(self, arg0: str, arg1: str) -> None:
        ...
    def set_model_buffer(self, arg0: str, arg1: int, arg2: str, arg3: int) -> None:
        ...
    def set_optim_cache_dir(self, arg0: str) -> None:
        ...
    def set_optimization_level(self, opt_level: int = 2) -> None:
        ...
    def set_params_file(self, arg0: str) -> None:
        ...
    def set_prog_file(self, arg0: str) -> None:
        ...
    def set_tensorrt_optimization_level(self, arg0: int) -> None:
        ...
    def set_trt_dynamic_shape_info(self, min_input_shape: dict[str, list[int]] = {}, max_input_shape: dict[str, list[int]] = {}, optim_input_shape: dict[str, list[int]] = {}, disable_trt_plugin_fp16: bool = False) -> None:
        ...
    def set_xpu_config(self, arg0: XpuConfig) -> None:
        ...
    def set_xpu_device_id(self, device_id: int = 0) -> None:
        ...
    def shape_range_info_collected(self) -> bool:
        ...
    def shape_range_info_path(self) -> str:
        ...
    def specify_input_name(self) -> bool:
        ...
    def summary(self) -> str:
        ...
    def switch_ir_debug(self, x: int = True, passes: list[str] = []) -> None:
        ...
    def switch_ir_optim(self, x: int = True) -> None:
        ...
    def switch_specify_input_names(self, x: bool = True) -> None:
        ...
    def switch_use_feed_fetch_ops(self, x: int = True) -> None:
        ...
    def tensorrt_dla_enabled(self) -> bool:
        ...
    def tensorrt_dynamic_shape_enabled(self) -> bool:
        ...
    def tensorrt_engine_enabled(self) -> bool:
        ...
    def tensorrt_explicit_quantization_enabled(self) -> bool:
        ...
    def tensorrt_inspector_enabled(self) -> bool:
        ...
    def tensorrt_optimization_level(self) -> int:
        ...
    def tensorrt_precision_mode(self) -> AnalysisConfig.Precision:
        ...
    def tensorrt_varseqlen_enabled(self) -> bool:
        ...
    def to_native_config(self) -> NativeConfig:
        ...
    def trt_allow_build_at_runtime(self) -> bool:
        ...
    def tuned_tensorrt_dynamic_shape(self) -> bool:
        ...
    def use_feed_fetch_ops_enabled(self) -> bool:
        ...
    def use_gpu(self) -> bool:
        ...
    def use_optimized_model(self, x: bool = True) -> None:
        ...
    def use_xpu(self) -> bool:
        ...
    def xpu_config(self) -> XpuConfig:
        ...
    def xpu_device_id(self) -> int:
        ...
class AnalysisPredictor(PaddlePredictor):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: AnalysisConfig) -> None:
        ...
    def analysis_argument(self) -> typing.Any:
        ...
    def clear_intermediate_tensor(self) -> None:
        ...
    def clone(self) -> PaddlePredictor:
        ...
    def create_feed_fetch_var(self, arg0: _Scope) -> None:
        ...
    def get_input_names(self) -> list[str]:
        ...
    def get_input_tensor(self, arg0: str) -> typing.Any:
        ...
    def get_input_tensor_shape(self) -> dict[str, list[int]]:
        ...
    def get_output_names(self) -> list[str]:
        ...
    def get_output_tensor(self, arg0: str) -> typing.Any:
        ...
    def get_serialized_program(self) -> str:
        ...
    def init(self, arg0: _Scope, arg1: ProgramDesc) -> bool:
        ...
    def optimize_inference_program(self) -> None:
        ...
    def prepare_argument(self) -> None:
        ...
    def prepare_feed_fetch(self) -> None:
        ...
    def program(self) -> ProgramDesc:
        ...
    def run(self, arg0: list[PaddleTensor]) -> list[PaddleTensor]:
        ...
    def scope(self) -> _Scope:
        ...
    def try_shrink_memory(self) -> int:
        ...
    def zero_copy_run(self, switch_stream: bool = False) -> bool:
        ...
class AttrType:
    """
    
    
    Members:
    
      INT
    
      INTS
    
      LONG
    
      LONGS
    
      FLOAT
    
      FLOATS
    
      FLOAT64
    
      FLOAT64S
    
      STRING
    
      STRINGS
    
      BOOL
    
      BOOLS
    
      BLOCK
    
      BLOCKS
    
      VAR
    
      VARS
    
      SCALAR
    
      SCALARS
    """
    BLOCK: typing.ClassVar[AttrType]  # value = <AttrType.BLOCK: 8>
    BLOCKS: typing.ClassVar[AttrType]  # value = <AttrType.BLOCKS: 10>
    BOOL: typing.ClassVar[AttrType]  # value = <AttrType.BOOL: 6>
    BOOLS: typing.ClassVar[AttrType]  # value = <AttrType.BOOLS: 7>
    FLOAT: typing.ClassVar[AttrType]  # value = <AttrType.FLOAT: 1>
    FLOAT64: typing.ClassVar[AttrType]  # value = <AttrType.FLOAT64: 15>
    FLOAT64S: typing.ClassVar[AttrType]  # value = <AttrType.FLOAT64S: 12>
    FLOATS: typing.ClassVar[AttrType]  # value = <AttrType.FLOATS: 4>
    INT: typing.ClassVar[AttrType]  # value = <AttrType.INT: 0>
    INTS: typing.ClassVar[AttrType]  # value = <AttrType.INTS: 3>
    LONG: typing.ClassVar[AttrType]  # value = <AttrType.LONG: 9>
    LONGS: typing.ClassVar[AttrType]  # value = <AttrType.LONGS: 11>
    SCALAR: typing.ClassVar[AttrType]  # value = <AttrType.SCALAR: 16>
    SCALARS: typing.ClassVar[AttrType]  # value = <AttrType.SCALARS: 17>
    STRING: typing.ClassVar[AttrType]  # value = <AttrType.STRING: 2>
    STRINGS: typing.ClassVar[AttrType]  # value = <AttrType.STRINGS: 5>
    VAR: typing.ClassVar[AttrType]  # value = <AttrType.VAR: 13>
    VARS: typing.ClassVar[AttrType]  # value = <AttrType.VARS: 14>
    __members__: typing.ClassVar[dict[str, AttrType]]  # value = {'INT': <AttrType.INT: 0>, 'INTS': <AttrType.INTS: 3>, 'LONG': <AttrType.LONG: 9>, 'LONGS': <AttrType.LONGS: 11>, 'FLOAT': <AttrType.FLOAT: 1>, 'FLOATS': <AttrType.FLOATS: 4>, 'FLOAT64': <AttrType.FLOAT64: 15>, 'FLOAT64S': <AttrType.FLOAT64S: 12>, 'STRING': <AttrType.STRING: 2>, 'STRINGS': <AttrType.STRINGS: 5>, 'BOOL': <AttrType.BOOL: 6>, 'BOOLS': <AttrType.BOOLS: 7>, 'BLOCK': <AttrType.BLOCK: 8>, 'BLOCKS': <AttrType.BLOCKS: 10>, 'VAR': <AttrType.VAR: 13>, 'VARS': <AttrType.VARS: 14>, 'SCALAR': <AttrType.SCALAR: 16>, 'SCALARS': <AttrType.SCALARS: 17>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class AttributeMatchGuard(GuardBase):
    """
    AttributeMatchGuard Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, obj: typing.Any, attr_name: str) -> None:
        ...
class BarrierOptions:
    device_id: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class BlockDesc:
    """
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def _insert_op(self, arg0: int) -> typing.Any:
        ...
    def _move_from(self, arg0: BlockDesc) -> None:
        ...
    def _prepend_op(self) -> typing.Any:
        ...
    def _remove_op(self, arg0: int, arg1: int) -> None:
        ...
    def _remove_var(self, arg0: bytes) -> None:
        ...
    def _rename_var(self, arg0: bytes, arg1: bytes) -> None:
        ...
    def _set_forward_block_idx(self, arg0: int) -> None:
        ...
    def all_vars(self) -> list[typing.Any]:
        ...
    def append_op(self) -> typing.Any:
        ...
    def find_var(self, arg0: bytes) -> typing.Any:
        ...
    def find_var_recursive(self, arg0: bytes) -> typing.Any:
        ...
    def get_forward_block_idx(self) -> int:
        ...
    def has_var(self, arg0: bytes) -> bool:
        ...
    def has_var_recursive(self, arg0: bytes) -> bool:
        ...
    def op(self, arg0: int) -> typing.Any:
        ...
    def op_size(self) -> int:
        ...
    def serialize_to_string(self) -> bytes:
        ...
    def set_parent_idx(self, arg0: int) -> None:
        ...
    def var(self, arg0: bytes) -> typing.Any:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def parent(self) -> int:
        ...
class BoxPS:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: typing.Any) -> None:
        ...
    def begin_pass(self) -> None:
        ...
    def end_pass(self, arg0: bool) -> None:
        ...
    def load_into_memory(self) -> None:
        ...
    def preload_into_memory(self) -> None:
        ...
    def set_date(self, arg0: int, arg1: int, arg2: int) -> None:
        ...
    def slots_shuffle(self, arg0: set[str]) -> None:
        ...
    def wait_feed_pass_done(self) -> None:
        ...
class BroadcastOptions:
    source_rank: int
    source_root: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class CPUPlace:
    """
    
        CPUPlace is a descriptor of a device.
        It represents a CPU device on which a tensor will be allocated and a model will run.
    
        Examples:
            .. code-block:: python
    
                >>> import paddle
                >>> cpu_place = paddle.CPUPlace()
    
            
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @typing.overload
    def _equals(self, arg0: typing.Any) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: XPUPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CUDAPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CPUPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: typing.Any) -> bool:
        ...
    def _type(self) -> int:
        ...
class CUDAEvent:
    """
    
          The handle of the CUDA event.
    
          Parameters:
              enable_timing(bool, optional): Whether the event will measure time. Default: False.
              blocking(bool, optional): Whether the wait() func will be blocking. Default: False;
              interprocess(bool, optional): Whether the event can be shared between processes. Default: False.
    
          Examples:
              .. code-block:: python
    
                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> event = paddle.device.cuda.Event()
    
          
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False) -> None:
        ...
class CUDAPinnedPlace:
    """
    
        CUDAPinnedPlace is a descriptor of a device.
        It refers to the page locked memory allocated by the CUDA function `cudaHostAlloc()` in the host memory.
        The host operating system will not paging and exchanging the memory.
        It can be accessed through direct memory access technology to speed up the copy of data between the host and GPU.
        For more information on CUDA data transfer and `pinned memory`,
        please refer to `official document <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory>`_ .
    
        Examples:
            .. code-block:: python
    
                >>> # doctest: +REQUIRES(env:GPU)
                >>> import paddle
                >>> place = paddle.CUDAPinnedPlace()
    
            
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @typing.overload
    def _equals(self, arg0: typing.Any) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CUDAPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: XPUPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CPUPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CUDAPinnedPlace) -> bool:
        ...
    def _type(self) -> int:
        ...
class CUDAPlace:
    """
    
    
        CUDAPlace is a descriptor of a device.
        It represents a GPU device allocated or to be allocated with Tensor.
        Each CUDAPlace has a dev_id to indicate the graphics card ID represented by the current CUDAPlace,
        staring from 0.
        The memory of CUDAPlace with different dev_id is not accessible.
        Numbering here refers to the logical ID of the visible graphics card, not the actual ID of the graphics card.
        You can set visible GPU devices by setting the `CUDA_VISIBLE_DEVICES` environment variable.
        When the program starts, visible GPU devices will be numbered from 0.
        If `CUDA_VISIBLE_DEVICES` is not set, all devices are visible by default,
        and the logical ID is the same as the actual ID.
    
        Parameters:
            id (int): GPU device ID.
    
        Examples:
            .. code-block:: python
    
                >>> # doctest: +REQUIRES(env:GPU)
                >>> import paddle
                >>> place = paddle.CUDAPlace(0)
    
            
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: int) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class CUDAStream:
    """
    
          The handle of the CUDA stream.
    
          Parameters:
              device(paddle.CUDAPlace()|int|None, optional): The device which wanted to allocate the stream.
                  If device is None or negative integer, device will be the current device.
                  If device is positive integer, it must less than the device count. Default: None.
              priority(int|None, optional): The priority of stream. The priority can be 1(high) or 2(normal).
                  If priority is None, the priority is 2(normal). Default: None.
    
          Examples:
              .. code-block:: python
    
                  >>> # doctest: +REQUIRES(env:GPU)
                  >>> import paddle
                  >>> s1 = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
                  >>> s2 = paddle.device.cuda.Stream(0, 1)
                  >>> s3 = paddle.device.cuda.Stream()
    
          
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, device: typing.Any = None, priority: int = 2) -> None:
        ...
    @typing.overload
    def __init__(self, device: int = -1, priority: int = 2) -> None:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
class Cipher:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def decrypt(self, arg0: str, arg1: str) -> bytes:
        ...
    def decrypt_from_file(self, arg0: str, arg1: str) -> bytes:
        ...
    def encrypt(self, arg0: str, arg1: str) -> bytes:
        ...
    def encrypt_to_file(self, arg0: str, arg1: str, arg2: str) -> None:
        ...
class CipherFactory:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @staticmethod
    def create_cipher(config_file: str = '') -> Cipher:
        ...
    def __init__(self) -> None:
        ...
class CipherUtils:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @staticmethod
    def gen_key(arg0: int) -> bytes:
        ...
    @staticmethod
    def gen_key_to_file(arg0: int, arg1: str) -> bytes:
        ...
    @staticmethod
    def read_key_from_file(arg0: str) -> bytes:
        ...
class CommContextManager:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @staticmethod
    def set_device_id(arg0: int) -> None:
        ...
    def set_store(self, arg0: Store) -> None:
        ...
class CompiledProgram:
    class BuildStrategy:
        """
        
            BuildStrategy allows the user to more preciously control how to
            build the SSA Graph in CompiledProgram by setting the property.
        
            Returns:
                BuildStrategy: An BuildStrategy object.
        
            Examples:
                .. code-block:: python
        
                    >>> import paddle
                    >>> import paddle.static as static
        
                    >>> paddle.enable_static()
        
                    >>> data = static.data(name="x", shape=[None, 1], dtype="float32")
                    >>> hidden = static.nn.fc(data, size=10)
                    >>> loss = paddle.mean(hidden)
                    >>> paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
        
                    >>> build_strategy = static.BuildStrategy()
                    >>> build_strategy.enable_inplace = True
                    >>> build_strategy.memory_optimize = True
                    >>> build_strategy.reduce_strategy = static.BuildStrategy.ReduceStrategy.Reduce
                    >>> program = static.CompiledProgram(static.default_main_program(), build_strategy=build_strategy)
        """
        class ReduceStrategy:
            """
            Members:
            
              Reduce
            
              AllReduce
            
              _NoReduce
            """
            AllReduce: typing.ClassVar[CompiledProgram.BuildStrategy.ReduceStrategy]  # value = <ReduceStrategy.AllReduce: 0>
            Reduce: typing.ClassVar[CompiledProgram.BuildStrategy.ReduceStrategy]  # value = <ReduceStrategy.Reduce: 1>
            _NoReduce: typing.ClassVar[CompiledProgram.BuildStrategy.ReduceStrategy]  # value = <ReduceStrategy._NoReduce: 2>
            __members__: typing.ClassVar[dict[str, CompiledProgram.BuildStrategy.ReduceStrategy]]  # value = {'Reduce': <ReduceStrategy.Reduce: 1>, 'AllReduce': <ReduceStrategy.AllReduce: 0>, '_NoReduce': <ReduceStrategy._NoReduce: 2>}
            @staticmethod
            def _pybind11_conduit_v1_(*args, **kwargs):
                ...
            def __eq__(self, other: typing.Any) -> bool:
                ...
            def __getstate__(self) -> int:
                ...
            def __hash__(self) -> int:
                ...
            def __index__(self) -> int:
                ...
            def __init__(self, value: int) -> None:
                ...
            def __int__(self) -> int:
                ...
            def __ne__(self, other: typing.Any) -> bool:
                ...
            def __repr__(self) -> str:
                ...
            def __setstate__(self, state: int) -> None:
                ...
            def __str__(self) -> str:
                ...
            @property
            def name(self) -> str:
                ...
            @property
            def value(self) -> int:
                ...
        allow_cuda_graph_capture: bool
        async_mode: bool
        bkcl_comm_num: int
        cache_runtime_context: bool
        enable_addto: bool
        enable_backward_optimizer_op_deps: bool
        enable_inplace: bool
        fuse_all_optimizer_ops: bool
        fuse_all_reduce_ops: bool
        hierarchical_allreduce_inter_nranks: int
        mkldnn_enabled_op_types: set[str]
        nccl_comm_num: int
        num_trainers: int
        trainer_id: int
        trainers_endpoints: list[str]
        use_hierarchical_allreduce: bool
        @staticmethod
        def _pybind11_conduit_v1_(*args, **kwargs):
            ...
        def __init__(self) -> None:
            ...
        def __str__(self) -> str:
            ...
        def _clear_finalized(self) -> None:
            ...
        def _copy(self) -> CompiledProgram.BuildStrategy:
            ...
        def _finalize_strategy_and_create_passes(self) -> typing.Any:
            """
            Allow user to customized passes. Normally model-specific
                            optimization passes should be defined in this way. BuildStrategy
                            cannot be updated after being finalized.
            """
        @property
        def build_cinn_pass(self) -> bool:
            """
            (bool, optional): build_cinn_pass indicates whether
                                  to lowering some operators in graph into cinn ops
                                  to execute, which will speed up the process of execution.
                                  Default False.
            
                                  Examples:
                                        .. code-block:: python
            
                                            >>> import paddle
                                            >>> import paddle.static as static
                                            >>> paddle.enable_static()
                                            >>> build_strategy = static.BuildStrategy()
                                            >>> build_strategy.build_cinn_pass = True
            """
        @build_cinn_pass.setter
        def build_cinn_pass(self, arg1: bool) -> None:
            ...
        @property
        def debug_graphviz_path(self) -> str:
            """
            (str, optional): debug_graphviz_path indicates the path that
                            writing the SSA Graph to file in the form of graphviz.
                            It is useful for debugging. Default is empty string, that is, ""
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.debug_graphviz_path = "./graph"
            """
        @debug_graphviz_path.setter
        def debug_graphviz_path(self, arg1: str) -> None:
            ...
        @property
        def enable_auto_fusion(self) -> bool:
            """
            (bool, optional): Whether to enable fusing subgraph to a
                            fusion_group. Now we only support fusing subgraph that composed
                            of elementwise-like operators, such as elementwise_add/mul
                            without broadcast and activations.
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.enable_auto_fusion = True
            """
        @enable_auto_fusion.setter
        def enable_auto_fusion(self, arg1: bool) -> None:
            ...
        @property
        def fuse_adamw(self) -> bool:
            """
            (bool, optional): fuse_adamw indicate whether
                            to fuse all adamw optimizers with multi_tensor_adam,
                            it may make the execution faster. Default is False.
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
                                    >>> paddle.enable_static()
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.fuse_adamw = True
            """
        @fuse_adamw.setter
        def fuse_adamw(self, arg1: bool) -> None:
            ...
        @property
        def fuse_bn_act_ops(self) -> bool:
            """
            (bool, optional): fuse_bn_act_ops indicate whether
                            to fuse batch_norm and activation_op,
                            it may make the execution faster. Default is False.
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.fuse_bn_act_ops = True
            """
        @fuse_bn_act_ops.setter
        def fuse_bn_act_ops(self, arg1: bool) -> None:
            ...
        @property
        def fuse_bn_add_act_ops(self) -> bool:
            """
            (bool, optional): fuse_bn_add_act_ops indicate whether
                            to fuse batch_norm, elementwise_add and activation_op,
                            it may make the execution faster. Default is True
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.fuse_bn_add_act_ops = True
            """
        @fuse_bn_add_act_ops.setter
        def fuse_bn_add_act_ops(self, arg1: bool) -> None:
            ...
        @property
        def fuse_broadcast_ops(self) -> bool:
            """
            (bool, optional): fuse_broadcast_op indicates whether
                                  to fuse the broadcast ops. Note that, in Reduce mode,
                                  fusing broadcast ops may make the program faster. Because
                                  fusing broadcast OP equals delaying the execution of all
                                  broadcast Ops, in this case, all nccl streams are used only
                                  for NCCLReduce operations for a period of time. Default False.
            
                                  Examples:
                                        .. code-block:: python
            
                                            >>> import paddle
                                            >>> import paddle.static as static
                                            >>> paddle.enable_static()
            
                                            >>> build_strategy = static.BuildStrategy()
                                            >>> build_strategy.fuse_broadcast_ops = True
            """
        @fuse_broadcast_ops.setter
        def fuse_broadcast_ops(self, arg1: bool) -> None:
            ...
        @property
        def fuse_dot_product_attention(self) -> bool:
            """
            (bool, optional): fuse_dot_product_attention indicate whether
                            to fuse dot product attention,
                            it would make the execution faster. Default is False.
            
                            Examples:
                                .. code-block:: python
            
                                    import paddle
                                    import paddle.static as static
            
                                    paddle.enable_static()
            
                                    build_strategy = static.BuildStrategy()
                                    build_strategy.fuse_dot_product_attention = True
            """
        @fuse_dot_product_attention.setter
        def fuse_dot_product_attention(self, arg1: bool) -> None:
            ...
        @property
        def fuse_elewise_add_act_ops(self) -> bool:
            """
            (bool, optional): fuse_elewise_add_act_ops indicate whether
                            to fuse elementwise_add_op and activation_op,
                            it may make the execution faster. Default is False.
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.fuse_elewise_add_act_ops = True
            """
        @fuse_elewise_add_act_ops.setter
        def fuse_elewise_add_act_ops(self, arg1: bool) -> None:
            ...
        @property
        def fuse_gemm_epilogue(self) -> bool:
            """
            (bool, optional): fuse_gemm_epilogue indicate whether
                            to fuse matmul_op, elemenewist_add_op and activation_op,
                            it may make the execution faster. Default is False.
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.fuse_gemm_epilogue = True
            """
        @fuse_gemm_epilogue.setter
        def fuse_gemm_epilogue(self, arg1: bool) -> None:
            ...
        @property
        def fuse_relu_depthwise_conv(self) -> bool:
            """
            (bool, optional): fuse_relu_depthwise_conv indicate whether
                            to fuse relu and depthwise_conv2d,
                            it will save GPU memory and may make the execution faster.
                            This options is only available in GPU devices.
                            Default is False.
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.fuse_relu_depthwise_conv = True
            """
        @fuse_relu_depthwise_conv.setter
        def fuse_relu_depthwise_conv(self, arg1: bool) -> None:
            ...
        @property
        def fuse_resunit(self) -> bool:
            """
            (bool, optional): fuse_resunit Default is False.
            
                            Examples:
                                .. code-block:: python
            
                                    import paddle
                                    import paddle.static as static
            
                                    paddle.enable_static()
            
                                    build_strategy = static.BuildStrategy()
                                    build_strategy.fuse_resunit = True
            """
        @fuse_resunit.setter
        def fuse_resunit(self, arg1: bool) -> None:
            ...
        @property
        def fused_attention(self) -> bool:
            """
            (bool, optional): fused_attention indicate whether
                            to fuse the whole multi head attention part with one op,
                            it may make the execution faster. Default is False.
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.fused_attention = True
            """
        @fused_attention.setter
        def fused_attention(self, arg1: bool) -> None:
            ...
        @property
        def fused_feedforward(self) -> bool:
            """
            (bool, optional): fused_feedforward indicate whether
                            to fuse the whole feed_forward part with one op,
                            it may make the execution faster. Default is False.
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.fused_feedforward = True
            """
        @fused_feedforward.setter
        def fused_feedforward(self, arg1: bool) -> None:
            ...
        @property
        def memory_optimize(self) -> typing.Any:
            """
            (bool, optional): memory optimize aims to save total memory
                            consumption, set to True to enable it.
            
                            Default None. None means framework would choose to use or not use
                            this strategy automatically. Currently, None means that it is
                            enabled when GC is disabled, and disabled when GC is enabled.
                            True means enabling and False means disabling. Default is None.
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.memory_optimize = True
            """
        @memory_optimize.setter
        def memory_optimize(self, arg1: typing.Any) -> None:
            ...
        @property
        def reduce_strategy(self) -> CompiledProgram.BuildStrategy.ReduceStrategy:
            """
            (fluid.BuildStrategy.ReduceStrategy, optional): there are two reduce
                            strategies in CompiledProgram, AllReduce and Reduce. If you want
                            that all the parameters' optimization are done on all devices independently,
                            you should choose AllReduce; otherwise, if you choose Reduce, all the parameters'
                            optimization will be evenly distributed to different devices, and then
                            broadcast the optimized parameter to other devices.
                            Default is 'AllReduce'.
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.reduce_strategy = static.BuildStrategy.ReduceStrategy.Reduce
            """
        @reduce_strategy.setter
        def reduce_strategy(self, arg1: CompiledProgram.BuildStrategy.ReduceStrategy) -> None:
            ...
        @property
        def sequential_run(self) -> bool:
            """
            (bool, optional): sequential_run is used to let the `StandaloneExecutor` run ops by the
                      order of `ProgramDesc`. Default is False.
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.sequential_run = True
            """
        @sequential_run.setter
        def sequential_run(self, arg1: bool) -> None:
            ...
        @property
        def sync_batch_norm(self) -> bool:
            """
            (bool, optional): sync_batch_norm indicates whether to use
                            synchronous batch normalization which synchronizes the mean
                            and variance through multi-devices in training phase.
                            Current implementation doesn't support FP16 training and CPU.
                            And only synchronous on one machine, not all machines.
                            Default is False.
            
                            Examples:
                                .. code-block:: python
            
                                    >>> import paddle
                                    >>> import paddle.static as static
            
                                    >>> paddle.enable_static()
            
                                    >>> build_strategy = static.BuildStrategy()
                                    >>> build_strategy.sync_batch_norm = True
            """
        @sync_batch_norm.setter
        def sync_batch_norm(self, arg1: bool) -> None:
            ...
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: list[typing.Any], arg1: list[str], arg2: str, arg3: _Scope, arg4: list[_Scope], arg5: CompiledProgram.BuildStrategy, arg6: typing.Any) -> None:
        ...
    def local_scopes(self) -> list[_Scope]:
        ...
class CostData:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def get_op_time_ms(self, arg0: int) -> float:
        ...
    def get_whole_time_ms(self) -> float:
        ...
class CostInfo:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def device_memory_bytes(self) -> int:
        ...
    def total_time(self) -> float:
        ...
class CostModel:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def profile_measure(self, arg0: typing.Any, arg1: typing.Any, arg2: str, arg3: list[str]) -> CostData:
        ...
class CpuPassStrategy(PassStrategy):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: CpuPassStrategy) -> None:
        ...
    def enable_cudnn(self) -> None:
        ...
    def enable_mkldnn(self) -> None:
        ...
    def enable_mkldnn_bfloat16(self) -> None:
        ...
class CustomDeviceEvent:
    """
    
          The handle of the custom device event.
    
          Parameters:
              device(paddle.CustomPlace()|str): The device which wanted to allocate the stream.
              device_id(int, optional): The id of the device which wanted to allocate the stream.
                  If device is None or negative integer, device will be the current device.
                  If device is positive integer, it must less than the device count. Default: None.
              enable_timing(bool, optional): Whether the event will measure time. Default: False.
              blocking(bool, optional): Whether the wait() func will be blocking. Default: False.
              interprocess(bool, optional): Whether the event can be shared between processes. Default: False.
    
          Examples:
              .. code-block:: python
    
                  >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                  >>> import paddle
                  >>> place = paddle.CustomPlace('custom_cpu', 0)
                  >>> event = paddle.device.custom.Event(place)
    
          
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, device: typing.Any, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False) -> None:
        ...
    @typing.overload
    def __init__(self, device: str, device_id: int = -1, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False) -> None:
        ...
    def query(self) -> None:
        """
                  Queries the event's status.
        
                  Returns:
                      A boolean which indicates all work currently captured by the event has been completed.
        
                  Examples:
                      .. code-block:: python
        
                          >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                          >>> import paddle
                          >>> place = paddle.CustomPlace('custom_cpu', 0)
                          >>> event = paddle.device.cuda.Event(place)
                          >>> is_done = event.query()
        """
    def record(self, arg0: CustomDeviceStream) -> None:
        """
                  Records the event in the given stream.
        
                  Parameters:
                      stream(CustomDeviceStream, optional): The handle of custom device stream. If None, the stream is the current stream. Default: None.
        
                  Examples:
                      .. code-block:: python
        
                          >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                          >>> import paddle
                          >>> place = paddle.CustomPlace('custom_cpu', 0)
                          >>> event = paddle.device.custom.Event(place)
                          >>> event.record()
        """
    def synchronize(self) -> None:
        """
                    Waits for an event to complete.
        
                    Examples:
                        .. code-block:: python
        
                            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                            >>> import paddle
                            >>> place = paddle.CustomPlace('custom_cpu', 0)
                            >>> event = paddle.device.custom.Event(place)
                            >>> event.synchronize()
        """
    @property
    def place(self) -> None:
        ...
    @property
    def raw_event(self) -> None:
        """
                  return the raw event of type CustomDeviceEvent as type int.
        
                  Examples:
                      .. code-block:: python
        
                          >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                          >>> import paddle
                          >>> import ctypes
                          >>> place = paddle.CustomPlace('custom_cpu', 0)
                          >>> event = paddle.device.custom.Event(place)
                          >>> raw_event = event.raw_event
                          >>> print(raw_event)
        
                          >>> ptr = ctypes.c_void_p(raw_event)  # convert back to void*
                          >>> print(ptr)
        """
class CustomDeviceStream:
    """
    
          The handle of the custom device stream.
    
          Parameters:
              device(paddle.CustomPlace()|str): The device which wanted to allocate the stream.
              device_id(int, optional): The id of the device which wanted to allocate the stream.
                  If device is None or negative integer, device will be the current device.
                  If device is positive integer, it must less than the device count. Default: None.
              priority(int|None, optional): The priority of stream. The priority can be 1(high) or 2(normal).
                  If priority is None, the priority is 2(normal). Default: None.
              blocking(int|None, optional): Whether the stream is executed synchronously. Default: False.
    
          Examples:
              .. code-block:: python
    
                  >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                  >>> import paddle
                  >>> s3 = paddle.device.custom.Stream('custom_cpu')
                  >>> s2 = paddle.device.custom.Stream('custom_cpu', 0)
                  >>> s1 = paddle.device.custom.Stream(paddle.CustomPlace('custom_cpu'))
                  >>> s1 = paddle.device.custom.Stream(paddle.CustomPlace('custom_cpu'), 1)
                  >>> s1 = paddle.device.custom.Stream(paddle.CustomPlace('custom_cpu'), 1, True)
    
          
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, device: typing.Any, priority: int = 2, blocking: bool = False) -> None:
        ...
    @typing.overload
    def __init__(self, device: str, device_id: int = -1, priority: int = 2, blocking: bool = False) -> None:
        ...
    def query(self) -> None:
        """
                  Return the status whether if all operations in stream have completed.
        
                  Returns:
                      A boolean value.
        
                  Examples:
                      .. code-block:: python
        
                          >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                          >>> import paddle
                          >>> place = paddle.CustomPlace('custom_cpu', 0)
                          >>> s = paddle.device.custom.Stream(place)
                          >>> is_done = s.query()
        """
    def record_event(self, event: typing.Any = None) -> None:
        """
                  Record an event in the stream.
        
                  Parameters:
                      event(CustomDeviceEvent, optional): The event to be record. If event is None, a new event is created.
                          Default: None.
        
                  Returns:
                      The record event.
        
                  Examples:
                      .. code-block:: python
        
                          >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                          >>> import paddle
                          >>> place = paddle.CustomPlace('custom_cpu', 0)
                          >>> s = paddle.device.custom.Stream(place)
                          >>> event = s.record_event()
        """
    def synchronize(self) -> None:
        """
                  Waits for stream tasks to complete.
        
                  Examples:
                      .. code-block:: python
        
                          >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                          >>> import paddle
                          >>> place = paddle.CustomPlace('custom_cpu', 0)
                          >>> s = paddle.device.custom.Stream(place)
                          >>> s.synchronize()
        """
    def wait_event(self, arg0: typing.Any) -> None:
        """
                  Makes all future work submitted to stream wait for all work captured in event.
        
                  Parameters:
                      event(CustomDeviceEvent): The event to wait on.
        
                  Examples:
                      .. code-block:: python
        
                          >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                          >>> import paddle
                          >>> place = paddle.CustomPlace('custom_cpu', 0)
                          >>> s = paddle.device.custom.Stream(place)
                          >>> event = paddle.device.custom.Event(place)
                          >>> s.wait_event(event)
        """
    def wait_stream(self, arg0: CustomDeviceStream) -> None:
        """
                  Synchronizes with the given stream.
        
                  Parameters:
                      stream(CUDAStream): The stream to synchronize with.
        
                  Examples:
                      .. code-block:: python
        
                          >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                          >>> import paddle
                          >>> place = paddle.CustomPlace('custom_cpu', 0)
                          >>> s1 = paddle.device.custom.Stream(place)
                          >>> s2 = paddle.device.custom.Stream(place)
                          >>> s1.wait_stream(s2)
        """
    @property
    def place(self) -> None:
        ...
    @property
    def raw_stream(self) -> None:
        """
                  return the raw stream of type CustomDeviceStream as type int.
        
                  Examples:
                    .. code-block:: python
        
                        >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                        >>> import paddle
                        >>> import ctypes
                        >>> stream  = paddle.device.custom.current_stream().raw_stream
                        >>> print(stream)
        
                        >>> ptr = ctypes.c_void_p(stream)  # convert back to void*
                        >>> print(ptr)
        """
class CustomPlace:
    """
    
        CustomPlace is a descriptor of a device.
        It represents a custom device on which a tensor will be allocated and a model will run.
    
        Examples:
            .. code-block:: python
    
                >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
                >>> import paddle
                >>> fake_cpu_place = paddle.CustomPlace("FakeCPU", 0)
                                                    
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: str, arg1: int) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def _type(self) -> int:
        ...
    def get_device_id(self) -> int:
        ...
    def get_device_type(self) -> str:
        ...
class Dataset:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: str) -> None:
        ...
    def clear_sample_state(self) -> None:
        ...
    def create_channel(self) -> None:
        ...
    def create_preload_readers(self) -> None:
        ...
    def create_readers(self) -> None:
        ...
    def destroy_preload_readers(self) -> None:
        ...
    def destroy_readers(self) -> None:
        ...
    def dump_sample_neighbors(self, arg0: str) -> None:
        ...
    def dump_walk_path(self, arg0: str, arg1: int) -> None:
        ...
    def dynamic_adjust_channel_num(self, arg0: int, arg1: bool) -> None:
        ...
    def dynamic_adjust_readers_num(self, arg0: int) -> None:
        ...
    def enable_pv_merge(self) -> bool:
        ...
    def generate_local_tables_unlock(self, arg0: int, arg1: int, arg2: int, arg3: int, arg4: int) -> None:
        ...
    def get_data_feed_desc(self) -> typing.Any:
        ...
    def get_download_cmd(self) -> str:
        ...
    def get_epoch_finish(self) -> bool:
        ...
    def get_filelist(self) -> list[str]:
        ...
    def get_fleet_send_batch_size(self) -> int:
        ...
    def get_hdfs_config(self) -> tuple[str, str]:
        ...
    def get_memory_data_size(self) -> int:
        ...
    def get_pass_id(self) -> int:
        ...
    def get_pv_data_size(self) -> int:
        ...
    def get_shuffle_data_size(self) -> int:
        ...
    def get_thread_num(self) -> int:
        ...
    def get_trainer_num(self) -> int:
        ...
    def global_shuffle(self, arg0: int) -> None:
        ...
    def load_into_memory(self) -> None:
        ...
    def local_shuffle(self) -> None:
        ...
    def merge_by_lineid(self) -> None:
        ...
    def postprocess_instance(self) -> None:
        ...
    def preload_into_memory(self) -> None:
        ...
    def preprocess_instance(self) -> None:
        ...
    def register_client2client_msg_handler(self) -> None:
        ...
    def release_memory(self) -> None:
        ...
    def set_current_phase(self, arg0: int) -> None:
        ...
    def set_data_feed_desc(self, arg0: str) -> None:
        ...
    def set_download_cmd(self, arg0: str) -> None:
        ...
    def set_enable_pv_merge(self, arg0: bool) -> None:
        ...
    def set_fea_eval(self, arg0: bool, arg1: int) -> None:
        ...
    def set_filelist(self, arg0: list[str]) -> None:
        ...
    def set_fleet_send_batch_size(self, arg0: int) -> None:
        ...
    def set_fleet_send_sleep_seconds(self, arg0: int) -> None:
        ...
    def set_generate_unique_feasigns(self, arg0: bool) -> None:
        ...
    def set_gpu_graph_mode(self, arg0: int) -> None:
        ...
    def set_hdfs_config(self, arg0: str, arg1: str) -> None:
        ...
    def set_merge_by_lineid(self, arg0: int) -> None:
        ...
    def set_merge_by_sid(self, arg0: bool) -> None:
        ...
    def set_parse_content(self, arg0: bool) -> None:
        ...
    def set_parse_ins_id(self, arg0: bool) -> None:
        ...
    def set_parse_logkey(self, arg0: bool) -> None:
        ...
    def set_pass_id(self, arg0: int) -> None:
        ...
    def set_preload_thread_num(self, arg0: int) -> None:
        ...
    def set_queue_num(self, arg0: int) -> None:
        ...
    def set_shuffle_by_uid(self, arg0: bool) -> None:
        ...
    def set_thread_num(self, arg0: int) -> None:
        ...
    def set_trainer_num(self, arg0: int) -> None:
        ...
    def slots_shuffle(self, arg0: set[str]) -> None:
        ...
    def tdm_sample(self, arg0: str, arg1: str, arg2: list[int], arg3: int, arg4: bool, arg5: int, arg6: int) -> None:
        ...
    def wait_preload_done(self) -> None:
        ...
class DenseTensor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __array__(self, dtype: typing.Any = None, copy: typing.Any = None) -> numpy.ndarray:
        ...
    def __getitem__(self, arg0: typing.Any) -> DenseTensor:
        ...
    @typing.overload
    def __init__(self, arg0: list[list[int]]) -> None:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    def __str__(self) -> str:
        ...
    def _alloc_double(self, arg0: CPUPlace) -> None:
        ...
    @typing.overload
    def _alloc_float(self, arg0: CustomPlace) -> None:
        ...
    @typing.overload
    def _alloc_float(self, arg0: CUDAPlace) -> None:
        ...
    @typing.overload
    def _alloc_float(self, arg0: XPUPlace) -> None:
        ...
    @typing.overload
    def _alloc_float(self, arg0: CPUPlace) -> None:
        ...
    @typing.overload
    def _alloc_float(self, arg0: CUDAPinnedPlace) -> None:
        ...
    @typing.overload
    def _alloc_int(self, arg0: CPUPlace) -> None:
        ...
    @typing.overload
    def _alloc_int(self, arg0: CustomPlace) -> None:
        ...
    @typing.overload
    def _alloc_int(self, arg0: XPUPlace) -> None:
        ...
    @typing.overload
    def _alloc_int(self, arg0: CUDAPlace) -> None:
        ...
    @typing.overload
    def _alloc_int(self, arg0: CUDAPinnedPlace) -> None:
        ...
    def _as_type(self, arg0: VarDesc.VarType) -> DenseTensor:
        ...
    def _clear(self) -> None:
        ...
    def _copy(self, arg0: Place) -> DenseTensor:
        ...
    @typing.overload
    def _copy_from(self, tensor: DenseTensor, place: CPUPlace, batch_size: int = -1) -> None:
        ...
    @typing.overload
    def _copy_from(self, tensor: DenseTensor, place: CustomPlace, batch_size: int = -1) -> None:
        ...
    @typing.overload
    def _copy_from(self, tensor: DenseTensor, place: XPUPlace, batch_size: int = -1) -> None:
        ...
    @typing.overload
    def _copy_from(self, tensor: DenseTensor, place: CUDAPlace, batch_size: int = -1) -> None:
        ...
    @typing.overload
    def _copy_from(self, tensor: DenseTensor, place: CUDAPinnedPlace, batch_size: int = -1) -> None:
        ...
    @typing.overload
    def _copy_from(self, tensor: DenseTensor, place: IPUPlace, batch_size: int = -1) -> None:
        ...
    @typing.overload
    def _copy_from(self, tensor: DenseTensor, place: Place, batch_size: int = -1) -> None:
        ...
    def _dtype(self) -> VarDesc.VarType:
        ...
    def _get_complex128_element(self, arg0: int) -> complex:
        ...
    def _get_complex64_element(self, arg0: int) -> complex:
        ...
    def _get_dims(self) -> list[int]:
        ...
    def _get_double_element(self, arg0: int) -> float:
        ...
    def _get_float_element(self, arg0: int) -> float:
        ...
    def _is_initialized(self) -> bool:
        ...
    def _layout(self) -> str:
        ...
    @typing.overload
    def _mutable_data(self, arg0: CPUPlace, arg1: VarDesc.VarType) -> int:
        ...
    @typing.overload
    def _mutable_data(self, arg0: CustomPlace, arg1: VarDesc.VarType) -> int:
        ...
    @typing.overload
    def _mutable_data(self, arg0: XPUPlace, arg1: VarDesc.VarType) -> int:
        ...
    @typing.overload
    def _mutable_data(self, arg0: CUDAPlace, arg1: VarDesc.VarType) -> int:
        ...
    @typing.overload
    def _mutable_data(self, arg0: CUDAPinnedPlace, arg1: VarDesc.VarType) -> int:
        ...
    def _numel(self) -> int:
        ...
    def _place(self) -> Place:
        ...
    def _ptr(self) -> int:
        ...
    def _set_complex128_element(self, arg0: int, arg1: complex) -> None:
        ...
    def _set_complex64_element(self, arg0: int, arg1: complex) -> None:
        ...
    def _set_dims(self, arg0: list[int]) -> None:
        ...
    def _set_double_element(self, arg0: int, arg1: float) -> None:
        ...
    def _set_float_element(self, arg0: int, arg1: float) -> None:
        ...
    def _set_layout(self, arg0: str) -> None:
        ...
    def _share_data_nocheck_with(self, arg0: DenseTensor) -> DenseTensor:
        ...
    def _share_data_with(self, arg0: DenseTensor) -> DenseTensor:
        ...
    def _slice(self, arg0: int, arg1: int) -> DenseTensor:
        ...
    def _to_dlpack(self) -> typing_extensions.CapsuleType:
        ...
    def lod(self) -> list[list[int]]:
        """
                   Return the LegacyLoD of the Tensor.
        
                   Returns:
                       list[list[int]]: The lod of the Tensor.
        
                   Examples:
                        .. code-block:: python
        
                            >>> import paddle
                            >>> import numpy as np
        
                            >>> t = paddle.framework.core.Tensor()
                            >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
                            >>> t.set_lod([[0, 2, 5]])
                            >>> print(t.lod())
                            [[0, 2, 5]]
        """
    @typing.overload
    def set(self, array: typing.Any, place: CPUPlace, zero_copy: bool = False) -> None:
        ...
    @typing.overload
    def set(self, array: typing.Any, place: CustomPlace, zero_copy: bool = False) -> None:
        ...
    @typing.overload
    def set(self, array: typing.Any, place: XPUPlace, zero_copy: bool = False) -> None:
        ...
    @typing.overload
    def set(self, array: typing.Any, place: CUDAPlace, zero_copy: bool = False) -> None:
        ...
    @typing.overload
    def set(self, array: typing.Any, place: IPUPlace, zero_copy: bool = False) -> None:
        ...
    @typing.overload
    def set(self, array: typing.Any, place: CUDAPinnedPlace, zero_copy: bool = False) -> None:
        """
                Set the data of Tensor on place with given numpy array.
        
                Args:
                  array (numpy.ndarray): The shape where the DenseTensor is to be set.
                  place (CPUPlace|CUDAPlace|XPUPlace|IPUPlace|CUDAPinnedPlace): The place where the
                  Tensor is to be set.
                  zero_copy (bool, optional): Whether to share memory with the input numpy array.
                  This parameter only works with CPUPlace. Default: False.
        
                Returns:
                    None.
        
                Examples:
                    .. code-block:: python
        
                        >>> import paddle
                        >>> import numpy as np
        
                        >>> t = paddle.framework.core.Tensor()
                        >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
        """
    def set_lod(self, lod: list[list[int]]) -> None:
        """
                   Set LegacyLoD of the Tensor.
        
                   Args:
                       lod (list[list[int]]): The lod to set.
        
                   Returns:
                        None.
        
                   Examples:
                        .. code-block:: python
        
                            >>> import paddle
                            >>> import numpy as np
        
                            >>> t = paddle.framework.core.Tensor()
                            >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
                            >>> t.set_lod([[0, 2, 5]])
                            >>> print(t.lod())
                            [[0, 2, 5]]
        """
    def set_recursive_sequence_lengths(self, recursive_sequence_lengths: list[list[int]]) -> None:
        """
                   Set LegacyLoD of the Tensor according to recursive sequence lengths.
        
                   For example, if recursive_sequence_lengths=[[2, 3]], which means
                   there are two sequences with length 2 and 3 respectively, the
                   corresponding lod would be [[0, 2, 2+3]], i.e., [[0, 2, 5]].
        
                   Args:
                        recursive_sequence_lengths (list[list[int]]): The recursive sequence lengths.
        
                   Returns:
                        None.
        
                   Examples:
                        .. code-block:: python
        
                            >>> import paddle
                            >>> import numpy as np
        
                            >>> t = paddle.framework.core.Tensor()
                            >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
                            >>> t.set_recursive_sequence_lengths([[2, 3]])
                            >>> print(t.recursive_sequence_lengths())
                            [[2, 3]]
                            >>> print(t.lod())
                            [[0, 2, 5]]
        """
    def shape(self) -> list[int]:
        """
                   Return the shape of Tensor.
        
                   Returns:
                       list[int]: The shape of Tensor.
        
        
                   Examples:
                        .. code-block:: python
        
                            >>> import paddle
                            >>> import numpy as np
        
                            >>> t = paddle.framework.core.Tensor()
                            >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
                            >>> print(t.shape())
                            [5, 30]
        """
class DenseTensorArray:
    """
    
        DenseTensorArray is array of DenseTensor, it supports operator[], len() and for-loop iteration.
    
        Examples:
            .. code-block:: python
    
                >>> import paddle
                >>> arr = paddle.framework.core.DenseTensorArray()
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __getitem__(self, arg0: int) -> typing.Any:
        ...
    def __init__(self) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __setitem__(self, arg0: int, arg1: typing.Any) -> None:
        ...
    def _move_to_list(self) -> list:
        ...
    def append(self, tensor: typing.Any) -> None:
        """
                     Append a DenseTensor to DenseTensorArray.
        
                     Args:
                           tensor (DenseTensor): The DenseTensor to be appended.
        
                     Returns:
                           None.
        
                     Examples:
                            .. code-block:: python
        
                                >>> import paddle
                                >>> import numpy as np
        
                                >>> arr = paddle.framework.core.DenseTensorArray()
                                >>> t = paddle.framework.core.DenseTensor()
                                >>> t.set(np.ndarray([5, 30]), paddle.CPUPlace())
                                >>> arr.append(t)
        """
class DenseTensorBlockingQueue:
    """
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def capacity(self) -> int:
        ...
    def close(self) -> None:
        ...
    def kill(self) -> None:
        ...
    def push(self, arg0: typing.Any) -> bool:
        ...
    def size(self) -> int:
        ...
    def wait_for_inited(self, arg0: int) -> bool:
        ...
class Device:
    __hash__: typing.ClassVar[None] = None
    capability: DeviceCapability
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, arg0: Device) -> bool:
        ...
    def __init__(self, global_id: int, local_id: int, machine_id: int, type: str) -> None:
        ...
    def __ne__(self, arg0: Device) -> bool:
        ...
    def __str__(self) -> str:
        ...
    @property
    def global_id(self) -> int:
        ...
    @property
    def local_id(self) -> int:
        ...
    @property
    def machine_id(self) -> int:
        ...
    @property
    def type(self) -> str:
        ...
class DeviceCapability:
    dflops: float
    memory: float
    rate: float
    sflops: float
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __str__(self) -> str:
        ...
class DeviceContext:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @staticmethod
    @typing.overload
    def create(arg0: typing.Any) -> DeviceContext:
        ...
    @staticmethod
    @typing.overload
    def create(arg0: typing.Any) -> DeviceContext:
        ...
    @staticmethod
    @typing.overload
    def create(arg0: typing.Any) -> DeviceContext:
        ...
    @staticmethod
    @typing.overload
    def create(arg0: typing.Any) -> DeviceContext:
        ...
    @staticmethod
    @typing.overload
    def create(arg0: typing.Any) -> DeviceContext:
        ...
class DeviceMesh:
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __copy__(self: typing.Any) -> typing.Any:
        ...
    def __deepcopy__(self: typing.Any, memo: dict) -> typing.Any:
        ...
    def __eq__(self, arg0: DeviceMesh) -> bool:
        ...
    def __init__(self, name: str, shape: list[int], device_ids: list[int], dim_names: list[str]) -> None:
        ...
    def __ne__(self, arg0: DeviceMesh) -> bool:
        ...
    def __str__(self) -> str:
        ...
    def add_device(self, arg0: Device) -> None:
        ...
    def add_link(self, arg0: Link) -> None:
        ...
    def contains(self, arg0: int) -> bool:
        ...
    def device(self, arg0: int) -> Device:
        ...
    @typing.overload
    def dim_size(self, arg0: int) -> int:
        ...
    @typing.overload
    def dim_size(self, arg0: str) -> int:
        ...
    def empty(self) -> bool:
        ...
    def link(self, arg0: int, arg1: int) -> Link:
        ...
    def machine(self, arg0: int) -> Machine:
        ...
    @property
    def device_ids(self) -> list[int]:
        ...
    @property
    def device_type(self) -> str:
        ...
    @property
    def devices(self) -> dict[int, Device]:
        ...
    @property
    def dim_names(self) -> list[str]:
        ...
    @property
    def links(self) -> dict[int, dict[int, Link]]:
        ...
    @property
    def machines(self) -> dict[int, Machine]:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def ndim(self) -> int:
        ...
    @property
    def shape(self) -> list[int]:
        ...
    @property
    def size(self) -> int:
        ...
class DevicePythonNode:
    block_x: int
    block_y: int
    block_z: int
    blocks_per_sm: float
    context_id: int
    correlation_id: int
    device_id: int
    end_ns: int
    grid_x: int
    grid_y: int
    grid_z: int
    name: str
    num_bytes: int
    occupancy: float
    registers_per_thread: int
    shared_memory: int
    start_ns: int
    stream_id: int
    type: typing.Any
    value: int
    warps_per_sm: float
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class DeviceType:
    """
    Members:
    
      CPU
    
      CUDA
    
      XPU
    """
    CPU: typing.ClassVar[DeviceType]  # value = <DeviceType.CPU: 0>
    CUDA: typing.ClassVar[DeviceType]  # value = <DeviceType.CUDA: 1>
    XPU: typing.ClassVar[DeviceType]  # value = <DeviceType.XPU: 3>
    __members__: typing.ClassVar[dict[str, DeviceType]]  # value = {'CPU': <DeviceType.CPU: 0>, 'CUDA': <DeviceType.CUDA: 1>, 'XPU': <DeviceType.XPU: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __and__(self, other: typing.Any) -> typing.Any:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> typing.Any:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, other: typing.Any) -> typing.Any:
        ...
    def __rand__(self, other: typing.Any) -> typing.Any:
        ...
    def __repr__(self) -> str:
        ...
    def __ror__(self, other: typing.Any) -> typing.Any:
        ...
    def __rxor__(self, other: typing.Any) -> typing.Any:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, other: typing.Any) -> typing.Any:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DistTensorSpec:
    shape: list[int]
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __copy__(self) -> DistTensorSpec:
        ...
    def __deepcopy__(self, memo: dict) -> DistTensorSpec:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: DistTensorSpec) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: list[int], arg1: TensorDistAttr) -> None:
        ...
    def __str__(self) -> str:
        ...
    def dims_mapping(self) -> list[int]:
        ...
    def process_mesh(self) -> ProcessMesh:
        ...
    def set_dims_mapping(self, arg0: list[int]) -> None:
        ...
    def set_process_mesh(self, arg0: ProcessMesh) -> None:
        ...
class DtypeMatchGuard(GuardBase):
    """
    DtypeMatchGuard Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, dtype: typing.Any) -> None:
        ...
    @typing.overload
    def __init__(self, dtype: typing.Any) -> None:
        ...
class EOFException(Exception):
    pass
class EagerReducer:
    """
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: typing.Any, arg1: list[list[int]], arg2: list[bool], arg3: ProcessGroup, arg4: list[int], arg5: bool) -> None:
        ...
    def prepare_for_backward(self, tensors: typing.Any) -> None:
        ...
class EnforceNotMet(Exception):
    pass
class EventSortingKey:
    """
    Members:
    
      kDefault
    
      kCalls
    
      kTotal
    
      kMin
    
      kMax
    
      kAve
    """
    __members__: typing.ClassVar[dict[str, EventSortingKey]]  # value = {'kDefault': <EventSortingKey.kDefault: 0>, 'kCalls': <EventSortingKey.kCalls: 1>, 'kTotal': <EventSortingKey.kTotal: 2>, 'kMin': <EventSortingKey.kMin: 3>, 'kMax': <EventSortingKey.kMax: 4>, 'kAve': <EventSortingKey.kAve: 5>}
    kAve: typing.ClassVar[EventSortingKey]  # value = <EventSortingKey.kAve: 5>
    kCalls: typing.ClassVar[EventSortingKey]  # value = <EventSortingKey.kCalls: 1>
    kDefault: typing.ClassVar[EventSortingKey]  # value = <EventSortingKey.kDefault: 0>
    kMax: typing.ClassVar[EventSortingKey]  # value = <EventSortingKey.kMax: 4>
    kMin: typing.ClassVar[EventSortingKey]  # value = <EventSortingKey.kMin: 3>
    kTotal: typing.ClassVar[EventSortingKey]  # value = <EventSortingKey.kTotal: 2>
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Executor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: typing.Any) -> None:
        ...
    def close(self) -> None:
        ...
    def create_variables(self, arg0: typing.Any, arg1: _Scope, arg2: int) -> None:
        ...
    def get_place(self) -> typing.Any:
        ...
    def init_for_dataset(self, arg0: typing.Any, arg1: str, arg2: _Scope, arg3: typing.Any) -> TrainerBase:
        ...
    def prepare(self, arg0: typing.Any, arg1: int, arg2: list[str], arg3: bool) -> ExecutorPrepareContext:
        ...
    def release_trainer(self, arg0: TrainerBase) -> None:
        ...
    def run(self, arg0: typing.Any, arg1: _Scope, arg2: int, arg3: bool, arg4: bool, arg5: list[str]) -> None:
        ...
    @typing.overload
    def run_from_dataset(self, arg0: TrainerBase) -> None:
        ...
    @typing.overload
    def run_from_dataset(self, arg0: TrainerBase) -> None:
        ...
    @typing.overload
    def run_prepared_ctx(self, arg0: ExecutorPrepareContext, arg1: _Scope, arg2: dict[str, typing.Any], arg3: dict[str, typing.Any, typing.Any, typing.Any, typing.Any], arg4: bool, arg5: bool, arg6: str, arg7: str) -> None:
        ...
    @typing.overload
    def run_prepared_ctx(self, arg0: ExecutorPrepareContext, arg1: _Scope, arg2: bool, arg3: bool, arg4: bool) -> None:
        ...
class ExecutorPrepareContext:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: typing.Any, arg1: int) -> None:
        ...
class FetchList:
    """
     FetchList is a
            vector of paddle::variant<DenseTensor, DenseTensorArray>.
            
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def _move_to_list(self) -> list:
        ...
    @typing.overload
    def append(self, var: typing.Any) -> None:
        ...
    @typing.overload
    def append(self, var: DenseTensorArray) -> None:
        ...
class FetchUnmergedList:
    """
    
            FetchUnmergedList is 2-D array of FetchType(paddle::variant(DenseTensor, DenseTensorArray)).
            
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def _move_to_list(self) -> list:
        ...
class Fleet:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def cache_shuffle(self, arg0: int, arg1: str, arg2: int, arg3: float) -> None:
        ...
    def clear_model(self) -> None:
        ...
    def clear_one_table(self, arg0: int) -> None:
        ...
    def client_flush(self) -> None:
        ...
    def confirm(self) -> None:
        ...
    def copy_table(self, arg0: int, arg1: int) -> int:
        ...
    def copy_table_by_feasign(self, arg0: int, arg1: int, arg2: list[int]) -> int:
        ...
    def create_client2client_connection(self) -> None:
        ...
    def finalize_worker(self) -> None:
        ...
    def gather_clients(self, arg0: list[int]) -> None:
        ...
    def gather_servers(self, arg0: list[int], arg1: int) -> None:
        ...
    def get_cache_threshold(self, arg0: int) -> float:
        ...
    def get_clients_info(self) -> list[int]:
        ...
    def init_model(self, arg0: _Scope, arg1: int, arg2: list[str]) -> None:
        ...
    def init_server(self, arg0: str, arg1: int) -> None:
        ...
    def init_worker(self, arg0: str, arg1: list[int], arg2: int, arg3: int) -> None:
        ...
    def load_from_paddle_model(self, arg0: _Scope, arg1: int, arg2: list[str], arg3: str, arg4: str, arg5: list[str], arg6: bool) -> None:
        ...
    def load_model(self, arg0: str, arg1: int) -> None:
        ...
    def load_model_one_table(self, arg0: int, arg1: str, arg2: int) -> None:
        ...
    def load_table_with_whitelist(self, arg0: int, arg1: str, arg2: int) -> None:
        ...
    def print_table_stat(self, arg0: int, arg1: int, arg2: int) -> None:
        ...
    def pull_dense(self, arg0: _Scope, arg1: int, arg2: list[str]) -> None:
        ...
    def push_dense(self, arg0: _Scope, arg1: int, arg2: list[str]) -> None:
        ...
    def revert(self) -> None:
        ...
    @typing.overload
    def run_server(self) -> int:
        ...
    @typing.overload
    def run_server(self, arg0: str, arg1: int) -> int:
        ...
    def save_cache(self, arg0: int, arg1: str, arg2: int) -> int:
        ...
    def save_model(self, arg0: str, arg1: int) -> None:
        ...
    def save_model_one_table(self, arg0: int, arg1: str, arg2: int) -> None:
        ...
    def save_model_one_table_with_prefix(self, arg0: int, arg1: str, arg2: int, arg3: str) -> None:
        ...
    def save_model_with_whitelist(self, arg0: int, arg1: str, arg2: int, arg3: str) -> int:
        ...
    def save_multi_table_one_path(self, arg0: list[int], arg1: str, arg2: int) -> None:
        ...
    def set_client2client_config(self, arg0: int, arg1: int, arg2: int) -> None:
        ...
    def set_date(self, arg0: int, arg1: str) -> None:
        ...
    def set_file_num_one_shard(self, arg0: int, arg1: int) -> None:
        ...
    def set_pull_local_thread_num(self, arg0: int) -> None:
        ...
    def shrink_dense_table(self, arg0: int, arg1: _Scope, arg2: list[str], arg3: float, arg4: int) -> None:
        ...
    def shrink_sparse_table(self, arg0: int) -> None:
        ...
    def stop_server(self) -> None:
        ...
class Function:
    """
    Function Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
class FunctionInfo:
    """
    BaseFunctionInfo Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def input_names(self) -> list[str]:
        ...
    def name(self) -> str:
        ...
    def output_names(self) -> list[str]:
        ...
class GatherOptions:
    root_rank: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class Generator:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def get_state(self) -> GeneratorState:
        ...
    def get_state_index(self) -> int:
        ...
    def initial_seed(self) -> int:
        ...
    def manual_seed(self, arg0: int) -> Generator:
        ...
    def random(self) -> int:
        ...
    def register_state_index(self, arg0: GeneratorState) -> int:
        ...
    def seed(self) -> int:
        ...
    def set_state(self, arg0: GeneratorState) -> None:
        ...
    def set_state_index(self, arg0: int) -> None:
        ...
class GeneratorState:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __str__(self) -> str:
        ...
    def current_seed(self) -> int:
        ...
class GlobalVarGetterSetterRegistry:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __contains__(self, arg0: str) -> bool:
        ...
    def __getitem__(self, arg0: str) -> typing.Any:
        ...
    def __setitem__(self, arg0: str, arg1: typing.Any) -> None:
        ...
    def get(self, key: str, default: typing.Any = None) -> typing.Any:
        ...
    def get_default(self, key: str) -> typing.Any:
        ...
    def is_public(self, arg0: str) -> bool:
        ...
    def keys(self) -> set[str]:
        ...
class Gloo:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    @typing.overload
    def all_gather(self, arg0: int) -> list[int]:
        ...
    @typing.overload
    def all_gather(self, arg0: int) -> list[int]:
        ...
    @typing.overload
    def all_gather(self, arg0: float) -> list[float]:
        ...
    @typing.overload
    def all_gather(self, arg0: float) -> list[float]:
        ...
    @typing.overload
    def all_reduce(self, arg0: list[int], arg1: str) -> list[int]:
        ...
    @typing.overload
    def all_reduce(self, arg0: list[int], arg1: str) -> list[int]:
        ...
    @typing.overload
    def all_reduce(self, arg0: list[float], arg1: str) -> list[float]:
        ...
    @typing.overload
    def all_reduce(self, arg0: list[float], arg1: str) -> list[float]:
        ...
    def barrier(self) -> None:
        ...
    def init(self) -> None:
        ...
    def rank(self) -> int:
        ...
    def set_hdfs_store(self, arg0: str, arg1: str, arg2: str) -> None:
        ...
    def set_http_store(self, arg0: str, arg1: int, arg2: str) -> None:
        ...
    def set_iface(self, arg0: str) -> None:
        ...
    def set_prefix(self, arg0: str) -> None:
        ...
    def set_rank(self, arg0: int) -> None:
        ...
    def set_size(self, arg0: int) -> None:
        ...
    def set_timeout_seconds(self, arg0: int, arg1: int) -> None:
        ...
    def size(self) -> int:
        ...
class GpuPassStrategy(PassStrategy):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: GpuPassStrategy) -> None:
        ...
    def enable_cudnn(self) -> None:
        ...
    def enable_mkldnn(self) -> None:
        ...
    def enable_mkldnn_bfloat16(self) -> None:
        ...
class GradNodeBase:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def _register_post_hook(self, arg0: typing.Any) -> NodePostHookRemoveHelper:
        ...
    def input_meta(self) -> typing.Any:
        ...
    def name(self) -> str:
        ...
    def node_ptr(self) -> int:
        ...
    def output_meta(self) -> typing.Any:
        ...
    @property
    def next_functions(self) -> list[GradNodeBase]:
        ...
class Graph:
    """
    The graph is a Directed Acyclic Single Static Assignment Graph, see `paddle::ir::Graph` for details.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, arg0: ProgramDesc) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: ProgramDesc, arg1: int, arg2: int) -> None:
        ...
    def clone(self) -> Graph:
        ...
    def create_control_dep_var(self) -> typing.Any:
        ...
    def create_empty_node(self, arg0: str, arg1: typing.Any) -> typing.Any:
        ...
    def create_op_node(self, arg0: OpDesc) -> typing.Any:
        ...
    def create_var_node(self, arg0: VarDesc) -> typing.Any:
        ...
    def erase(self, arg0: str) -> None:
        ...
    def get_bool(self, arg0: str) -> bool:
        ...
    def get_double(self, arg0: str) -> float:
        ...
    def get_float(self, arg0: str) -> float:
        ...
    def get_int(self, arg0: str) -> int:
        ...
    def get_marked_nodes(self, arg0: str) -> set[typing.Any]:
        ...
    def get_string(self, arg0: str) -> str:
        ...
    def get_sub_graph(self, arg0: int) -> Graph:
        ...
    def has(self, arg0: str) -> bool:
        ...
    def nodes(self) -> set[typing.Any]:
        ...
    def origin_program_desc(self) -> ProgramDesc:
        ...
    def release_nodes(self) -> list[typing.Any]:
        ...
    def remove_node(self, arg0: typing.Any) -> typing.Any:
        ...
    def resolve_hazard(self, arg0: dict[str, list[typing.Any]]) -> None:
        ...
    def retrieve_node(self, arg0: int) -> typing.Any:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: bool) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: int) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: str) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: float) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: float) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: set[typing.Any]) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: set[str]) -> None:
        ...
    def set_not_owned(self, arg0: str, arg1: _Scope) -> None:
        ...
    def sub_graph_size(self) -> int:
        ...
class GuardBase:
    """
    GuardBase Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def check(self, arg0: typing.Any) -> bool:
        ...
class GuardGroup(GuardBase):
    """
    GuardGroup Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, guards: list[GuardBase]) -> None:
        ...
class HostPythonNode:
    attributes: dict[str, typing.Any]
    callstack: str
    children_node: list[HostPythonNode]
    correlation_id: int
    device_node: list[DevicePythonNode]
    dtypes: dict[str, list[str]]
    end_ns: int
    input_shapes: dict[str, list[list[int]]]
    mem_node: list[MemPythonNode]
    name: str
    op_id: int
    process_id: int
    runtime_node: list[HostPythonNode]
    start_ns: int
    thread_id: int
    type: typing.Any
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class IPUPlace:
    """
    
        IPUPlace is a descriptor of a device.
        It represents a IPU device on which a tensor will be allocated and a model will run.
    
        Examples:
            .. code-block:: python
    
                >>> # doctest: +REQUIRES(env:IPU)
                >>> import paddle
                >>> ipu_place = paddle.IPUPlace()
    
            
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __str__(self) -> str:
        ...
    @typing.overload
    def _equals(self, arg0: typing.Any) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CUDAPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CPUPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: XPUPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: IPUPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CUDAPinnedPlace) -> bool:
        ...
    def _type(self) -> int:
        ...
class IdMatchGuard(GuardBase):
    """
    IdMatchGuard Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, py_obj: typing.Any) -> None:
        ...
class InstanceCheckGuard(GuardBase):
    """
    InstanceCheckGuard Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, isinstance_obj: typing.Any) -> None:
        ...
class InternalUtils:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @staticmethod
    def disable_tensorrt_half_ops(arg0: AnalysisConfig, arg1: set[str]) -> None:
        ...
    @staticmethod
    def set_transformer_maskid(arg0: AnalysisConfig, arg1: str) -> None:
        ...
    @staticmethod
    def set_transformer_posid(arg0: AnalysisConfig, arg1: str) -> None:
        ...
class IterableDatasetWrapper:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: Dataset, arg1: list[str], arg2: list[Place], arg3: int, arg4: bool) -> None:
        ...
    def _next(self) -> list[dict[str, DenseTensor]]:
        ...
    def _start(self) -> None:
        ...
class Job:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, type: str) -> None:
        ...
    def micro_batch_id(self) -> int:
        ...
    def set_micro_batch_id(self, arg0: int) -> None:
        ...
    def set_skip_gc_vars(self, arg0: set[str]) -> None:
        ...
    def type(self) -> str:
        ...
class LambdaGuard(GuardBase):
    """
    LambdaGuard Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, guard_check_fn: typing.Callable) -> None:
        ...
class Layer:
    """
    Layer Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def function(self, arg0: str) -> typing.Any:
        ...
    def function_info(self, arg0: str) -> typing.Any:
        ...
    def function_names(self) -> list[str]:
        ...
class LayerMatchGuard(GuardBase):
    """
    LayerMatchGuard Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, layer_obj: typing.Any) -> None:
        ...
class LengthMatchGuard(GuardBase):
    """
    LengthMatchGuard Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, length: int) -> None:
        ...
class Link:
    __hash__: typing.ClassVar[None] = None
    capability: LinkCapability
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, arg0: Link) -> bool:
        ...
    def __init__(self, source_id: int, target_id: int, type: str) -> None:
        ...
    def __ne__(self, arg0: Link) -> bool:
        ...
    def __str__(self) -> str:
        ...
    @property
    def source_id(self) -> int:
        ...
    @property
    def target_id(self) -> int:
        ...
    @property
    def type(self) -> str:
        ...
class LinkCapability:
    bandwidth: int
    latency: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __str__(self) -> str:
        ...
class Machine:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __str__(self) -> str:
        ...
    def contains(self, arg0: int) -> bool:
        ...
    def device(self, arg0: int) -> Device:
        ...
    def link(self, arg0: int, arg1: int) -> Link:
        ...
    @property
    def devices(self) -> dict[int, Device]:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def links(self) -> dict[int, dict[int, Link]]:
        ...
class MemPythonNode:
    addr: int
    current_allocated: int
    current_reserved: int
    increase_bytes: int
    peak_allocated: int
    peak_reserved: int
    place: str
    process_id: int
    thread_id: int
    timestamp_ns: int
    type: typing.Any
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class MultiDeviceFeedReader:
    """
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def read_next(self) -> list[dict[str, typing.Any]]:
        ...
    def read_next_list(self) -> list[typing.Any]:
        ...
    def read_next_var_list(self) -> list[typing.Any]:
        ...
    def reset(self) -> None:
        ...
    def shutdown(self) -> None:
        ...
class NativeConfig(PaddlePredictor.Config):
    device: int
    fraction_of_gpu_memory: float
    param_file: str
    prog_file: str
    specify_input_name: bool
    use_gpu: bool
    use_xpu: bool
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def cpu_math_library_num_threads(self) -> int:
        ...
    def set_cpu_math_library_num_threads(self, arg0: int) -> None:
        ...
class NativePaddlePredictor(PaddlePredictor):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: NativeConfig) -> None:
        ...
    def clone(self) -> PaddlePredictor:
        ...
    def get_input_tensor(self, arg0: str) -> typing.Any:
        ...
    def get_output_tensor(self, arg0: str) -> typing.Any:
        ...
    def init(self, arg0: _Scope) -> bool:
        ...
    def run(self, arg0: list[PaddleTensor]) -> list[PaddleTensor]:
        ...
    def scope(self) -> _Scope:
        ...
    def zero_copy_run(self, switch_stream: bool = False) -> bool:
        ...
class Node:
    class Dep:
        """
        Members:
        
          Same
        
          Before
        
          After
        
          NoDep
        """
        After: typing.ClassVar[Node.Dep]  # value = <Dep.After: 2>
        Before: typing.ClassVar[Node.Dep]  # value = <Dep.Before: 1>
        NoDep: typing.ClassVar[Node.Dep]  # value = <Dep.NoDep: 3>
        Same: typing.ClassVar[Node.Dep]  # value = <Dep.Same: 0>
        __members__: typing.ClassVar[dict[str, Node.Dep]]  # value = {'Same': <Dep.Same: 0>, 'Before': <Dep.Before: 1>, 'After': <Dep.After: 2>, 'NoDep': <Dep.NoDep: 3>}
        @staticmethod
        def _pybind11_conduit_v1_(*args, **kwargs):
            ...
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: int) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    class Type:
        """
        Members:
        
          Operation
        
          Variable
        """
        Operation: typing.ClassVar[Node.Type]  # value = <Type.Operation: 0>
        Variable: typing.ClassVar[Node.Type]  # value = <Type.Variable: 1>
        __members__: typing.ClassVar[dict[str, Node.Type]]  # value = {'Operation': <Type.Operation: 0>, 'Variable': <Type.Variable: 1>}
        @staticmethod
        def _pybind11_conduit_v1_(*args, **kwargs):
            ...
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: int) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    After: typing.ClassVar[Node.Dep]  # value = <Dep.After: 2>
    Before: typing.ClassVar[Node.Dep]  # value = <Dep.Before: 1>
    NoDep: typing.ClassVar[Node.Dep]  # value = <Dep.NoDep: 3>
    Operation: typing.ClassVar[Node.Type]  # value = <Type.Operation: 0>
    Same: typing.ClassVar[Node.Dep]  # value = <Dep.Same: 0>
    Variable: typing.ClassVar[Node.Type]  # value = <Type.Variable: 1>
    inputs: list[Node]
    outputs: list[Node]
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def append_input(self, arg0: Node) -> None:
        ...
    def append_output(self, arg0: Node) -> None:
        ...
    def clear_inputs(self) -> None:
        ...
    def clear_outputs(self) -> None:
        ...
    def graph_id(self) -> int:
        ...
    def id(self) -> int:
        ...
    def is_ctrl_var(self) -> bool:
        ...
    def is_op(self) -> bool:
        ...
    def is_var(self) -> bool:
        ...
    def name(self) -> str:
        ...
    def node_type(self) -> typing.Any:
        ...
    def op(self) -> OpDesc:
        ...
    def original_desc_id(self) -> int:
        ...
    @typing.overload
    def remove_input(self, arg0: int) -> None:
        ...
    @typing.overload
    def remove_input(self, arg0: Node) -> None:
        ...
    @typing.overload
    def remove_output(self, arg0: int) -> None:
        ...
    @typing.overload
    def remove_output(self, arg0: Node) -> None:
        ...
    def var(self) -> VarDesc:
        ...
class NodePostHookRemoveHelper:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def remove(self) -> bool:
        ...
class OpAttrInfo(OpUpdateInfo):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: str, arg2: typing.Any) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: OpAttrInfo) -> None:
        ...
    def default_value(self) -> typing.Any:
        ...
    def name(self) -> str:
        ...
    def remark(self) -> str:
        ...
class OpBugfixInfo(OpUpdateInfo):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: OpBugfixInfo) -> None:
        ...
    def remark(self) -> str:
        ...
class OpCheckpoint:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def note(self) -> str:
        ...
    def version_desc(self) -> OpVersionDesc:
        ...
class OpDesc:
    """
    """
    dist_attr: typing.Any
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def _block_attr_id(self, arg0: str) -> int:
        ...
    def _blocks_attr_ids(self, arg0: str) -> list[int]:
        ...
    def _rename_input(self, arg0: str, arg1: str) -> None:
        ...
    def _rename_output(self, arg0: str, arg1: str) -> None:
        ...
    def _set_attr(self, arg0: str, arg1: typing.Any) -> None:
        ...
    def _set_bool_attr(self, arg0: str, arg1: bool) -> None:
        ...
    def _set_bools_attr(self, arg0: str, arg1: list[bool]) -> None:
        ...
    def _set_float32_attr(self, arg0: str, arg1: float) -> None:
        ...
    def _set_float32s_attr(self, arg0: str, arg1: list[float]) -> None:
        ...
    def _set_float64_attr(self, arg0: str, arg1: float) -> None:
        ...
    def _set_float64s_attr(self, arg0: str, arg1: list[float]) -> None:
        ...
    def _set_int32_attr(self, arg0: str, arg1: int) -> None:
        ...
    def _set_int32s_attr(self, arg0: str, arg1: list[int]) -> None:
        ...
    def _set_int64_attr(self, arg0: str, arg1: int) -> None:
        ...
    def _set_int64s_attr(self, arg0: str, arg1: list[int]) -> None:
        ...
    def _set_scalar_attr(self, arg0: str, arg1: typing.Any) -> None:
        ...
    def _set_scalars_attr(self, arg0: str, arg1: list[typing.Any]) -> None:
        ...
    def _set_str_attr(self, arg0: str, arg1: str) -> None:
        ...
    def _set_strs_attr(self, arg0: str, arg1: list[str]) -> None:
        ...
    def attr(self, name: str, with_attr_var: bool = False) -> typing.Any:
        ...
    def attr_names(self, with_attr_var: bool = False) -> list[str]:
        ...
    def attr_type(self, name: str, with_attr_var: bool = False) -> AttrType:
        ...
    def block(self) -> BlockDesc:
        ...
    def check_attrs(self) -> None:
        ...
    def copy_from(self, arg0: OpDesc) -> None:
        ...
    def get_attr_map(self) -> dict[str, typing.Any]:
        ...
    def has_attr(self, name: str, with_attr_var: bool = False) -> bool:
        ...
    def id(self) -> int:
        ...
    def infer_shape(self, arg0: BlockDesc) -> None:
        ...
    def infer_var_type(self, arg0: BlockDesc) -> None:
        ...
    def input(self, arg0: str) -> list[str]:
        ...
    def input_arg_names(self, with_attr_var: bool = False) -> list[str]:
        ...
    def input_names(self, with_attr_var: bool = False) -> list[str]:
        ...
    def inputs(self) -> dict[str, list[str]]:
        ...
    def original_id(self) -> int:
        ...
    def output(self, arg0: str) -> list[str]:
        ...
    def output_arg_names(self) -> list[str]:
        ...
    def output_names(self) -> list[str]:
        ...
    def outputs(self) -> dict[str, list[str]]:
        ...
    def remove_attr(self, arg0: str) -> None:
        ...
    def remove_input(self, arg0: str) -> None:
        ...
    def remove_output(self, arg0: str) -> None:
        ...
    def serialize_to_string(self) -> bytes:
        ...
    def set_block_attr(self, arg0: str, arg1: BlockDesc) -> None:
        ...
    def set_blocks_attr(self, arg0: str, arg1: list[BlockDesc]) -> None:
        ...
    def set_input(self, arg0: str, arg1: list[str]) -> None:
        ...
    def set_is_target(self, arg0: bool) -> None:
        ...
    def set_original_id(self, arg0: int) -> None:
        ...
    def set_output(self, arg0: str, arg1: list[str]) -> None:
        ...
    def set_serialized_attr(self, arg0: str, arg1: bytes) -> None:
        ...
    def set_type(self, arg0: str) -> None:
        ...
    def set_var_attr(self, arg0: str, arg1: VarDesc) -> None:
        ...
    def set_vars_attr(self, arg0: str, arg1: list[VarDesc]) -> None:
        ...
    def type(self) -> str:
        ...
class OpInputOutputInfo(OpUpdateInfo):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: OpInputOutputInfo) -> None:
        ...
    def name(self) -> str:
        ...
    def remark(self) -> str:
        ...
class OpUpdateBase:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def info(self) -> OpUpdateInfo:
        ...
    def type(self) -> OpUpdateType:
        ...
class OpUpdateInfo:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class OpUpdateType:
    """
    Members:
    
      kInvalid
    
      kModifyAttr
    
      kNewAttr
    
      kNewInput
    
      kNewOutput
    
      kBugfixWithBehaviorChanged
    """
    __members__: typing.ClassVar[dict[str, OpUpdateType]]  # value = {'kInvalid': <OpUpdateType.kInvalid: 0>, 'kModifyAttr': <OpUpdateType.kModifyAttr: 1>, 'kNewAttr': <OpUpdateType.kNewAttr: 2>, 'kNewInput': <OpUpdateType.kNewInput: 3>, 'kNewOutput': <OpUpdateType.kNewOutput: 4>, 'kBugfixWithBehaviorChanged': <OpUpdateType.kBugfixWithBehaviorChanged: 5>}
    kBugfixWithBehaviorChanged: typing.ClassVar[OpUpdateType]  # value = <OpUpdateType.kBugfixWithBehaviorChanged: 5>
    kInvalid: typing.ClassVar[OpUpdateType]  # value = <OpUpdateType.kInvalid: 0>
    kModifyAttr: typing.ClassVar[OpUpdateType]  # value = <OpUpdateType.kModifyAttr: 1>
    kNewAttr: typing.ClassVar[OpUpdateType]  # value = <OpUpdateType.kNewAttr: 2>
    kNewInput: typing.ClassVar[OpUpdateType]  # value = <OpUpdateType.kNewInput: 3>
    kNewOutput: typing.ClassVar[OpUpdateType]  # value = <OpUpdateType.kNewOutput: 4>
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class OpVersion:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def checkpoints(self) -> list[OpCheckpoint]:
        ...
    def version_id(self) -> int:
        ...
class OpVersionDesc:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def infos(self) -> list:
        ...
class Operator:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @staticmethod
    def create(arg0: bytes) -> Operator:
        ...
    def __str__(self) -> str:
        ...
    def input_vars(self) -> list[str]:
        ...
    def inputs(self) -> dict[str, list[str]]:
        ...
    def no_intermediate_outputs(self) -> list[str]:
        ...
    def output_vars(self) -> list[str]:
        ...
    def outputs(self) -> dict[str, list[str]]:
        ...
    @typing.overload
    def run(self, arg0: _Scope, arg1: typing.Any) -> None:
        ...
    @typing.overload
    def run(self, arg0: _Scope, arg1: typing.Any) -> None:
        ...
    @typing.overload
    def run(self, arg0: _Scope, arg1: typing.Any) -> None:
        ...
    @typing.overload
    def run(self, arg0: _Scope, arg1: typing.Any) -> None:
        ...
    @typing.overload
    def run(self, arg0: _Scope, arg1: typing.Any) -> None:
        ...
    def support_gpu(self) -> bool:
        ...
    def type(self) -> str:
        ...
class OperatorDistAttr:
    __hash__: typing.ClassVar[None] = None
    annotated: dict[str, bool]
    chunk_id: int
    event_to_record: str
    events_to_wait: list[str]
    execution_stream: str
    force_record_event: bool
    impl_idx: int
    impl_type: str
    inputs_dist_attrs: dict[str, TensorDistAttr]
    is_recompute: bool
    op_type: str
    outputs_dist_attrs: dict[str, TensorDistAttr]
    process_mesh: ProcessMesh
    run_time_us: float
    scheduling_priority: int
    stream_priority: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __copy__(self) -> OperatorDistAttr:
        ...
    def __deepcopy__(self, memo: dict) -> OperatorDistAttr:
        ...
    def __eq__(self, arg0: OperatorDistAttr) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: OpDesc) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: OperatorDistAttr) -> None:
        ...
    def __ne__(self, arg0: OperatorDistAttr) -> bool:
        ...
    def __str__(self) -> str:
        ...
    def clear_annotated(self) -> None:
        ...
    def del_input_dist_attr(self, arg0: str) -> None:
        ...
    def del_output_dist_attr(self, arg0: str) -> None:
        ...
    def get_input_dims_mapping(self, arg0: str) -> list[int]:
        ...
    def get_input_dist_attr(self, arg0: str) -> TensorDistAttr:
        ...
    def get_output_dims_mapping(self, arg0: str) -> list[int]:
        ...
    def get_output_dist_attr(self, arg0: str) -> TensorDistAttr:
        ...
    def is_annotated(self, arg0: str) -> bool:
        ...
    def is_annotated_input_dims_mapping(self, arg0: str) -> bool:
        ...
    def is_annotated_output_dims_mapping(self, arg0: str) -> bool:
        ...
    def mark_annotated(self, arg0: str) -> None:
        ...
    def parse_from_string(self, arg0: str) -> None:
        ...
    def rename_input(self, arg0: str, arg1: str) -> None:
        ...
    def rename_output(self, arg0: str, arg1: str) -> None:
        ...
    def reset(self) -> None:
        ...
    def serialize_to_string(self) -> bytes:
        ...
    def set_input_dims_mapping(self, arg0: str, arg1: list[int]) -> None:
        ...
    def set_input_dist_attr(self, arg0: str, arg1: TensorDistAttr) -> None:
        ...
    def set_output_dims_mapping(self, arg0: str, arg1: list[int]) -> None:
        ...
    def set_output_dist_attr(self, arg0: str, arg1: TensorDistAttr) -> None:
        ...
    def verify(self, op: OpDesc = None) -> bool:
        ...
class OrderedMultiDeviceDenseTensorBlockingQueue:
    """
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def capacity(self) -> int:
        ...
    def close(self) -> None:
        ...
    def kill(self) -> None:
        ...
    def push(self, arg0: typing.Any) -> bool:
        ...
    def reset(self) -> None:
        ...
    def size(self) -> int:
        ...
    def wait_for_inited(self, arg0: int) -> bool:
        ...
class OrderedMultiDeviceFeedReader:
    """
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def read_next(self) -> list[dict[str, typing.Any]]:
        ...
    def read_next_list(self) -> list[typing.Any]:
        ...
    def read_next_var_list(self) -> list[typing.Any]:
        ...
    def reset(self) -> None:
        ...
    def shutdown(self) -> None:
        ...
class P2POption:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class PToRReshardFunction(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class PToRReshardFunctionCrossMesh(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class PToSReshardFunction(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class PaddleBuf:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, arg0: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: list[float]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int32]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int64]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.float32]) -> None:
        ...
    def empty(self) -> bool:
        ...
    def float_data(self) -> list[float]:
        ...
    def int32_data(self) -> list[int]:
        ...
    def int64_data(self) -> list[int]:
        ...
    def length(self) -> int:
        ...
    @typing.overload
    def reset(self, arg0: list[float]) -> None:
        ...
    @typing.overload
    def reset(self, arg0: numpy.ndarray[numpy.int32]) -> None:
        ...
    @typing.overload
    def reset(self, arg0: numpy.ndarray[numpy.int64]) -> None:
        ...
    @typing.overload
    def reset(self, arg0: numpy.ndarray[numpy.float32]) -> None:
        ...
    def resize(self, arg0: int) -> None:
        ...
    def tolist(self, arg0: str) -> list:
        ...
class PaddleDType:
    """
    Members:
    
      FLOAT64
    
      FLOAT32
    
      FLOAT16
    
      BFLOAT16
    
      INT64
    
      INT32
    
      UINT8
    
      INT8
    
      BOOL
    """
    BFLOAT16: typing.ClassVar[PaddleDType]  # value = <PaddleDType.BFLOAT16: 8>
    BOOL: typing.ClassVar[PaddleDType]  # value = <PaddleDType.BOOL: 6>
    FLOAT16: typing.ClassVar[PaddleDType]  # value = <PaddleDType.FLOAT16: 5>
    FLOAT32: typing.ClassVar[PaddleDType]  # value = <PaddleDType.FLOAT32: 0>
    FLOAT64: typing.ClassVar[PaddleDType]  # value = <PaddleDType.FLOAT64: 7>
    INT32: typing.ClassVar[PaddleDType]  # value = <PaddleDType.INT32: 2>
    INT64: typing.ClassVar[PaddleDType]  # value = <PaddleDType.INT64: 1>
    INT8: typing.ClassVar[PaddleDType]  # value = <PaddleDType.INT8: 4>
    UINT8: typing.ClassVar[PaddleDType]  # value = <PaddleDType.UINT8: 3>
    __members__: typing.ClassVar[dict[str, PaddleDType]]  # value = {'FLOAT64': <PaddleDType.FLOAT64: 7>, 'FLOAT32': <PaddleDType.FLOAT32: 0>, 'FLOAT16': <PaddleDType.FLOAT16: 5>, 'BFLOAT16': <PaddleDType.BFLOAT16: 8>, 'INT64': <PaddleDType.INT64: 1>, 'INT32': <PaddleDType.INT32: 2>, 'UINT8': <PaddleDType.UINT8: 3>, 'INT8': <PaddleDType.INT8: 4>, 'BOOL': <PaddleDType.BOOL: 6>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PaddleDataLayout:
    """
    Members:
    
      UNK
    
      Any
    
      NHWC
    
      NCHW
    """
    Any: typing.ClassVar[PaddleDataLayout]  # value = <PaddleDataLayout.Any: 0>
    NCHW: typing.ClassVar[PaddleDataLayout]  # value = <PaddleDataLayout.NCHW: 2>
    NHWC: typing.ClassVar[PaddleDataLayout]  # value = <PaddleDataLayout.NHWC: 1>
    UNK: typing.ClassVar[PaddleDataLayout]  # value = <PaddleDataLayout.UNK: -1>
    __members__: typing.ClassVar[dict[str, PaddleDataLayout]]  # value = {'UNK': <PaddleDataLayout.UNK: -1>, 'Any': <PaddleDataLayout.Any: 0>, 'NHWC': <PaddleDataLayout.NHWC: 1>, 'NCHW': <PaddleDataLayout.NCHW: 2>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PaddleInferPredictor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: AnalysisConfig) -> None:
        ...
    def clear_intermediate_tensor(self) -> None:
        ...
    def clone(self) -> PaddleInferPredictor:
        ...
    def get_input_handle(self, arg0: str) -> typing.Any:
        ...
    def get_input_names(self) -> list[str]:
        ...
    def get_output_handle(self, arg0: str) -> typing.Any:
        ...
    def get_output_names(self) -> list[str]:
        ...
    def register_input_hook(self, arg0: typing.Callable[[str, str, typing.Any], None]) -> None:
        ...
    def register_output_hook(self, arg0: typing.Callable[[str, str, typing.Any], None]) -> None:
        ...
    @typing.overload
    def run(self, inputs: list[typing.Any]) -> list[typing.Any]:
        ...
    @typing.overload
    def run(self) -> None:
        ...
    def try_shrink_memory(self) -> int:
        ...
class PaddleInferTensor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def _copy_from_cpu_bind(self, arg0: numpy.ndarray[numpy.int8]) -> None:
        ...
    @typing.overload
    def _copy_from_cpu_bind(self, arg0: numpy.ndarray[numpy.uint8]) -> None:
        ...
    @typing.overload
    def _copy_from_cpu_bind(self, arg0: numpy.ndarray[numpy.int32]) -> None:
        ...
    @typing.overload
    def _copy_from_cpu_bind(self, arg0: numpy.ndarray[numpy.int64]) -> None:
        ...
    @typing.overload
    def _copy_from_cpu_bind(self, arg0: numpy.ndarray[numpy.float32]) -> None:
        ...
    @typing.overload
    def _copy_from_cpu_bind(self, arg0: numpy.ndarray[numpy.float16]) -> None:
        ...
    @typing.overload
    def _copy_from_cpu_bind(self, arg0: numpy.ndarray[numpy.float64]) -> None:
        ...
    @typing.overload
    def _copy_from_cpu_bind(self, arg0: numpy.ndarray[bool]) -> None:
        ...
    @typing.overload
    def _copy_from_cpu_bind(self, arg0: list[str]) -> None:
        ...
    def _share_external_data_bind(self, arg0: DenseTensor) -> None:
        ...
    def _share_external_data_by_ptr_name_bind(self, arg0: str, arg1: list[int], arg2: int, arg3: int) -> None:
        ...
    def _share_external_data_paddle_tensor_bind(self, arg0: typing.Any) -> None:
        ...
    def copy_from_cpu(self, data: npt.NDArray[typing.Any] | list[str]) -> None:
        """
        
            Support input type check based on tensor.copy_from_cpu.
            
        """
    def copy_to_cpu(self) -> numpy.ndarray:
        ...
    def lod(self) -> list[list[int]]:
        ...
    @typing.overload
    def reshape(self, arg0: list[int]) -> None:
        ...
    @typing.overload
    def reshape(self, arg0: int) -> None:
        ...
    def set_lod(self, arg0: list[list[int]]) -> None:
        ...
    def shape(self) -> list[int]:
        ...
    def share_external_data(self, data: paddle.Tensor) -> None:
        """
        
            Support input type check based on tensor.share_external_data.
            
        """
    def share_external_data_by_ptr_name(self, data, shape, dtype, place):
        """
        
            Support tensor.share_external_data_by_ptr_name.
            
        """
    def type(self) -> PaddleDType:
        ...
class PaddlePassBuilder:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: list[str]) -> None:
        ...
    def all_passes(self) -> list[str]:
        ...
    def analysis_passes(self) -> list[str]:
        ...
    def append_analysis_pass(self, arg0: str) -> None:
        ...
    def append_pass(self, arg0: str) -> None:
        ...
    def debug_string(self) -> str:
        ...
    def delete_pass(self, arg0: str) -> None:
        ...
    def insert_pass(self, arg0: int, arg1: str) -> None:
        ...
    def set_passes(self, arg0: list[str]) -> None:
        ...
    def turn_on_debug(self) -> None:
        ...
class PaddlePlace:
    """
    Members:
    
      UNK
    
      CPU
    
      GPU
    
      XPU
    
      CUSTOM
    """
    CPU: typing.ClassVar[PaddlePlace]  # value = <PaddlePlace.CPU: 0>
    CUSTOM: typing.ClassVar[PaddlePlace]  # value = <PaddlePlace.CUSTOM: 4>
    GPU: typing.ClassVar[PaddlePlace]  # value = <PaddlePlace.GPU: 1>
    UNK: typing.ClassVar[PaddlePlace]  # value = <PaddlePlace.UNK: -1>
    XPU: typing.ClassVar[PaddlePlace]  # value = <PaddlePlace.XPU: 2>
    __members__: typing.ClassVar[dict[str, PaddlePlace]]  # value = {'UNK': <PaddlePlace.UNK: -1>, 'CPU': <PaddlePlace.CPU: 0>, 'GPU': <PaddlePlace.GPU: 1>, 'XPU': <PaddlePlace.XPU: 2>, 'CUSTOM': <PaddlePlace.CUSTOM: 4>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PaddlePredictor:
    class Config:
        model_dir: str
        @staticmethod
        def _pybind11_conduit_v1_(*args, **kwargs):
            ...
        def __init__(self) -> None:
            ...
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def clone(self) -> PaddlePredictor:
        ...
    def get_input_names(self) -> list[str]:
        ...
    def get_input_tensor(self, arg0: str) -> typing.Any:
        ...
    def get_output_names(self) -> list[str]:
        ...
    def get_output_tensor(self, arg0: str) -> typing.Any:
        ...
    def get_serialized_program(self) -> str:
        ...
    def run(self, arg0: list[PaddleTensor]) -> list[PaddleTensor]:
        ...
    def zero_copy_run(self, switch_stream: bool = False) -> bool:
        ...
class PaddleTensor:
    data: PaddleBuf
    dtype: PaddleDType
    lod: list[list[int]]
    name: str
    shape: list[int]
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, data: numpy.ndarray[numpy.int32], name: str = '', lod: list[list[int]] = [], copy: bool = True) -> None:
        ...
    @typing.overload
    def __init__(self, data: numpy.ndarray[numpy.int64], name: str = '', lod: list[list[int]] = [], copy: bool = True) -> None:
        ...
    @typing.overload
    def __init__(self, data: numpy.ndarray[numpy.float32], name: str = '', lod: list[list[int]] = [], copy: bool = True) -> None:
        ...
    def as_ndarray(self) -> numpy.ndarray:
        ...
class ParallelStrategy:
    """
    """
    current_endpoint: str
    local_rank: int
    nranks: int
    nrings: int
    trainer_endpoints: list[str]
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class Partial(Placement):
    """
    
                     The `Partial` describes `Tensor` across multiple devices, this type of tensor has the same shape but only a fraction of the value, which can be further reduce (e.g. sum/min/max) to obtain dist_tensor, often used as an intermediate representation.
    
                     Parameters:
                       reduce_type (paddle.distributed.ReduceType): the reduce type of the Partial state, default `paddle.distributed.ReduceType.kRedSum`.
    
                     Examples:
                         .. code-block:: python
    
                             >>> import paddle
                             >>> import paddle.distributed as dist
                             >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
                             >>> a = paddle.ones([10, 20])
                             >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                             >>> # distributed tensor
                             >>> d_tensor = dist.shard_tensor(a, mesh, [dist.Partial()])
    
                     
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __copy__(self) -> Partial:
        ...
    def __deepcopy__(self, memo: dict) -> Partial:
        ...
    def __eq__(self, arg0: Partial) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __init__(self, reduce_type: ReduceType = typing.Any) -> None:
        ...
    def __ne__(self, arg0: Partial) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def reduce_type(self) -> ReduceType:
        ...
class Pass:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def apply(self, arg0: typing.Any) -> None:
        ...
    def has(self, arg0: str) -> bool:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: str) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: bool) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: int) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: list[str]) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: set[str]) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: set[int]) -> None:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: dict[str, tuple[bool, typing.Any]]) -> None:
        ...
    def set_not_owned(self, arg0: str, arg1: ProgramDesc) -> None:
        ...
    def type(self) -> str:
        ...
class PassBuilder:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def all_passes(self) -> list[Pass]:
        ...
    def append_pass(self, arg0: str) -> Pass:
        ...
    def insert_pass(self, arg0: int, arg1: str) -> Pass:
        ...
    def remove_pass(self, arg0: int) -> None:
        ...
class PassStrategy(PaddlePassBuilder):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: list[str]) -> None:
        ...
    def enable_cudnn(self) -> None:
        ...
    def enable_mkldnn(self) -> None:
        ...
    def enable_mkldnn_bfloat16(self) -> None:
        ...
    def use_gpu(self) -> bool:
        ...
class PassVersionChecker:
    @staticmethod
    def IsCompatible(arg0: str) -> bool:
        ...
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
class Place:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @typing.overload
    def _equals(self, arg0: Place) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CUDAPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CPUPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: XPUPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: IPUPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CUDAPinnedPlace) -> bool:
        ...
    @typing.overload
    def _equals(self, arg0: CustomPlace) -> bool:
        ...
    def _type(self) -> int:
        ...
    def custom_device_id(self) -> int:
        ...
    def custom_device_type(self) -> str:
        ...
    def gpu_device_id(self) -> int:
        ...
    def ipu_device_id(self) -> int:
        ...
    def is_cpu_place(self) -> bool:
        ...
    def is_cuda_pinned_place(self) -> bool:
        ...
    def is_custom_place(self) -> bool:
        ...
    def is_gpu_place(self) -> bool:
        ...
    def is_ipu_place(self) -> bool:
        ...
    def is_xpu_place(self) -> bool:
        ...
    @typing.overload
    def set_place(self, arg0: Place) -> None:
        ...
    @typing.overload
    def set_place(self, arg0: CPUPlace) -> None:
        ...
    @typing.overload
    def set_place(self, arg0: XPUPlace) -> None:
        ...
    @typing.overload
    def set_place(self, arg0: CUDAPlace) -> None:
        ...
    @typing.overload
    def set_place(self, arg0: CUDAPinnedPlace) -> None:
        ...
    @typing.overload
    def set_place(self, arg0: IPUPlace) -> None:
        ...
    @typing.overload
    def set_place(self, arg0: CustomPlace) -> None:
        ...
    def xpu_device_id(self) -> int:
        ...
class Placement:
    """
    
            The `Placement` is base class that describes how to place the tensor on ProcessMesh. it has three subclass: `Replicate`, `Shard` and `Partial`.
    
            Examples:
                .. code-block:: python
    
                    >>> import paddle.distributed as dist
                    >>> placements = [dist.Replicate(), dist.Shard(0), dist.Partial()]
                    >>> for p in placements:
                    >>>     if isinstance(p, dist.Placement):
                    >>>         if p.is_replicated():
                    >>>             print("replicate.")
                    >>>         elif p.is_shard():
                    >>>             print("shard.")
                    >>>         elif p.is_partial():
                    >>>             print("partial.")
    
          
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __copy__(self) -> Placement:
        ...
    def __deepcopy__(self, memo: dict) -> Placement:
        ...
    def __eq__(self, arg0: Placement) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __init__(self) -> None:
        ...
    def __ne__(self, arg0: Placement) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def is_partial(self) -> bool:
        ...
    def is_replicated(self) -> bool:
        ...
    def is_shard(self, dim: int | None = None) -> bool:
        ...
class Plan:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, job_list: list[Job], type_to_program: dict[str, typing.Any]) -> None:
        ...
    @typing.overload
    def __init__(self, job_list: list[Job], type_to_ir_program: dict[str, typing.Any]) -> None:
        ...
    def ir_program(self, arg0: str) -> typing.Any:
        ...
    def job_list(self) -> list[Job]:
        ...
    def job_types(self) -> list[str]:
        ...
    def micro_batch_num(self) -> int:
        ...
    def program(self, arg0: str) -> typing.Any:
        ...
    def set_ir_program(self, arg0: str, arg1: typing.Any) -> None:
        ...
class PredictorPool:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: AnalysisConfig, arg1: int) -> None:
        ...
    def retrieve(self, arg0: int) -> PaddleInferPredictor:
        ...
class ProcessGroup:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def _end_coalescing(self, tasks: list[typing.Any] | None = None) -> None:
        ...
    def _start_coalescing(self) -> None:
        ...
    @typing.overload
    def all_gather(self, out: typing.Any, in_: typing.Any, sync_op: bool) -> typing.Any:
        ...
    @typing.overload
    def all_gather(self, in_: typing.Any, out: typing.Any) -> typing.Any:
        ...
    def all_gather_into_tensor(self, out: typing.Any, in_: typing.Any, sync_op: bool) -> typing.Any:
        ...
    def all_gather_into_tensor_on_calc_stream(self, out: typing.Any, in_: typing.Any) -> typing.Any:
        ...
    def all_gather_on_calc_stream(self, out: typing.Any, in_: typing.Any) -> typing.Any:
        ...
    def all_gather_partial(self, out: typing.Any, in_: typing.Any, num: int, id: int) -> typing.Any:
        ...
    def all_gather_partial_on_calc_stream(self, out: typing.Any, in_: typing.Any, num: int, id: int) -> typing.Any:
        ...
    def all_reduce(self, tensor: typing.Any, op: ReduceOp, sync_op: bool) -> typing.Any:
        ...
    def all_reduce_on_calc_stream(self, tensor: typing.Any, op: ReduceOp = typing.Any) -> typing.Any:
        ...
    def all_to_all(self, out: typing.Any, in_: typing.Any, sync_op: bool) -> typing.Any:
        ...
    def all_to_all_on_calc_stream(self, out: typing.Any, in_: typing.Any) -> typing.Any:
        ...
    def all_to_all_single(self, out: typing.Any, in_: typing.Any, out_sizes: list[int], in_sizes: list[int], sync_op: bool) -> typing.Any:
        ...
    def all_to_all_single_on_calc_stream(self, out: typing.Any, in_: typing.Any, out_sizes: list[int], in_sizes: list[int]) -> typing.Any:
        ...
    def all_to_all_tensor(self, out: typing.Any, in_: typing.Any, sync_op: bool) -> typing.Any:
        ...
    def all_to_all_tensor_on_calc_stream(self, out: typing.Any, in_: typing.Any) -> typing.Any:
        ...
    def allreduce(self, tensor: typing.Any, op: ReduceOp = typing.Any) -> typing.Any:
        ...
    def alltoall(self, in_: typing.Any, out: typing.Any) -> typing.Any:
        ...
    def alltoall_single(self, in_: typing.Any, out: typing.Any, in_sizes: list[int], out_sizes: list[int]) -> typing.Any:
        ...
    def barrier(self, device_id: int = -1) -> typing.Any:
        ...
    @typing.overload
    def broadcast(self, tensor: typing.Any, src: int, sync_op: bool) -> typing.Any:
        ...
    @typing.overload
    def broadcast(self, tensor: typing.Any, source_rank: int) -> typing.Any:
        ...
    def broadcast_on_calc_stream(self, tensor: typing.Any, src: int) -> typing.Any:
        ...
    def eager_connect(self) -> None:
        ...
    def eager_connect_ring_exchange(self) -> None:
        ...
    def gather(self, in_: typing.Any, out: typing.Any, dst: int, sync_op: bool, use_calc_stream: bool = False) -> typing.Any:
        ...
    def name(self) -> str:
        ...
    def rank(self) -> int:
        ...
    @typing.overload
    def recv(self, tensor: typing.Any, src: int, sync_op: bool) -> typing.Any:
        ...
    @typing.overload
    def recv(self, tensor: typing.Any, src: int) -> typing.Any:
        ...
    def recv_on_calc_stream(self, tensor: typing.Any, src: int) -> typing.Any:
        ...
    def recv_partial(self, tensor: typing.Any, src: int, num: int, id: int, sync_op: bool = True) -> typing.Any:
        ...
    def recv_partial_on_calc_stream(self, tensor: typing.Any, src: int, num: int, id: int) -> typing.Any:
        ...
    @typing.overload
    def reduce(self, tensor: typing.Any, dst: int, op: ReduceOp, sync_op: bool) -> typing.Any:
        ...
    @typing.overload
    def reduce(self, tensor: typing.Any, dst: int, op: ReduceOp = typing.Any) -> typing.Any:
        ...
    def reduce_on_calc_stream(self, tensor: typing.Any, dst: int, op: ReduceOp) -> typing.Any:
        ...
    def reduce_scatter(self, out: typing.Any, in_: typing.Any, op: ReduceOp, sync_op: bool) -> typing.Any:
        ...
    def reduce_scatter_on_calc_stream(self, out: typing.Any, in_: typing.Any, op: ReduceOp) -> typing.Any:
        ...
    def reduce_scatter_tensor(self, out: typing.Any, in_: typing.Any, op: ReduceOp, sync_op: bool) -> typing.Any:
        ...
    def reduce_scatter_tensor_on_calc_stream(self, out: typing.Any, in_: typing.Any, op: ReduceOp) -> typing.Any:
        ...
    @typing.overload
    def scatter(self, out: typing.Any, in_: typing.Any, src: int, sync_op: bool) -> typing.Any:
        ...
    @typing.overload
    def scatter(self, in_: typing.Any, out: typing.Any, src: int) -> typing.Any:
        ...
    def scatter_on_calc_stream(self, out: typing.Any, in_: typing.Any, src: int) -> typing.Any:
        ...
    def scatter_tensor(self, out: typing.Any, in_: typing.Any, src: int, sync_op: bool) -> typing.Any:
        ...
    def scatter_tensor_on_calc_stream(self, out: typing.Any, in_: typing.Any, src: int) -> typing.Any:
        ...
    @typing.overload
    def send(self, tensor: typing.Any, dst: int, sync_op: bool) -> typing.Any:
        ...
    @typing.overload
    def send(self, tensor: typing.Any, dst: int) -> typing.Any:
        ...
    def send_on_calc_stream(self, tensor: typing.Any, dst: int) -> typing.Any:
        ...
    def send_partial(self, tensor: typing.Any, dst: int, num: int, id: int, sync_op: bool = True) -> typing.Any:
        ...
    def send_partial_on_calc_stream(self, tensor: typing.Any, dst: int, num: int, id: int) -> typing.Any:
        ...
    def size(self) -> int:
        ...
class ProcessGroupIdMap:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @staticmethod
    def destroy() -> None:
        ...
class ProcessMesh:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __copy__(self) -> ProcessMesh:
        ...
    def __deepcopy__(self, memo: dict) -> ProcessMesh:
        ...
    def __eq__(self, arg0: ProcessMesh) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, shape: list[int], process_ids: list[int], dim_names: list[str]) -> None:
        ...
    def __ne__(self, arg0: ProcessMesh) -> bool:
        ...
    def __str__(self) -> str:
        ...
    def contains(self, arg0: int) -> bool:
        ...
    @typing.overload
    def dim_size(self, arg0: int) -> int:
        ...
    @typing.overload
    def dim_size(self, arg0: str) -> int:
        ...
    def empty(self) -> bool:
        ...
    @property
    def dim_names(self) -> list[str]:
        ...
    @property
    def ndim(self) -> int:
        ...
    @property
    def process_ids(self) -> list[int]:
        ...
    @property
    def shape(self) -> list[int]:
        ...
    @property
    def size(self) -> int:
        ...
class ProfilerOptions:
    trace_switch: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class ProfilerState:
    """
    Members:
    
      kDisabled
    
      kCPU
    
      kCUDA
    
      kAll
    """
    __members__: typing.ClassVar[dict[str, ProfilerState]]  # value = {'kDisabled': <ProfilerState.kDisabled: 0>, 'kCPU': <ProfilerState.kCPU: 1>, 'kCUDA': <ProfilerState.kCUDA: 2>, 'kAll': <ProfilerState.kAll: 3>}
    kAll: typing.ClassVar[ProfilerState]  # value = <ProfilerState.kAll: 3>
    kCPU: typing.ClassVar[ProfilerState]  # value = <ProfilerState.kCPU: 1>
    kCUDA: typing.ClassVar[ProfilerState]  # value = <ProfilerState.kCUDA: 2>
    kDisabled: typing.ClassVar[ProfilerState]  # value = <ProfilerState.kDisabled: 0>
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ProgramDesc:
    """
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: ProgramDesc) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: bytes) -> None:
        ...
    def _set_version(self, version: int = 3000000) -> None:
        ...
    def _version(self) -> int:
        ...
    def append_block(self, arg0: typing.Any) -> typing.Any:
        ...
    def block(self, arg0: int) -> typing.Any:
        ...
    def cached_hash_str(self) -> str:
        ...
    def flush(self) -> None:
        ...
    def get_feed_target_names(self) -> list[str]:
        ...
    def get_fetch_target_names(self) -> list[str]:
        ...
    def get_op_deps(self) -> list[list[list[typing.Any]]]:
        ...
    @typing.overload
    def need_update(self) -> bool:
        ...
    @typing.overload
    def need_update(self) -> bool:
        ...
    def num_blocks(self) -> int:
        ...
    def parse_from_string(self, arg0: str) -> None:
        ...
    def serialize_to_string(self, legacy_format: bool = False) -> bytes:
        ...
class Property:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    @typing.overload
    def get_float(self, arg0: int) -> float:
        ...
    @typing.overload
    def get_float(self, arg0: str) -> float:
        ...
    def parse_from_string(self, arg0: str) -> None:
        ...
    def serialize_to_string(self) -> bytes:
        ...
    def set_float(self, name: str, var: float) -> None:
        """
        set float
        """
    def set_floats(self, name: str, val: list[float]) -> None:
        """
        set list of float
        """
    def set_int(self, name: str, val: int) -> None:
        """
        set int
        """
    def set_ints(self, name: str, val: list[int]) -> None:
        """
        set list of int
        """
    def set_string(self, name: str, val: str) -> None:
        """
        set string
        """
    def set_strings(self, name: str, val: list[str]) -> None:
        """
        set list of string
        """
    def size(self) -> int:
        ...
class RToPReshardFunction(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class RToPReshardFunctionCrossMesh(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class RToSReshardFunction(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class RToSReshardFunctionCrossMesh(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class RangeMatchGuard(GuardBase):
    """
    RangeMatchGuard Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, range_obj: typing.Any) -> None:
        ...
class Reader:
    """
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def reset(self) -> None:
        ...
    def start(self) -> None:
        ...
class ReduceOp:
    """
    Members:
    
      SUM
    
      AVG
    
      MAX
    
      MIN
    
      PRODUCT
    """
    AVG: typing.ClassVar[ReduceOp]  # value = <ReduceOp.AVG: 4>
    MAX: typing.ClassVar[ReduceOp]  # value = <ReduceOp.MAX: 1>
    MIN: typing.ClassVar[ReduceOp]  # value = <ReduceOp.MIN: 2>
    PRODUCT: typing.ClassVar[ReduceOp]  # value = <ReduceOp.PRODUCT: 3>
    SUM: typing.ClassVar[ReduceOp]  # value = <ReduceOp.SUM: 0>
    __members__: typing.ClassVar[dict[str, ReduceOp]]  # value = {'SUM': <ReduceOp.SUM: 0>, 'AVG': <ReduceOp.AVG: 4>, 'MAX': <ReduceOp.MAX: 1>, 'MIN': <ReduceOp.MIN: 2>, 'PRODUCT': <ReduceOp.PRODUCT: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ReduceOptions:
    reduce_op: ReduceOp
    source_root: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class ReduceType:
    """
    
        Specify the type of operation used for paddle.distributed.Partial().
        It should be one of the following values:
    
            - ReduceType.kRedSum
            - ReduceType.kRedMax
            - ReduceType.kRedMin
            - ReduceType.kRedProd
            - ReduceType.kRedAvg
            - ReduceType.kRedAny
            - ReduceType.kRedAll
    
        Examples:
            .. code-block:: python
    
                >>> import paddle
                >>> import paddle.distributed as dist
                >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
                >>> a = paddle.ones([10, 20])
                >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                >>> # distributed tensor
                >>> d_tensor = dist.shard_tensor(a, mesh, [dist.Partial(dist.ReduceType.kRedSum)])
    
          
    
    Members:
    
      kRedSum
    
      kRedMax
    
      kRedMin
    
      kRedProd
    
      kRedAvg
    
      kRedAny
    
      kRedAll
    """
    __members__: typing.ClassVar[dict[str, ReduceType]]  # value = {'kRedSum': <ReduceType.kRedSum: 0>, 'kRedMax': <ReduceType.kRedMax: 1>, 'kRedMin': <ReduceType.kRedMin: 2>, 'kRedProd': <ReduceType.kRedProd: 3>, 'kRedAvg': <ReduceType.kRedAvg: 4>, 'kRedAny': <ReduceType.kRedAny: 5>, 'kRedAll': <ReduceType.kRedAll: 6>}
    kRedAll: typing.ClassVar[ReduceType]  # value = <ReduceType.kRedAll: 6>
    kRedAny: typing.ClassVar[ReduceType]  # value = <ReduceType.kRedAny: 5>
    kRedAvg: typing.ClassVar[ReduceType]  # value = <ReduceType.kRedAvg: 4>
    kRedMax: typing.ClassVar[ReduceType]  # value = <ReduceType.kRedMax: 1>
    kRedMin: typing.ClassVar[ReduceType]  # value = <ReduceType.kRedMin: 2>
    kRedProd: typing.ClassVar[ReduceType]  # value = <ReduceType.kRedProd: 3>
    kRedSum: typing.ClassVar[ReduceType]  # value = <ReduceType.kRedSum: 0>
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Replicate(Placement):
    """
    
                       The `Replicate` describes the tensor placed repeatedly on ProcessMesh.
    
                       Examples:
                           .. code-block:: python
    
                               >>> import paddle
                               >>> import paddle.distributed as dist
                               >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
                               >>> a = paddle.ones([10, 20])
                               >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                               >>> # distributed tensor
                               >>> d_tensor = dist.shard_tensor(a, mesh, [dist.Replicate()])
    
                       
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __copy__(self) -> Replicate:
        ...
    def __deepcopy__(self, memo: dict) -> Replicate:
        ...
    def __eq__(self, arg0: Replicate) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __init__(self) -> None:
        ...
    def __ne__(self, arg0: Replicate) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class ReshardFunction:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def eval(self, arg0: DeviceContext, arg1: typing.Any, arg2: typing.Any) -> typing.Any:
        ...
    def is_suitable(self, arg0: typing.Any, arg1: typing.Any) -> bool:
        ...
class SToPReshardFunction(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class SToRReshardFunction(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class SToRReshardFunctionCrossMesh(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class SToSReshardFunction(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class SameNdMeshReshardFunction(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class SameStatusReshardFunction(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class Scalar:
    """
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, arg0: Scalar) -> bool:
        ...
    @typing.overload
    def __init__(self, arg0: bool) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: float) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: complex) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: Scalar) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def value(self) -> typing.Any:
        ...
class SelectedRows:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: list[int], arg1: int) -> None:
        ...
    def get_tensor(self) -> DenseTensor:
        ...
    def height(self) -> int:
        ...
    def numel(self) -> int:
        ...
    def rows(self) -> list[int]:
        ...
    def set_height(self, arg0: int) -> None:
        ...
    def set_rows(self, arg0: list[int]) -> None:
        ...
    def sync_index(self) -> None:
        ...
class ShapeMatchGuard(GuardBase):
    """
    ShapeMatchGuard Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, shape: list[typing.Any]) -> None:
        ...
class ShapeMode:
    """
    Members:
    
      kMIN
    
      kMAX
    
      kOPT
    """
    __members__: typing.ClassVar[dict[str, ShapeMode]]  # value = {'kMIN': <ShapeMode.kMIN: 0>, 'kMAX': <ShapeMode.kMAX: 2>, 'kOPT': <ShapeMode.kOPT: 1>}
    kMAX: typing.ClassVar[ShapeMode]  # value = <ShapeMode.kMAX: 2>
    kMIN: typing.ClassVar[ShapeMode]  # value = <ShapeMode.kMIN: 0>
    kOPT: typing.ClassVar[ShapeMode]  # value = <ShapeMode.kOPT: 1>
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Shard(Placement):
    """
    
                   The `Shard` describes how `Tensor` splitted across multiple devices according to specified dimensions.
    
                   Parameters:
                       dim (int): specify the slicing dimension of the tensor.
    
                   Examples:
                       .. code-block:: python
    
                           >>> import paddle
                           >>> import paddle.distributed as dist
                           >>> mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=['x', 'y'])
                           >>> a = paddle.to_tensor([[1,2,3],[5,6,7]])
                           >>> # doctest: +REQUIRES(env:DISTRIBUTED)
                           >>> # distributed tensor
                           >>> d_tensor = dist.shard_tensor(a, mesh, [dist.Shard(0), dist.Shard(1)])
    
                   
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __copy__(self) -> Shard:
        ...
    def __deepcopy__(self, memo: dict) -> Shard:
        ...
    def __eq__(self, arg0: Shard) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __init__(self, arg0: int) -> None:
        ...
    def __ne__(self, arg0: Shard) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def get_dim(self) -> int:
        ...
class SparseCooTensor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def indices(self) -> DenseTensor:
        ...
    def numel(self) -> int:
        ...
class SpmdRule:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def infer_backward(self, *args) -> tuple[list[typing.Any], list[typing.Any]]:
        ...
    def infer_forward(self, *args) -> tuple[list[typing.Any], list[typing.Any]]:
        ...
class StandaloneExecutor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: typing.Any, arg1: typing.Any, arg2: _Scope) -> None:
        ...
    def run(self, arg0: list[str], arg1: bool) -> typing.Any:
        ...
    def run_profile(self, arg0: list[str]) -> typing.Any:
        ...
class Store:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def add(self, arg0: str, arg1: int) -> int:
        ...
    def get(self, key: str) -> bytes:
        ...
    def set(self, key: str, value: str) -> None:
        ...
    def wait(self, arg0: str) -> None:
        ...
class TCPStore(Store):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, hostname: str, port: int, is_master: bool, world_size: int, timeout: int = 900) -> None:
        ...
class TRTEngineParams:
    engine_serialized_data: str
    max_input_shape: dict[str, list[int]]
    max_shape_tensor: dict[str, list[int]]
    max_workspace_size: int
    min_input_shape: dict[str, list[int]]
    min_shape_tensor: dict[str, list[int]]
    optim_input_shape: dict[str, list[int]]
    optim_shape_tensor: dict[str, list[int]]
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class TensorDistAttr:
    __hash__: typing.ClassVar[None] = None
    annotated: dict[str, bool]
    batch_dim: int
    chunk_id: int
    dims_mapping: list[int]
    dynamic_dims: list[bool]
    process_mesh: ProcessMesh
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __copy__(self) -> TensorDistAttr:
        ...
    def __deepcopy__(self, memo: dict) -> TensorDistAttr:
        ...
    def __eq__(self, arg0: TensorDistAttr) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: VarDesc) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: TensorDistAttr) -> None:
        ...
    def __ne__(self, arg0: TensorDistAttr) -> bool:
        ...
    def __str__(self) -> str:
        ...
    def _clean_partial_dims(self, arg0: list[int]) -> None:
        ...
    def _clean_partial_status(self) -> None:
        ...
    def _is_partial(self, mesh_axis: int = -1) -> bool:
        ...
    def _partial_dims(self) -> set[int]:
        ...
    def _set_partial_dims(self, arg0: list[int]) -> None:
        ...
    def clear_annotated(self) -> None:
        ...
    def is_annotated(self, arg0: str) -> bool:
        ...
    def mark_annotated(self, arg0: str) -> None:
        ...
    def parse_from_string(self, arg0: str) -> None:
        ...
    def reset(self) -> None:
        ...
    def serialize_to_string(self) -> bytes:
        ...
    def verify(self, tensor: VarDesc = None) -> bool:
        ...
class Tracer:
    """
    """
    _amp_dtype: str
    _amp_level: AmpLevel
    _expected_place: typing.Any
    _has_grad: bool
    _use_promote: bool
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def _generate_unique_name(self, key: str = 'dygraph_tmp') -> str:
        ...
    def _get_amp_op_list(self) -> tuple[set[str], set[str]]:
        ...
    def _get_kernel_signature(self, arg0: str, arg1: dict[str, typing.Any], arg2: dict[str, typing.Any], arg3: dict[str, typing.Any]) -> tuple[list[str], list[str], list[str]]:
        ...
    def _set_amp_op_list(self, arg0: set[str], arg1: set[str]) -> None:
        ...
class TracerEventType:
    """
    Members:
    
      Operator
    
      Dataloader
    
      ProfileStep
    
      CudaRuntime
    
      Kernel
    
      Memcpy
    
      Memset
    
      UserDefined
    
      OperatorInner
    
      Forward
    
      Backward
    
      Optimization
    
      Communication
    
      PythonOp
    
      PythonUserDefined
    
      DygraphKernelLaunch
    
      StaticKernelLaunch
    """
    Backward: typing.ClassVar[TracerEventType]  # value = <TracerEventType.Backward: 10>
    Communication: typing.ClassVar[TracerEventType]  # value = <TracerEventType.Communication: 12>
    CudaRuntime: typing.ClassVar[TracerEventType]  # value = <TracerEventType.CudaRuntime: 3>
    Dataloader: typing.ClassVar[TracerEventType]  # value = <TracerEventType.Dataloader: 1>
    DygraphKernelLaunch: typing.ClassVar[TracerEventType]  # value = <TracerEventType.DygraphKernelLaunch: 15>
    Forward: typing.ClassVar[TracerEventType]  # value = <TracerEventType.Forward: 9>
    Kernel: typing.ClassVar[TracerEventType]  # value = <TracerEventType.Kernel: 4>
    Memcpy: typing.ClassVar[TracerEventType]  # value = <TracerEventType.Memcpy: 5>
    Memset: typing.ClassVar[TracerEventType]  # value = <TracerEventType.Memset: 6>
    Operator: typing.ClassVar[TracerEventType]  # value = <TracerEventType.Operator: 0>
    OperatorInner: typing.ClassVar[TracerEventType]  # value = <TracerEventType.OperatorInner: 8>
    Optimization: typing.ClassVar[TracerEventType]  # value = <TracerEventType.Optimization: 11>
    ProfileStep: typing.ClassVar[TracerEventType]  # value = <TracerEventType.ProfileStep: 2>
    PythonOp: typing.ClassVar[TracerEventType]  # value = <TracerEventType.PythonOp: 13>
    PythonUserDefined: typing.ClassVar[TracerEventType]  # value = <TracerEventType.PythonUserDefined: 14>
    StaticKernelLaunch: typing.ClassVar[TracerEventType]  # value = <TracerEventType.StaticKernelLaunch: 16>
    UserDefined: typing.ClassVar[TracerEventType]  # value = <TracerEventType.UserDefined: 7>
    __members__: typing.ClassVar[dict[str, TracerEventType]]  # value = {'Operator': <TracerEventType.Operator: 0>, 'Dataloader': <TracerEventType.Dataloader: 1>, 'ProfileStep': <TracerEventType.ProfileStep: 2>, 'CudaRuntime': <TracerEventType.CudaRuntime: 3>, 'Kernel': <TracerEventType.Kernel: 4>, 'Memcpy': <TracerEventType.Memcpy: 5>, 'Memset': <TracerEventType.Memset: 6>, 'UserDefined': <TracerEventType.UserDefined: 7>, 'OperatorInner': <TracerEventType.OperatorInner: 8>, 'Forward': <TracerEventType.Forward: 9>, 'Backward': <TracerEventType.Backward: 10>, 'Optimization': <TracerEventType.Optimization: 11>, 'Communication': <TracerEventType.Communication: 12>, 'PythonOp': <TracerEventType.PythonOp: 13>, 'PythonUserDefined': <TracerEventType.PythonUserDefined: 14>, 'DygraphKernelLaunch': <TracerEventType.DygraphKernelLaunch: 15>, 'StaticKernelLaunch': <TracerEventType.StaticKernelLaunch: 16>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TracerMemEventType:
    """
    Members:
    
      Allocate
    
      Free
    
      ReservedAllocate
    
      ReservedFree
    """
    Allocate: typing.ClassVar[TracerMemEventType]  # value = <TracerMemEventType.Allocate: 0>
    Free: typing.ClassVar[TracerMemEventType]  # value = <TracerMemEventType.Free: 1>
    ReservedAllocate: typing.ClassVar[TracerMemEventType]  # value = <TracerMemEventType.ReservedAllocate: 2>
    ReservedFree: typing.ClassVar[TracerMemEventType]  # value = <TracerMemEventType.ReservedFree: 3>
    __members__: typing.ClassVar[dict[str, TracerMemEventType]]  # value = {'Allocate': <TracerMemEventType.Allocate: 0>, 'Free': <TracerMemEventType.Free: 1>, 'ReservedAllocate': <TracerMemEventType.ReservedAllocate: 2>, 'ReservedFree': <TracerMemEventType.ReservedFree: 3>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TracerOption:
    """
    Members:
    
      kDefault
    
      kOpDetail
    
      kAllOpDetail
    """
    __members__: typing.ClassVar[dict[str, TracerOption]]  # value = {'kDefault': <TracerOption.kDefault: 0>, 'kOpDetail': <TracerOption.kOpDetail: 1>, 'kAllOpDetail': <TracerOption.kAllOpDetail: 2>}
    kAllOpDetail: typing.ClassVar[TracerOption]  # value = <TracerOption.kAllOpDetail: 2>
    kDefault: typing.ClassVar[TracerOption]  # value = <TracerOption.kDefault: 0>
    kOpDetail: typing.ClassVar[TracerOption]  # value = <TracerOption.kOpDetail: 1>
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TrainerBase:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def ResetDataset(self, arg0: typing.Any) -> None:
        ...
    def finalize(self) -> None:
        ...
    def get_worker_scope(self, arg0: int) -> _Scope:
        ...
class TypeMatchGuard(GuardBase):
    """
    TypeMatchGuard Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, py_type: type) -> None:
        ...
class ValueMatchGuard(GuardBase):
    """
    ValueMatchGuard Class.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, py_value: typing.Any) -> None:
        ...
class VarDesc:
    """
    """
    class VarType:
        """
        
        
        Members:
        
          BOOL
        
          UINT8
        
          INT8
        
          INT16
        
          INT32
        
          INT64
        
          FP16
        
          FP32
        
          FP64
        
          BF16
        
          COMPLEX64
        
          COMPLEX128
        
          FP8_E4M3FN
        
          FP8_E5M2
        
          DENSE_TENSOR
        
          SELECTED_ROWS
        
          FEED_MINIBATCH
        
          FETCH_LIST
        
          STEP_SCOPES
        
          DENSE_TENSOR_ARRAY
        
          PLACE_LIST
        
          READER
        
          RAW
        
          STRING
        
          STRINGS
        
          VOCAB
        
          SPARSE_COO
        """
        BF16: typing.ClassVar[VarDesc.VarType]  # value = <VarType.BF16: 22>
        BOOL: typing.ClassVar[VarDesc.VarType]  # value = <VarType.BOOL: 0>
        COMPLEX128: typing.ClassVar[VarDesc.VarType]  # value = <VarType.COMPLEX128: 24>
        COMPLEX64: typing.ClassVar[VarDesc.VarType]  # value = <VarType.COMPLEX64: 23>
        DENSE_TENSOR: typing.ClassVar[VarDesc.VarType]  # value = <VarType.DENSE_TENSOR: 7>
        DENSE_TENSOR_ARRAY: typing.ClassVar[VarDesc.VarType]  # value = <VarType.DENSE_TENSOR_ARRAY: 13>
        FEED_MINIBATCH: typing.ClassVar[VarDesc.VarType]  # value = <VarType.FEED_MINIBATCH: 9>
        FETCH_LIST: typing.ClassVar[VarDesc.VarType]  # value = <VarType.FETCH_LIST: 10>
        FP16: typing.ClassVar[VarDesc.VarType]  # value = <VarType.FP16: 4>
        FP32: typing.ClassVar[VarDesc.VarType]  # value = <VarType.FP32: 5>
        FP64: typing.ClassVar[VarDesc.VarType]  # value = <VarType.FP64: 6>
        FP8_E4M3FN: typing.ClassVar[VarDesc.VarType]  # value = <VarType.FP8_E4M3FN: 32>
        FP8_E5M2: typing.ClassVar[VarDesc.VarType]  # value = <VarType.FP8_E5M2: 33>
        INT16: typing.ClassVar[VarDesc.VarType]  # value = <VarType.INT16: 1>
        INT32: typing.ClassVar[VarDesc.VarType]  # value = <VarType.INT32: 2>
        INT64: typing.ClassVar[VarDesc.VarType]  # value = <VarType.INT64: 3>
        INT8: typing.ClassVar[VarDesc.VarType]  # value = <VarType.INT8: 21>
        LOD_TENSOR: typing.ClassVar[VarDesc.VarType]  # value = <VarType.DENSE_TENSOR: 7>
        LOD_TENSOR_ARRAY: typing.ClassVar[VarDesc.VarType]  # value = <VarType.DENSE_TENSOR_ARRAY: 13>
        PLACE_LIST: typing.ClassVar[VarDesc.VarType]  # value = <VarType.PLACE_LIST: 14>
        RAW: typing.ClassVar[VarDesc.VarType]  # value = <VarType.RAW: 17>
        READER: typing.ClassVar[VarDesc.VarType]  # value = <VarType.READER: 15>
        SELECTED_ROWS: typing.ClassVar[VarDesc.VarType]  # value = <VarType.SELECTED_ROWS: 8>
        SPARSE_COO: typing.ClassVar[VarDesc.VarType]  # value = <VarType.SPARSE_COO: 30>
        STEP_SCOPES: typing.ClassVar[VarDesc.VarType]  # value = <VarType.STEP_SCOPES: 11>
        STRING: typing.ClassVar[VarDesc.VarType]  # value = <VarType.STRING: 25>
        STRINGS: typing.ClassVar[VarDesc.VarType]  # value = <VarType.STRINGS: 26>
        UINT8: typing.ClassVar[VarDesc.VarType]  # value = <VarType.UINT8: 20>
        VOCAB: typing.ClassVar[VarDesc.VarType]  # value = <VarType.VOCAB: 27>
        __members__: typing.ClassVar[dict[str, VarDesc.VarType]]  # value = {'BOOL': <VarType.BOOL: 0>, 'UINT8': <VarType.UINT8: 20>, 'INT8': <VarType.INT8: 21>, 'INT16': <VarType.INT16: 1>, 'INT32': <VarType.INT32: 2>, 'INT64': <VarType.INT64: 3>, 'FP16': <VarType.FP16: 4>, 'FP32': <VarType.FP32: 5>, 'FP64': <VarType.FP64: 6>, 'BF16': <VarType.BF16: 22>, 'COMPLEX64': <VarType.COMPLEX64: 23>, 'COMPLEX128': <VarType.COMPLEX128: 24>, 'FP8_E4M3FN': <VarType.FP8_E4M3FN: 32>, 'FP8_E5M2': <VarType.FP8_E5M2: 33>, 'DENSE_TENSOR': <VarType.DENSE_TENSOR: 7>, 'SELECTED_ROWS': <VarType.SELECTED_ROWS: 8>, 'FEED_MINIBATCH': <VarType.FEED_MINIBATCH: 9>, 'FETCH_LIST': <VarType.FETCH_LIST: 10>, 'STEP_SCOPES': <VarType.STEP_SCOPES: 11>, 'DENSE_TENSOR_ARRAY': <VarType.DENSE_TENSOR_ARRAY: 13>, 'PLACE_LIST': <VarType.PLACE_LIST: 14>, 'READER': <VarType.READER: 15>, 'RAW': <VarType.RAW: 17>, 'STRING': <VarType.STRING: 25>, 'STRINGS': <VarType.STRINGS: 26>, 'VOCAB': <VarType.VOCAB: 27>, 'SPARSE_COO': <VarType.SPARSE_COO: 30>}
        @staticmethod
        def __str__(dtype):
            ...
        @staticmethod
        def _pybind11_conduit_v1_(*args, **kwargs):
            ...
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: int) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    dist_attr: typing.Any
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: str) -> None:
        ...
    def _set_attr(self, arg0: str, arg1: typing.Any) -> None:
        ...
    def attr(self, arg0: str) -> typing.Any:
        ...
    def attr_names(self) -> list[str]:
        ...
    def clear_is_parameter(self) -> None:
        ...
    def clear_stop_gradient(self) -> None:
        ...
    def dtype(self) -> typing.Any:
        ...
    def dtypes(self) -> list[typing.Any]:
        ...
    def element_size(self) -> int:
        ...
    def get_shape(self) -> list[int]:
        ...
    def has_attr(self, arg0: str) -> bool:
        ...
    def has_is_parameter(self) -> bool:
        ...
    def has_stop_gradient(self) -> bool:
        ...
    def id(self) -> int:
        ...
    def is_parameter(self) -> bool:
        ...
    def legacy_lod_level(self) -> int:
        ...
    def legacy_lod_levels(self) -> list[int]:
        ...
    def lod_level(self) -> int:
        ...
    def lod_levels(self) -> list[int]:
        ...
    def name(self) -> str:
        ...
    def need_check_feed(self) -> bool:
        ...
    def original_id(self) -> int:
        ...
    def persistable(self) -> bool:
        ...
    def remove_attr(self, arg0: str) -> None:
        ...
    def serialize_to_string(self) -> bytes:
        ...
    def set_dtype(self, arg0: typing.Any) -> None:
        ...
    def set_dtypes(self, arg0: list[typing.Any]) -> None:
        ...
    def set_is_parameter(self, arg0: bool) -> None:
        ...
    def set_legacy_lod_level(self, arg0: int) -> None:
        ...
    def set_legacy_lod_levels(self, arg0: list[int]) -> None:
        ...
    def set_lod_level(self, arg0: int) -> None:
        ...
    def set_lod_levels(self, arg0: list[int]) -> None:
        ...
    def set_name(self, arg0: str) -> None:
        ...
    def set_need_check_feed(self, arg0: bool) -> None:
        ...
    def set_original_id(self, arg0: int) -> None:
        ...
    def set_persistable(self, arg0: bool) -> None:
        ...
    def set_shape(self, arg0: list[int]) -> None:
        ...
    def set_shapes(self, arg0: list[list[int]]) -> None:
        ...
    def set_stop_gradient(self, arg0: bool) -> None:
        ...
    def set_type(self, arg0: typing.Any) -> None:
        ...
    def shape(self) -> list[int]:
        ...
    def shapes(self) -> list[list[int]]:
        ...
    def stop_gradient(self) -> bool:
        ...
    def type(self) -> typing.Any:
        ...
class Variable:
    """
    Variable Class.
    
    All parameter, weight, gradient are variables in Paddle.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def get_bytes(self) -> bytes:
        ...
    def get_dense_tensor_array(self) -> typing.Any:
        ...
    def get_fetch_list(self) -> typing.Any:
        ...
    def get_float(self) -> float:
        ...
    def get_int(self) -> int:
        ...
    def get_map_tensor(self) -> typing.Any:
        ...
    def get_reader(self) -> typing.Any:
        ...
    def get_scope(self) -> typing.Any:
        ...
    def get_selected_rows(self) -> typing.Any:
        ...
    def get_string_tensor(self) -> typing.Any:
        ...
    def get_tensor(self) -> typing.Any:
        ...
    def is_float(self) -> bool:
        ...
    def is_int(self) -> bool:
        ...
    def set_float(self, arg0: float) -> None:
        ...
    def set_int(self, arg0: int) -> None:
        ...
    def set_scope(self, arg0: typing.Any) -> None:
        ...
    def set_string_list(self, arg0: list[str]) -> None:
        ...
    def set_vocab(self, arg0: dict[str, int]) -> None:
        ...
class XPUPlace:
    """
    
        Return a Baidu Kunlun Place
    
        Examples:
            .. code-block:: python
    
                >>> # doctest: +REQUIRES(env:XPU)
                >>> import paddle.base as base
                >>> xpu_place = base.XPUPlace(0)
            
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: int) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class XToRShrinkReshardFunction(ReshardFunction):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class XpuConfig:
    context: typing_extensions.CapsuleType
    context_gm_size: int
    conv_autotune_file: str
    conv_autotune_file_writeback: bool
    conv_autotune_level: int
    device_id: int
    fc_autotune_file: str
    fc_autotune_file_writeback: bool
    fc_autotune_level: int
    gemm_compute_precision: int
    l3_autotune_size: int
    l3_ptr: typing_extensions.CapsuleType
    l3_size: int
    quant_post_dynamic_activation_method: int
    quant_post_dynamic_op_types: list[str]
    quant_post_dynamic_weight_precision: int
    quant_post_static_gelu_out_threshold: float
    stream: typing_extensions.CapsuleType
    transformer_encoder_adaptive_seqlen: bool
    transformer_softmax_optimize_level: int
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
class ZeroCopyTensor:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def copy_from_cpu(self, arg0: numpy.ndarray[numpy.int8]) -> None:
        ...
    @typing.overload
    def copy_from_cpu(self, arg0: numpy.ndarray[numpy.uint8]) -> None:
        ...
    @typing.overload
    def copy_from_cpu(self, arg0: numpy.ndarray[numpy.int32]) -> None:
        ...
    @typing.overload
    def copy_from_cpu(self, arg0: numpy.ndarray[numpy.int64]) -> None:
        ...
    @typing.overload
    def copy_from_cpu(self, arg0: numpy.ndarray[numpy.float32]) -> None:
        ...
    @typing.overload
    def copy_from_cpu(self, arg0: numpy.ndarray[numpy.float16]) -> None:
        ...
    @typing.overload
    def copy_from_cpu(self, arg0: numpy.ndarray[numpy.float64]) -> None:
        ...
    @typing.overload
    def copy_from_cpu(self, arg0: numpy.ndarray[bool]) -> None:
        ...
    @typing.overload
    def copy_from_cpu(self, arg0: list[str]) -> None:
        ...
    def copy_to_cpu(self) -> numpy.ndarray:
        ...
    def lod(self) -> list[list[int]]:
        ...
    @typing.overload
    def reshape(self, arg0: list[int]) -> None:
        ...
    @typing.overload
    def reshape(self, arg0: int) -> None:
        ...
    def set_lod(self, arg0: list[list[int]]) -> None:
        ...
    def shape(self) -> list[int]:
        ...
    def type(self) -> PaddleDType:
        ...
class _Profiler:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @staticmethod
    def is_cnpapi_supported() -> bool:
        ...
    @staticmethod
    def is_cupti_supported() -> bool:
        ...
    @staticmethod
    def is_xpti_supported() -> bool:
        ...
    def create(self: typing.Any, arg0: list[str]) -> _Profiler:
        ...
    def prepare(self) -> None:
        ...
    def start(self) -> None:
        ...
    def stop(self) -> _ProfilerResult:
        ...
class _ProfilerResult:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def get_data(self) -> dict[int, typing.Any]:
        ...
    def get_extra_info(self) -> dict[str, str]:
        ...
    def get_span_indx(self) -> int:
        ...
    def get_version(self) -> str:
        ...
    def save(self, arg0: str, arg1: str) -> None:
        ...
class _RecordEvent:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: str, arg1: typing.Any) -> None:
        ...
    def end(self) -> None:
        ...
class _Scope:
    """
    
        Scope is an association of a name to Variable. All variables belong to Scope.
    
        Variables in a parent scope can be retrieved from local scope.
    
        You need to specify a scope to run a Net, i.e., `exe.Run(&scope)`.
        One net can run in different scopes and update different variable in the
        scope.
    
        You can create var in a scope and get it from the scope.
    
        Examples:
            .. code-block:: python
    
                >>> import paddle
                >>> import numpy as np
    
                >>> scope = paddle.static.global_scope()
                >>> place = paddle.CPUPlace()
                >>> # create tensor from a scope and set value to it.
                >>> param = scope.var('Param').get_tensor()
                >>> param_array = np.full((10, 12), 5.0).astype("float32")
                >>> param.set(param_array, place)
            
    """
    _can_reused: bool
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def _kids(self) -> list[_Scope]:
        ...
    def _remove_from_pool(self) -> None:
        ...
    def drop_kids(self) -> None:
        """
                   Delete all sub-scopes of the current scope.
        """
    def erase(self, names: list[str]) -> None:
        """
                   Find variable named :code:`name` in the current scope or
                   its parent scope. Return None if not found.
        
                   Args:
                       name (str): the variable names to be erase.
        
                   Returns:
                       None
        """
    def find_var(self, name: str) -> Variable:
        """
                   Find variable named :code:`name` in the current scope or
                   its parent scope. Return None if not found.
        
                   Args:
                       name (str): the variable name.
        
                   Returns:
                       out (core.Variable|None): the found variable or None.
        """
    def local_var_names(self) -> list[str]:
        """
                  Get all variable names in the current scope.
        
                  Returns:
                      List[str]: The list of variable names.
        """
    def new_scope(self) -> _Scope:
        """
                   Create a new sub-scope of the current scope.
        
                   Returns:
                       out (core._Scope): the created sub-scope.
        """
    def raw_address(self) -> int:
        ...
    def size(self) -> int:
        ...
    def var(self, name: str) -> Variable:
        """
                   Find or create variable named :code:`name` in the current scope.
        
                   If the variable named :code:`name` does not exist in the
                   current scope, the variable would be created. Otherwise,
                   return the existing variable.
        
                   Args:
                       name (str): the variable name.
        
                   Returns:
                       out (core.Variable): the found or created variable.
        """
class finfo:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: typing.Any) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def bits(self) -> int:
        ...
    @property
    def dtype(self) -> str:
        ...
    @property
    def eps(self) -> float:
        ...
    @property
    def max(self) -> float:
        ...
    @property
    def min(self) -> float:
        ...
    @property
    def resolution(self) -> float:
        ...
    @property
    def smallest_normal(self) -> float:
        ...
    @property
    def tiny(self) -> float:
        ...
class iinfo:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: typing.Any) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def bits(self) -> int:
        ...
    @property
    def dtype(self) -> str:
        ...
    @property
    def max(self) -> int:
        ...
    @property
    def min(self) -> int:
        ...
class mt19937_64:
    """
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
class task:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def is_completed(self) -> bool:
        ...
    def is_sync(self) -> bool:
        ...
    def synchronize(self) -> None:
        ...
    def wait(self, timeout: datetime.timedelta = ...) -> bool:
        ...
@typing.overload
def Load(arg0: str, arg1: typing.Any) -> Layer:
    ...
@typing.overload
def Load(arg0: str, arg1: typing.Any) -> Layer:
    ...
def Scope() -> _Scope:
    """
            Create a new scope.
    
            Returns:
                out (core._Scope): the created scope.
    """
def __set_all_prim_enabled(arg0: bool) -> None:
    ...
def __set_bwd_prim_enabled(arg0: bool) -> None:
    ...
def __set_eager_prim_enabled(arg0: bool) -> None:
    ...
def __set_fwd_prim_enabled(arg0: bool) -> None:
    ...
def __unittest_throw_exception__() -> None:
    ...
def _add_skip_comp_ops(arg0: str) -> None:
    ...
def _append_python_callable_object_and_return_id(arg0: typing.Any) -> int:
    ...
def _create_loaded_parameter(arg0: typing.Any, arg1: typing.Any, arg2: typing.Any) -> None:
    ...
def _cuda_synchronize(arg0: typing.Any) -> None:
    ...
@typing.overload
def _device_synchronize(arg0: int) -> None:
    ...
@typing.overload
def _device_synchronize(arg0: int) -> None:
    ...
def _dygraph_debug_level() -> int:
    ...
def _get_all_register_op_kernels(lib: str = 'all') -> dict[str, list[str]]:
    """
               Return the registered kernels in paddle.
    
               Args:
                   lib[string]: the library, could be 'phi', 'fluid' and 'all'.
    """
def _get_amp_attrs() -> typing.Any:
    ...
def _get_amp_op_list() -> tuple[set[str], set[str]]:
    ...
def _get_current_custom_device_stream(device_type: str, device_id: int = -1) -> None:
    ...
@typing.overload
def _get_current_stream(arg0: int) -> None:
    ...
@typing.overload
def _get_current_stream(arg0: int) -> None:
    ...
def _get_device_min_chunk_size(arg0: str) -> int:
    ...
def _get_device_total_memory(device_type: str, device_id: int = -1) -> int:
    ...
def _get_eager_deletion_vars(arg0: typing.Any, arg1: list[str]) -> list[list[list[str]]]:
    ...
def _get_phi_kernel_name(arg0: str) -> str:
    ...
def _get_registered_phi_kernels(kernel_registered_type: str = 'function') -> dict[str, list[str]]:
    """
               Return the registered kernels in phi.
    
               Args:
                   kernel_registered_type[string]: the library, could be 'function', 'structure', and 'all'.
    """
def _get_use_default_grad_op_desc_maker_ops() -> list[str]:
    ...
def _has_grad() -> bool:
    ...
def _is_bwd_prim_enabled() -> bool:
    ...
def _is_compiled_with_heterps() -> bool:
    ...
def _is_dygraph_debug_enabled() -> bool:
    ...
def _is_eager_prim_enabled() -> bool:
    ...
def _is_fwd_prim_enabled() -> bool:
    ...
def _is_program_version_supported(arg0: int) -> bool:
    ...
def _promote_types_if_complex_exists(arg0: typing.Any, arg1: typing.Any) -> typing.Any:
    ...
def _remove_skip_comp_ops(arg0: str) -> None:
    ...
def _set_amp_op_list(arg0: set[str], arg1: set[str]) -> None:
    ...
def _set_bwd_prim_blacklist(arg0: set[str]) -> None:
    ...
def _set_current_custom_device_stream(device_type: str, device_id: int = -1, stream: typing.Any = None) -> None:
    ...
def _set_current_stream(arg0: typing.Any) -> None:
    ...
def _set_eager_deletion_mode(arg0: float, arg1: float, arg2: bool) -> None:
    ...
def _set_eager_tracer(arg0: typing.Any) -> None:
    ...
def _set_fuse_parameter_group_size(arg0: int) -> None:
    ...
def _set_fuse_parameter_memory_size(arg0: float) -> None:
    ...
def _set_has_grad(arg0: bool) -> None:
    ...
def _set_paddle_lib_path(arg0: str) -> None:
    ...
def _set_prim_target_grad_name(arg0: dict[str, str]) -> None:
    ...
def _set_warmup(arg0: bool) -> None:
    ...
def _switch_tracer(arg0: typing.Any) -> None:
    ...
def _synchronize_custom_device(arg0: str, arg1: int) -> None:
    ...
def _test_enforce_gpu_success() -> None:
    ...
def _xpu_device_synchronize(arg0: int) -> None:
    ...
def apply_pass(arg0: ProgramDesc, arg1: ProgramDesc, arg2: typing.Any, arg3: dict[str, typing.Any], arg4: dict[str, str]) -> dict[str, typing.Any]:
    ...
def autotune_status() -> dict:
    ...
def broadcast_shape(arg0: list[int], arg1: list[int]) -> list[int]:
    ...
def build_adjacency_list(arg0: typing.Any) -> dict[typing.Any, set[typing.Any]]:
    ...
def calc_place_hash(*args, **kwargs):
    """
    Calculate the hash value by tensors place.
    """
def call_decomp_rule(arg0: pir.Operation) -> list:
    ...
def call_decomp_vjp(arg0: pir.Operation) -> list:
    ...
def call_vjp(arg0: pir.Operation, arg1: list[list[pir.Value]], arg2: list[list[pir.Value]], arg3: list[list[pir.Value]], arg4: list[list[bool]]) -> list:
    ...
def clear_device_manager() -> None:
    ...
def clear_executor_cache() -> None:
    ...
def clear_gradients(arg0: list[typing.Any], arg1: bool) -> None:
    ...
def clear_kernel_factory() -> None:
    ...
def clear_low_precision_op_list() -> None:
    ...
def clear_shape_info() -> None:
    ...
def contains_spmd_rule(arg0: str) -> bool:
    ...
def convert_to_mixed_precision_bind(model_file: str, params_file: str, mixed_model_file: str, mixed_params_file: str, mixed_precision: AnalysisConfig.Precision, backend: PaddlePlace, keep_io_types: bool = True, black_list: set[str] = set(), white_list: set[str] = set()) -> None:
    ...
def copy_tensor(arg0: PaddleInferTensor, arg1: PaddleInferTensor) -> None:
    ...
def create_empty_tensors_with_values(*args, **kwargs):
    """
    GetEmptyTensorsWithValue.
    """
def create_empty_tensors_with_var_descs(*args, **kwargs):
    """
    GetEmptyTensorsWithVarDesc
    """
def create_or_get_global_tcp_store() -> Store:
    ...
@typing.overload
def create_paddle_predictor(config: AnalysisConfig) -> PaddlePredictor:
    ...
@typing.overload
def create_paddle_predictor(config: NativeConfig) -> PaddlePredictor:
    ...
def create_predictor(arg0: AnalysisConfig) -> PaddleInferPredictor:
    ...
@typing.overload
def create_py_reader(arg0: DenseTensorBlockingQueue, arg1: list[str], arg2: list[list[int]], arg3: list[typing.Any], arg4: list[bool], arg5: list[typing.Any], arg6: bool, arg7: bool, arg8: bool) -> MultiDeviceFeedReader:
    ...
@typing.overload
def create_py_reader(arg0: OrderedMultiDeviceDenseTensorBlockingQueue, arg1: list[str], arg2: list[list[int]], arg3: list[typing.Any], arg4: list[bool], arg5: list[typing.Any], arg6: bool, arg7: bool, arg8: bool) -> OrderedMultiDeviceFeedReader:
    ...
def default_cpu_generator() -> Generator:
    ...
def default_cuda_generator(arg0: int) -> Generator:
    ...
def default_custom_device_generator(arg0: CustomPlace) -> Generator:
    ...
def default_xpu_generator(arg0: int) -> Generator:
    ...
def deserialize_pir_program(file_path: str, program: typing.Any, pir_version: int = -1) -> bool:
    ...
def device_memory_stat_current_value(arg0: str, arg1: int) -> int:
    ...
def device_memory_stat_peak_value(arg0: str, arg1: int) -> int:
    ...
def device_memory_stat_reset_peak_value(arg0: str, arg1: int) -> None:
    ...
@typing.overload
def diff_tensor_shape(arg0: typing.Any, arg1: typing.Any, arg2: int) -> typing.Any:
    ...
@typing.overload
def diff_tensor_shape(arg0: typing.Any, arg1: list[int], arg2: int) -> typing.Any:
    ...
def disable_autotune() -> None:
    ...
def disable_layout_autotune() -> None:
    ...
def disable_memory_recorder() -> None:
    ...
def disable_op_info_recorder() -> None:
    ...
def disable_profiler(arg0: EventSortingKey, arg1: str) -> None:
    ...
def disable_signal_handler() -> None:
    ...
def dtensor_from_local(arg0: typing.Any, arg1: typing.Any, arg2: typing.Any) -> typing.Any:
    ...
def dtensor_to_local(arg0: typing.Any, arg1: typing.Any, arg2: typing.Any) -> typing.Any:
    ...
def dygraph_partial_grad(arg0: list[typing.Any], arg1: list[typing.Any], arg2: list[typing.Any], arg3: list[typing.Any], arg4: typing.Any, arg5: bool, arg6: bool, arg7: bool, arg8: bool) -> list[typing.Any]:
    ...
def dygraph_run_backward(arg0: list[typing.Any], arg1: list[typing.Any], arg2: bool, arg3: Tracer) -> None:
    ...
def eager_assign_group_by_size(tensors: typing.Any, is_sparse_gradient: list[bool], group_size_limits: list[int] = [26214400], tensor_indices: list[int] = []) -> list[list[int]]:
    ...
def empty_var_name() -> str:
    ...
def enable_autotune() -> None:
    ...
def enable_layout_autotune() -> None:
    ...
def enable_memory_recorder() -> None:
    ...
def enable_op_info_recorder() -> None:
    ...
def enable_profiler(arg0: ProfilerState) -> None:
    ...
def eval_frame_no_skip_codes(py_codes: typing.Any) -> typing.Any:
    ...
def eval_frame_skip_file_prefix(py_codes: typing.Any) -> typing.Any:
    ...
def from_dlpack(arg0: typing.Any) -> typing.Any:
    ...
def get_all_custom_device_type() -> list[str]:
    ...
def get_all_device_type() -> list[str]:
    ...
def get_all_op_names(lib: str = 'all') -> list[str]:
    """
          Return the operator names in paddle.
    
          Args:
              lib[string]: the library contains corresponding OpKernel, could be 'phi', 'fluid' and 'all'. Default value is 'all'.
    """
def get_all_op_protos() -> list[bytes]:
    ...
def get_attribute_type(arg0: str, arg1: str) -> typing.Any:
    ...
def get_available_custom_device() -> list[str]:
    ...
def get_available_device() -> list[str]:
    ...
def get_custom_device_count(arg0: str) -> int:
    ...
def get_fetch_variable(arg0: _Scope, arg1: str, arg2: int) -> typing.Any:
    ...
def get_float_stats() -> dict[str, float]:
    ...
def get_grad_op_desc(arg0: typing.Any, arg1: set[str], arg2: list[typing.Any]) -> tuple[list[typing.Any], dict[str, str]]:
    ...
def get_int_stats() -> dict[str, int]:
    ...
def get_low_precision_op_list() -> dict:
    ...
def get_no_need_buffer_values(arg0: dict[str, typing.Any]) -> dict[str, set[str]]:
    ...
def get_num_bytes_of_data_type(arg0: PaddleDType) -> int:
    ...
def get_op_attrs_default_value(arg0: bytes) -> dict[str, typing.Any]:
    ...
def get_op_extra_attrs(arg0: str) -> dict[str, typing.Any]:
    ...
def get_op_version_map() -> dict[str, OpVersion]:
    ...
def get_pass(arg0: str) -> typing.Any:
    ...
def get_phi_spmd_rule(arg0: str) -> SpmdRule:
    ...
def get_promote_dtype_old_ir(arg0: str, arg1: typing.Any, arg2: typing.Any) -> typing.Any:
    ...
def get_random_seed_generator(arg0: str) -> Generator:
    ...
def get_trt_compile_version() -> tuple[int, int, int]:
    ...
def get_trt_runtime_version() -> tuple[int, int, int]:
    ...
def get_value_shape_range_info(arg0: typing.Any, arg1: bool, arg2: ShapeMode) -> list:
    ...
def get_variable_tensor(arg0: _Scope, arg1: str) -> typing.Any:
    ...
def get_version() -> str:
    ...
def globals() -> GlobalVarGetterSetterRegistry:
    ...
def grad_var_suffix() -> str:
    ...
def graph_num(arg0: typing.Any) -> int:
    ...
def graph_safe_remove_nodes(arg0: typing.Any, arg1: set[typing.Any]) -> None:
    ...
def has_circle(arg0: typing.Any) -> bool:
    ...
def has_comp_grad_op_maker(arg0: str) -> bool:
    ...
def has_custom_getattro(arg0: typing.Any) -> bool:
    ...
def has_custom_vjp(arg0: pir.Operation) -> bool:
    """
               Return whether an op has custom vjp rules.
    
               Args:
                   op (pir::Operation): op to be checked
    
               Returns:
                   out (bool): True means that the op has custom vjp rules, False means it does not.
    """
def has_decomp_rule(arg0: pir.Operation) -> bool:
    ...
def has_decomp_vjp(arg0: pir.Operation) -> bool:
    ...
def has_empty_grad_op_maker(arg0: str) -> bool:
    ...
def has_grad_op_maker(arg0: str) -> bool:
    ...
def has_infer_inplace(arg0: str) -> bool:
    ...
def has_non_empty_grad_op_maker(arg0: str) -> bool:
    ...
def has_vjp(arg0: pir.Operation) -> bool:
    ...
def host_memory_stat_current_value(arg0: str, arg1: int) -> int:
    ...
def host_memory_stat_peak_value(arg0: str, arg1: int) -> int:
    ...
def host_memory_stat_reset_peak_value(arg0: str, arg1: int) -> None:
    ...
def infer_no_need_buffer_slots(arg0: str, arg1: dict[str, list[str]], arg2: dict[str, list[str]], arg3: dict[str, typing.Any]) -> set[str]:
    ...
def init_default_kernel_signatures() -> None:
    ...
def init_dense_tensor_blocking_queue(arg0: Variable, arg1: int, arg2: bool) -> typing.Any:
    ...
def init_devices() -> None:
    ...
def init_gflags(arg0: list[str]) -> bool:
    ...
def init_glog(arg0: str) -> None:
    ...
def init_memory_method() -> None:
    ...
def init_tensor_operants() -> None:
    ...
def is_bfloat16_supported(arg0: CPUPlace) -> bool:
    ...
def is_common_dtype_for_scalar(arg0: typing.Any, arg1: typing.Any) -> bool:
    ...
def is_compiled_with_avx() -> bool:
    ...
def is_compiled_with_brpc() -> bool:
    ...
def is_compiled_with_cinn() -> bool:
    ...
def is_compiled_with_cuda() -> bool:
    ...
def is_compiled_with_cudnn_frontend() -> bool:
    ...
def is_compiled_with_custom_device(arg0: str) -> bool:
    ...
def is_compiled_with_dist() -> bool:
    ...
def is_compiled_with_distribute() -> bool:
    ...
def is_compiled_with_flagcx() -> bool:
    ...
def is_compiled_with_ipu() -> bool:
    ...
def is_compiled_with_mkldnn() -> bool:
    ...
def is_compiled_with_mpi() -> bool:
    ...
def is_compiled_with_mpi_aware() -> bool:
    ...
def is_compiled_with_nccl() -> bool:
    ...
def is_compiled_with_rocm() -> bool:
    ...
def is_compiled_with_xpu() -> bool:
    ...
def is_cuda_graph_capturing() -> bool:
    ...
def is_float16_supported(arg0: CPUPlace) -> bool:
    ...
def is_forward_only(arg0: pir.Operation) -> bool:
    """
               Return whether an op is forward only op.
    
               Args:
                   op (pir::Operation): op to be checked
    
               Returns:
                   out (bool): True means that the op is forward only op, False means it does not.
    """
def is_profiler_enabled() -> bool:
    ...
def kAutoParallelSuffix() -> str:
    ...
def kControlDepVarName() -> str:
    ...
def kEmptyVarName() -> str:
    ...
def kGradVarSuffix() -> str:
    ...
def kNewGradSuffix() -> str:
    ...
def kNoneProcessMeshIndex() -> int:
    ...
def kTempVarName() -> str:
    ...
def kZeroVarSuffix() -> str:
    ...
@typing.overload
def load_combine_func(arg0: str, arg1: list[str], arg2: list[typing.Any], arg3: bool, arg4: typing.Any) -> None:
    ...
@typing.overload
def load_combine_func(arg0: str, arg1: list[str], arg2: list[typing.Any], arg3: bool, arg4: typing.Any) -> None:
    ...
@typing.overload
def load_combine_func(arg0: str, arg1: list[str], arg2: list[typing.Any], arg3: bool, arg4: typing.Any) -> None:
    ...
@typing.overload
def load_combine_func(arg0: str, arg1: list[str], arg2: list[typing.Any], arg3: bool, arg4: typing.Any) -> None:
    ...
@typing.overload
def load_combine_func(arg0: str, arg1: list[str], arg2: list[typing.Any], arg3: bool, arg4: typing.Any) -> None:
    ...
@typing.overload
def load_combine_func(arg0: str, arg1: list[str], arg2: list[typing.Any], arg3: bool, arg4: typing.Any) -> None:
    ...
@typing.overload
def load_combine_func(arg0: str, arg1: list[str], arg2: list[typing.Any], arg3: bool, arg4: typing.Any) -> None:
    ...
@typing.overload
def load_dense_tensor(arg0: typing.Any, arg1: str) -> int:
    ...
@typing.overload
def load_dense_tensor(arg0: str) -> typing.Any:
    ...
def load_dense_tensor_from_memory(arg0: typing.Any, arg1: str) -> None:
    ...
@typing.overload
def load_func(arg0: str, arg1: int, arg2: list[int], arg3: bool, arg4: typing.Any, arg5: typing.Any) -> None:
    ...
@typing.overload
def load_func(arg0: str, arg1: int, arg2: list[int], arg3: bool, arg4: typing.Any, arg5: typing.Any) -> None:
    ...
@typing.overload
def load_func(arg0: str, arg1: int, arg2: list[int], arg3: bool, arg4: typing.Any, arg5: typing.Any) -> None:
    ...
@typing.overload
def load_func(arg0: str, arg1: int, arg2: list[int], arg3: bool, arg4: typing.Any, arg5: typing.Any) -> None:
    ...
@typing.overload
def load_func(arg0: str, arg1: int, arg2: list[int], arg3: bool, arg4: typing.Any, arg5: typing.Any) -> None:
    ...
@typing.overload
def load_func(arg0: str, arg1: int, arg2: list[int], arg3: bool, arg4: typing.Any, arg5: typing.Any) -> None:
    ...
@typing.overload
def load_func(arg0: str, arg1: int, arg2: list[int], arg3: bool, arg4: typing.Any, arg5: typing.Any) -> None:
    ...
def load_op_meta_info_and_register_op(arg0: str) -> None:
    ...
def load_profiler_result(arg0: str) -> _ProfilerResult:
    ...
def load_selected_rows(arg0: typing.Any, arg1: str) -> int:
    ...
def load_selected_rows_from_memory(arg0: typing.Any, arg1: str) -> None:
    ...
def merge_guard(py_guards: list[GuardBase]) -> GuardGroup:
    ...
def need_type_promotion_old_ir(arg0: str, arg1: typing.Any, arg2: typing.Any) -> bool:
    ...
def op_support_gpu(arg0: str) -> bool:
    ...
def op_supported_infos(arg0: str, arg1: typing.Any) -> tuple[set[str], set[str], set[str]]:
    ...
def paddle_dtype_size(arg0: PaddleDType) -> int:
    ...
def paddle_tensor_to_bytes(arg0: PaddleTensor) -> bytes:
    ...
def parse_safe_eager_deletion_skip_vars(arg0: ProgramDesc, arg1: bool) -> set[str]:
    ...
def prune(arg0: typing.Any, arg1: set[str], arg2: list[typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(2)]]) -> tuple[typing.Any, dict[int, int]]:
    ...
def prune_backward(arg0: typing.Any) -> tuple[typing.Any, dict[int, int]]:
    """
                 Prune the backward part of a program, mostly called in
                 program.clone(for_test=True).
    
                Args:
                       program (ProgramDesc): The original program.
    
                 Returns:
                       tuple(ProgramDesc, map<int, int>): The first part is
                       the pruned program desc, and the second part is a map
                       which contains the id pair of pruned block and corresponding
                       origin block.
    """
def register_pass(arg0: str, arg1: typing.Any) -> None:
    ...
def register_subgraph_pass(arg0: str) -> None:
    ...
def reset_profiler() -> None:
    ...
def reshard(arg0: typing.Any, arg1: TensorDistAttr) -> typing.Any:
    ...
def run_cmd(cmd: str, time_out: int = -1, sleep_inter: int = -1) -> str:
    ...
def save_combine_func(arg0: list[typing.Any], arg1: list[str], arg2: str, arg3: bool, arg4: bool, arg5: bool) -> None:
    ...
def save_dense_tensor(arg0: typing.Any, arg1: str) -> int:
    ...
def save_dense_tensor_to_memory(arg0: typing.Any) -> bytes:
    ...
def save_func(arg0: typing.Any, arg1: str, arg2: str, arg3: bool, arg4: bool) -> None:
    ...
def save_op_version_info(arg0: typing.Any) -> None:
    ...
def save_selected_rows(arg0: typing.Any, arg1: str) -> int:
    ...
def save_selected_rows_to_memory(arg0: typing.Any) -> bytes:
    ...
def serialize_pir_program(program: typing.Any, file_path: str, pir_version: int, overwrite: bool = True, readable: bool = False, trainable: bool = True) -> None:
    ...
def set_autotune_range(arg0: int, arg1: int) -> None:
    ...
def set_checked_op_list(arg0: str) -> None:
    ...
def set_current_thread_name(arg0: str) -> bool:
    ...
def set_eval_frame(callback: typing.Any) -> typing.Any:
    ...
def set_feed_variable(arg0: _Scope, arg1: typing.Any, arg2: str, arg3: int) -> None:
    ...
def set_nan_inf_debug_path(arg0: str) -> None:
    ...
def set_nan_inf_stack_limit(arg0: int) -> None:
    ...
def set_num_threads(arg0: int) -> None:
    ...
def set_printoptions(**kwargs) -> None:
    ...
def set_random_seed_generator(arg0: str, arg1: int) -> Generator:
    ...
def set_skipped_op_list(arg0: str) -> None:
    ...
def set_static_op_arg_pre_cast_hook(*args, **kwargs):
    """
    Set hook for pre cast a static OP argument.
    """
def set_tracer_option(arg0: TracerOption) -> None:
    ...
def set_variable(arg0: _Scope, arg1: typing.Any, arg2: str) -> None:
    ...
def shell_execute_cmd(cmd: str, time_out: int = 0, sleep_inter: int = 0, redirect_stderr: bool = False) -> list[str]:
    ...
def sinking_decomp(arg0: pir.Program, arg1: list[pir.Value], arg2: set[str], arg3: set[str], arg4: int, arg5: int) -> list:
    ...
@typing.overload
def size_of_dtype(arg0: VarDesc.VarType) -> int:
    ...
@typing.overload
def size_of_dtype(arg0: typing.Any) -> int:
    ...
def sot_set_with_graph(py_codes: typing.Any) -> typing.Any:
    ...
def sot_setup_codes_with_graph(py_codes: typing.Any) -> typing.Any:
    ...
def start_imperative_gperf_profiler() -> None:
    ...
def stop_imperative_gperf_profiler() -> None:
    ...
def sub_mesh_dim(arg0: ProcessMesh, arg1: ProcessMesh) -> int:
    ...
def supports_avx512f() -> bool:
    ...
def supports_bfloat16() -> bool:
    ...
def supports_bfloat16_fast_performance() -> bool:
    ...
def supports_int8() -> bool:
    ...
def supports_vnni() -> bool:
    ...
def tensor_from_cuda_array_interface(arg0: typing.Any) -> typing.Any:
    ...
def topology_sort(arg0: typing.Any) -> list[typing.Any]:
    ...
def touch_dist_mapper() -> str:
    ...
def tracer_event_type_to_string(arg0: TracerEventType) -> str:
    ...
def tracer_mem_event_type_to_string(arg0: TracerMemEventType) -> str:
    ...
def update_autotune_status() -> None:
    ...
def use_layout_autotune() -> bool:
    ...
@typing.overload
def varbase_copy(arg0: typing.Any, arg1: typing.Any, arg2: typing.Any, arg3: bool) -> None:
    ...
@typing.overload
def varbase_copy(arg0: typing.Any, arg1: typing.Any, arg2: typing.Any, arg3: bool) -> None:
    ...
@typing.overload
def varbase_copy(arg0: typing.Any, arg1: typing.Any, arg2: typing.Any, arg3: bool) -> None:
    ...
@typing.overload
def varbase_copy(arg0: typing.Any, arg1: typing.Any, arg2: typing.Any, arg3: bool) -> None:
    ...
@typing.overload
def varbase_copy(arg0: typing.Any, arg1: typing.Any, arg2: typing.Any, arg3: bool) -> None:
    ...
@typing.overload
def varbase_copy(arg0: typing.Any, arg1: typing.Any, arg2: typing.Any, arg3: bool) -> None:
    ...
def wait_device(arg0: typing.Any) -> None:
    ...
ALL_DTYPE: paddle.dtype  # value = <DataType.UNDEFINED: 0>
BFLOAT16: paddle.dtype  # value = <DataType.BFLOAT16: 16>
BOOL: paddle.dtype  # value = <DataType.BOOL: 1>
COMPLEX128: paddle.dtype  # value = <DataType.COMPLEX128: 13>
COMPLEX64: paddle.dtype  # value = <DataType.COMPLEX64: 12>
FLOAT16: paddle.dtype  # value = <DataType.FLOAT16: 15>
FLOAT32: paddle.dtype  # value = <DataType.FLOAT32: 10>
FLOAT64: paddle.dtype  # value = <DataType.FLOAT64: 11>
FLOAT8_E4M3FN: paddle.dtype  # value = <DataType.FLOAT8_E4M3FN: 17>
FLOAT8_E5M2: paddle.dtype  # value = <DataType.FLOAT8_E5M2: 18>
INT16: paddle.dtype  # value = <DataType.INT16: 5>
INT32: paddle.dtype  # value = <DataType.INT32: 7>
INT64: paddle.dtype  # value = <DataType.INT64: 9>
INT8: paddle.dtype  # value = <DataType.INT8: 3>
O0: AmpLevel  # value = <AmpLevel.O0: 0>
O1: AmpLevel  # value = <AmpLevel.O1: 1>
O2: AmpLevel  # value = <AmpLevel.O2: 2>
O3: AmpLevel  # value = <AmpLevel.O3: 3>
OD: AmpLevel  # value = <AmpLevel.OD: 4>
PSTRING: paddle.dtype  # value = <DataType.PSTRING: 14>
UINT16: paddle.dtype  # value = <DataType.UINT16: 4>
UINT32: paddle.dtype  # value = <DataType.UINT32: 6>
UINT64: paddle.dtype  # value = <DataType.UINT64: 8>
UINT8: paddle.dtype  # value = <DataType.UINT8: 2>
UNDEFINED: paddle.dtype  # value = <DataType.UNDEFINED: 0>
_cleanup: typing.Any  # value = <capsule object>
kAll: ProfilerState  # value = <ProfilerState.kAll: 3>
kAllOpDetail: TracerOption  # value = <TracerOption.kAllOpDetail: 2>
kAve: EventSortingKey  # value = <EventSortingKey.kAve: 5>
kCPU: ProfilerState  # value = <ProfilerState.kCPU: 1>
kCUDA: ProfilerState  # value = <ProfilerState.kCUDA: 2>
kCalls: EventSortingKey  # value = <EventSortingKey.kCalls: 1>
kDefault: EventSortingKey  # value = <EventSortingKey.kDefault: 0>
kDisabled: ProfilerState  # value = <ProfilerState.kDisabled: 0>
kMAX: ShapeMode  # value = <ShapeMode.kMAX: 2>
kMIN: ShapeMode  # value = <ShapeMode.kMIN: 0>
kMax: EventSortingKey  # value = <EventSortingKey.kMax: 4>
kMin: EventSortingKey  # value = <EventSortingKey.kMin: 3>
kOPT: ShapeMode  # value = <ShapeMode.kOPT: 1>
kOpDetail: TracerOption  # value = <TracerOption.kOpDetail: 1>
kTotal: EventSortingKey  # value = <EventSortingKey.kTotal: 2>
LoDTensor = DenseTensor
Tensor = DenseTensor
