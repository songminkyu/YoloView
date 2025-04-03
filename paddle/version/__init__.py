# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
full_version     = '3.0.0'
major            = '3'
minor            = '0'
patch            = '0'
nccl_version     = '0'
rc               = '0'
cuda_version     = 'False'
cudnn_version    = 'False'
xpu_xre_version  = 'False'
xpu_xccl_version = 'False'
xpu_xhpc_version = 'False'
is_tagged        = True
commit           = '6ed5dd3833c32c3b21e14b1fb1a71f5a535a0fcc'
with_mkl         = 'ON'
cinn_version     = 'False'
tensorrt_version = 'None'
with_pip_cuda_libraries = 'ON'
with_pip_tensorrt       ='OFF'
compiled_cuda_archs     = []

__all__ = ['cuda', 'cudnn', 'nccl', 'show', 'xpu', 'xpu_xre', 'xpu_xccl', 'xpu_xhpc', 'tensorrt', 'cuda_archs']

def show() -> None:
    """Get the version of paddle if `paddle` package if tagged. Otherwise, output the corresponding commit id.

    Returns:
        If paddle package is not tagged, the commit-id of paddle will be output.
        Otherwise, the following information will be output.

        full_version: version of paddle

        major: the major version of paddle

        minor: the minor version of paddle

        patch: the patch level version of paddle

        rc: whether it's rc version

        cuda: the cuda version of package. It will return `False` if CPU version paddle package is installed

        cudnn: the cudnn version of package. It will return `False` if CPU version paddle package is installed

        xpu_xre: the xpu xre version of package. It will return `False` if non-XPU version paddle package is installed

        xpu_xccl: the xpu xccl version of package. It will return `False` if non-XPU version paddle package is installed

        xpu_xhpc: the xpu xhpc version of package. It will return `False` if non-XPU version paddle package is installed

        cinn: the cinn version of package. It will return `False` if paddle package is not compiled with CINN

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Case 1: paddle is tagged with 2.2.0
            >>> paddle.version.show()
            >>> # doctest: +SKIP('Different environments yield different output.')
            full_version: 2.2.0
            major: 2
            minor: 2
            patch: 0
            rc: 0
            cuda: '10.2'
            cudnn: '7.6.5'
            xpu_xre: '4.32.0.1'
            xpu_xccl: '1.0.7'
            xpu_xhpc: '20231208'
            cinn: False
            >>> # doctest: -SKIP

            >>> # Case 2: paddle is not tagged
            >>> paddle.version.show()
            >>> # doctest: +SKIP('Different environments yield different output.')
            commit: cfa357e984bfd2ffa16820e354020529df434f7d
            cuda: '10.2'
            cudnn: '7.6.5'
            xpu_xre: '4.32.0.1'
            xpu_xccl: '1.0.7'
            xpu_xhpc: '20231208'
            cinn: False
            >>> # doctest: -SKIP

    """
    if is_tagged:
        print('full_version:', full_version)
        print('major:', major)
        print('minor:', minor)
        print('patch:', patch)
        print('rc:', rc)
    else:
        print('commit:', commit)
    print('cuda:', cuda_version)
    print('cudnn:', cudnn_version)
    print('nccl:', nccl_version)
    print('xpu_xre:', xpu_xre_version)
    print('xpu_xccl:', xpu_xccl_version)
    print('xpu_xhpc:', xpu_xhpc_version)
    print('cinn:', cinn_version)
    print('tensorrt:', tensorrt_version)
    print('cuda_archs:', compiled_cuda_archs)

def mkl() -> str:
    return with_mkl

def nccl() -> str:
    """Get nccl version of paddle package.

    Returns:
        string: Return the version information of cuda nccl. If paddle package is CPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.nccl()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '2804'

    """
    return nccl_version

def cuda() -> str:
    """Get cuda version of paddle package.

    Returns:
        string: Return the version information of cuda. If paddle package is CPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.cuda()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '10.2'

    """
    return cuda_version

def cudnn() -> str:
    """Get cudnn version of paddle package.

    Returns:
        string: Return the version information of cudnn. If paddle package is CPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.cudnn()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '7.6.5'

    """
    return cudnn_version

def xpu() -> str:
    """Get xpu version of paddle package. The API is deprecated now, please use xpu_xhpc() instead.

    Returns:
        string: Return the version information of xpu. If paddle package is non-XPU version, it will return False.
    Examples:
        .. code-block:: python
            >>> import paddle
            >>> paddle.version.xpu()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '20230114'
    """
    return xpu_xhpc_version

def xpu_xre() -> str:
    """Get xpu xre version of paddle package.

    Returns:
        string: Return the version information of xpu. If paddle package is non-XPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.xpu_xre()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '4.32.0.1'

    """
    return xpu_xre_version

def xpu_xccl() -> str:
    """Get xpu xccl version of paddle package.

    Returns:
        string: Return the version information of xpu xccl. If paddle package is non-XPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.xpu_xccl()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '1.0.7'

    """
    return xpu_xccl_version

def xpu_xhpc() -> str:
    """Get xpu xhpc version of paddle package.

    Returns:
        string: Return the version information of xpu xhpc. If paddle package is non-XPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.xpu_xhpc()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '20231208'

    """
    return xpu_xhpc_version

def cinn() -> str:
    """Get CINN version of paddle package.

    Returns:
        string: Return the version information of CINN. If paddle package is not compiled with CINN, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.cinn()
            >>> # doctest: +SKIP('Different environments yield different output.')
            False

    """
    return cinn_version

def tensorrt() -> str:
    """Get TensorRT version of paddle package.

    Returns:
        string: Return the version information of TensorRT. If paddle package is not compiled with TensorRT, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.tensorrt()
            >>> # doctest: +SKIP('Different environments yield different output.')
            False

    """
    return tensorrt_version

def cuda_archs():
    """Get compiled cuda archs of paddle package.

    Returns:
        list[int]: Return the compiled cuda archs if with gpu. If paddle package is not compiled with gpu, it will return "".

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.cuda_archs()
            >>> # doctest: +SKIP('Different environments yield different output.')
            [86]

    """
    return compiled_cuda_archs
