"""Record and check the numerical environment before resuming cached stages."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from PIL import Image, features, __version__ as pillow_version
import torch
import torchvision


def execution_environment(device):
    device = torch.device(device)
    core = Path(Image.core.__file__)
    decoder = dict(jpeg=features.version_codec('jpg'),
                   libjpeg_turbo=features.check_feature('libjpeg_turbo'),
                   libjpeg_turbo_version=features.version_feature('libjpeg_turbo'),
                   imaging_sha256=hashlib.sha256(core.read_bytes()).hexdigest())
    return dict(
        python=platform.python_version(), platform=platform.platform(),
        torch=str(torch.__version__), torchvision=str(torchvision.__version__),
        numpy=np.__version__, pillow=pillow_version, decoder=decoder,
        cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        device_type=device.type,
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.machine(),
        device_capability=list(torch.cuda.get_device_capability(device)) if device.type == 'cuda' else None,
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cpu_threads=torch.get_num_threads(),
    )


def record_runtime(path, runtime):
    """Reject mismatched or unverifiable caches and record each accepted call."""
    path = Path(path)
    previous = None
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous.get('schema_version') != 2 or 'environment' not in previous:
            raise RuntimeError(
                'This output uses a legacy runtime record without decoder/build checks. '
                'Complete it with the original code and environment, or rerun with a new --output directory.')
        for field in ('source_sha256', 'config_sha256', 'bank_sha256', 'batch_size', 'dense_limit', 'lu_tile'):
            if field not in previous or previous[field] != runtime[field]:
                raise RuntimeError('Resumed execution inputs changed: ' + field)
        old_environment = previous['environment']
        new_environment = runtime['environment']
        changed = sorted(key for key in old_environment.keys() | new_environment.keys()
                         if old_environment.get(key) != new_environment.get(key))
        if changed:
            raise RuntimeError('Resumed numerical environment changed: ' + ', '.join(changed)
                               + '. Restore the recorded environment or use a new --output directory.')
    elif any(path.parent.glob('*.pt')) or (path.parent / 'selection.json').exists():
        raise RuntimeError('Existing cached outputs have no runtime record; use a new --output directory.')
    record = dict(runtime)
    record['runs'] = list(previous.get('runs', [])) if previous is not None else []
    record['runs'].append(dict(started_at=datetime.now(timezone.utc).isoformat(), invocation=runtime['invocation']))
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
