import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy


def test_libero_inputs_accepts_action_alias():
    transform = libero_policy.LiberoInputs(model_type=_model.ModelType.PI05)
    action = np.zeros((10, 7), dtype=np.float32)
    data = {
        "state": np.zeros((8,), dtype=np.float32),
        "image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "action": action,
    }

    transformed = transform(data)

    assert "actions" in transformed
    np.testing.assert_array_equal(transformed["actions"], action)


def test_libero_inputs_accepts_dotted_lerobot_keys():
    transform = libero_policy.LiberoInputs(model_type=_model.ModelType.PI05)
    action = np.zeros((10, 7), dtype=np.float32)
    data = {
        "observation.state": np.zeros((8,), dtype=np.float32),
        "observation.images.image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.images.wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "action": action,
    }

    transformed = transform(data)

    assert "actions" in transformed
    np.testing.assert_array_equal(transformed["actions"], action)
