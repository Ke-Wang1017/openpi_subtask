from openpi.training import config as _config
import openpi.transforms as _transforms


def test_pi05_libero_default_actions_column():
    config = _config.get_config("pi05_libero")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert tuple(data_config.action_sequence_keys) == ("actions",)
    assert isinstance(data_config.repack_transforms.inputs[0], _transforms.RepackTransform)
    assert data_config.repack_transforms.inputs[0].structure["actions"] == "actions"


def test_pi05_libero_action_column():
    config = _config.get_config("pi05_libero_action")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert tuple(data_config.action_sequence_keys) == ("action",)
    assert data_config.repack_transforms.inputs == ()
