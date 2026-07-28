import argparse

from slime.utils.arguments import get_slime_extra_args_provider


def _parser() -> argparse.ArgumentParser:
    return get_slime_extra_args_provider()(argparse.ArgumentParser())


def test_controller_max_action_scale_defaults_to_ten() -> None:
    assert (
        _parser()
        .parse_args(["--rollout-batch-size", "1"])
        .polar_controller_max_action_scale
        == 10.0
    )


def test_controller_max_action_scale_accepts_zero_for_uncapped() -> None:
    assert (
        _parser()
        .parse_args(
            [
                "--rollout-batch-size",
                "1",
                "--polar-controller-max-action-scale",
                "0",
            ]
        )
        .polar_controller_max_action_scale
        == 0.0
    )
