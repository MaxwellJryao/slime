import argparse

from slime.utils.arguments import get_slime_extra_args_provider


def _parser() -> argparse.ArgumentParser:
    return get_slime_extra_args_provider()(argparse.ArgumentParser())


def test_controller_turn_cost_arguments_are_disabled_by_default() -> None:
    args = _parser().parse_args(["--rollout-batch-size", "1"])

    assert args.polar_controller_cost_advantage_coef == 0.0
    assert args.polar_controller_cost_advantage_mode == "cost_to_go"
    assert args.polar_controller_cost_min_group_accuracy == 0.0
    assert args.polar_controller_cost_to_go_gamma == 1.0
    assert args.polar_controller_cost_to_go_horizon == 1
    assert args.polar_controller_cost_include_all_trajectories is False


def test_controller_turn_cost_arguments_are_configurable() -> None:
    args = _parser().parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--polar-controller-cost-advantage-coef",
            "0.5",
            "--polar-controller-cost-advantage-mode",
            "trajectory_total",
            "--polar-controller-cost-min-group-accuracy",
            "0.25",
            "--polar-controller-cost-to-go-gamma",
            "0.9",
            "--polar-controller-cost-to-go-horizon",
            "8",
            "--polar-controller-cost-include-all-trajectories",
        ]
    )

    assert args.polar_controller_cost_advantage_coef == 0.5
    assert args.polar_controller_cost_advantage_mode == "trajectory_total"
    assert args.polar_controller_cost_min_group_accuracy == 0.25
    assert args.polar_controller_cost_to_go_gamma == 0.9
    assert args.polar_controller_cost_to_go_horizon == 8
    assert args.polar_controller_cost_include_all_trajectories is True
