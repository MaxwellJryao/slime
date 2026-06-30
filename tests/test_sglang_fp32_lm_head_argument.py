import argparse

from slime.backends.sglang_utils.arguments import add_sglang_arguments


def test_sglang_exposes_prefixed_fp32_lm_head_argument() -> None:
    parser = add_sglang_arguments(argparse.ArgumentParser())

    args = parser.parse_args(["--sglang-enable-fp32-lm-head"])

    assert args.sglang_enable_fp32_lm_head is True
