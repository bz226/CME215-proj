import argparse

from trainer import loss_utils


def test_lamse_cli_flags_parse():
    parser = argparse.ArgumentParser()
    loss_utils.add_error_args(parser)
    args = parser.parse_args(["--lamse", "--lamse-lambda", "0.25", "--lamse-lmax", "16"])
    loss_utils.parse_arguments(args)
    assert loss_utils.config_dict["lamse"] is True
    assert loss_utils.config_dict["lamse_lambda"] == 0.25
    assert loss_utils.config_dict["lamse_lmax"] == 16
