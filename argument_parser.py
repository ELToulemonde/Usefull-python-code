# -*- coding: utf-8 -*-
"""
Created on Thu Aug  02 11:04:52 2018

@author: TOULEEM
"""
from urllib import parse

__all__ = ["encode_arguments"]


def encode_arguments(**kwargs):
    """
    Transform arguments to a list of coercible arguments in R and through cmd. (ie less failure).
    Readable by function get_inputs in R. + Handle some formats
    NB: It won't work if any argument contain an "="
    :param kwargs:
    :return: list ready to be sent to R.
    """
    prep_args = []
    for arg, value in kwargs.items():
        # Provide type for R
        if isinstance(value, bool):
            r_type = "logical"
        elif isinstance(value, float):
            r_type = "numeric"
        elif isinstance(value, int):
            r_type = "integer"
        elif isinstance(value, list):
            r_type = "list"
        else:
            r_type = "character"
        prep_args.append(arg + "=" + parse.quote(str(value)) + "=" + r_type)
    return prep_args


if __name__ == "__main__":
    print(encode_arguments(sep="|"))
