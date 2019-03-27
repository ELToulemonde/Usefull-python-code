# From https://github.com/ELToulemonde/Usefull-python-code/blob/master/filter_cdb.py
import logging
from math import log
import numpy as np
import pandas as pd
from tqdm import tqdm

__all__ = ["fast_is_constant", "fast_is_double", "fast_is_bijection", "filter_cdb", "which_are_constant",
           "which_are_double", "which_are_bijection"]
logger = logging.getLogger(__name__)


def fast_is_constant(obj):
    if isinstance(obj, pd.core.series.Series):
        obj = obj.values
    n = len(obj)
    if n > 1:
        exp_factor = 10
        max_power = int(log(n) / log(exp_factor)) + 1
        unique_list = np.array([], dtype=obj.dtype)
        for i in range(1, max_power + 1):
            I = range((exp_factor ** (i - 1)) - 1, min(exp_factor ** i - 2, n))
            if obj.dtype.type in [np.str_, np.object_, np.string_]:
                unique_list = np.unique(np.append(obj[I], unique_list))
                if len(unique_list) > 1:
                    return False
            else:
                # Handle nans for not str
                n_nans = np.sum(pd.isnull(obj[I]))
                if n_nans == len(I):
                    # All Nans
                    if not np.any(pd.isnull(unique_list)):
                        # If nan not yet in list
                        unique_list = np.append(unique_list, np.array(np.nan))
                elif n_nans == 0:
                    unique_list = np.append(unique_list, np.unique(obj[I]))
                else:  # If there are some nans but not only nans False
                    return False
                unique_list = np.unique(unique_list)
                if len(unique_list) > 1:
                    return False
    return True


def fast_is_double(obj1, obj2):
    # First conversion
    if isinstance(obj1, pd.core.series.Series):
        obj1 = obj1.values
    if isinstance(obj2, pd.core.series.Series):
        obj2 = obj2.values
    # Basic controls
    if obj1.dtype.type != obj2.dtype.type:
        return False
    if len(obj1) != len(obj2):
        return False
    n = len(obj1)
    if n >= 1:
        # Exponential search
        exp_factor = 10
        max_power = int(log(n) / log(exp_factor)) + 1
        for i in range(1, max_power + 1):
            I = range((exp_factor ** (i - 1)) - 1, min(exp_factor ** i - 2, n))
            if obj1.dtype.type not in [np.str_, np.object_, np.string_]:
                if not np.allclose(obj1[I], obj2[I], rtol=0, atol=0, equal_nan=True):
                    return False
            else:
                if not np.all(obj1[I] == obj2[I]):
                    return False
    return True


def fast_is_bijection(obj1, obj2):
    # Fist transform
    if isinstance(obj1, pd.core.series.Series):
        obj1 = obj1.values
    if isinstance(obj2, pd.core.series.Series):
        obj2 = obj2.values
    # Basic controls
    if len(obj1) != len(obj2):
        return False
    # Exponential search
    n = len(obj1)
    if n > 1:
        exp_factor = 10
        max_power = int(log(n) / log(exp_factor)) + 1
        for i in range(1, max_power + 1):
            I = range((exp_factor ** (i - 1)) - 1, min(exp_factor ** i - 2, n))
            tmp_df = pd.DataFrame({"obj1": obj1[I], "obj2": obj2[I]})
            n1 = len(tmp_df.loc[:, ["obj1"]].drop_duplicates())
            n2 = len(tmp_df.loc[:, ["obj2"]].drop_duplicates())
            if n1 != n2:
                return False
            n12 = tmp_df.drop_duplicates().shape[0]

            if n12 != n1:
                return False
    return True


def filter_cdb(data_set: pd.DataFrame, type="cdb", verbose=True):
    if len(np.unique(data_set.columns)) != len(data_set.columns):
        raise ValueError("data_set should have unique column names")
    if "c" in type:
        constant_list = which_are_constant(data_set, verbose)
        data_set.drop(columns=constant_list, inplace=True)
    if "d" in type:
        if verbose:
            logger.info("filter_cdb: Start to look for double.")
        double_list = which_are_double(data_set, verbose)
        data_set.drop(columns=double_list, inplace=True)
    if "b" in type:
        if verbose:
            logger.info("filter_cdb: Start to look for bijection.")
        bijection_list = which_are_bijection(data_set, verbose)
        data_set.drop(columns=bijection_list, inplace=True)

    return data_set


def which_are_bijection(data_set, verbose):
    bijection_list = []
    col_names = data_set.columns
    for col_ind1 in tqdm(range(0, len(col_names) - 1)):
        col1 = col_names[col_ind1]
        for col_ind2 in range(col_ind1 + 1, len(col_names)):
            col2 = col_names[col_ind2]
            if fast_is_bijection(data_set[col1], data_set[col2]):
                bijection_list.append(col1)
                if verbose:
                    logger.info("filter_cdb: " + col1 + " is a bijection of " + col2 + ". Will drop it.")
                break
    if verbose:
        logger.info("filter_cdb: found " + str(len(bijection_list)) + " bijection dropping them.")
    return bijection_list


def which_are_double(data_set, verbose):
    double_list = []
    col_names = data_set.columns
    for col_ind1 in tqdm(range(0, len(col_names) - 1)):
        col1 = col_names[col_ind1]
        for col_ind2 in range(col_ind1 + 1, len(col_names)):
            col2 = col_names[col_ind2]
            if fast_is_double(data_set[col1], data_set[col2]):
                double_list.append(col1)
                if verbose:
                    logger.info("filter_cdb: " + col1 + " is in double of " + col2 + ". Will drop it.")
                break
    if verbose:
        logger.info("filter_cdb: found " + str(len(double_list)) + " double dropping them.")
    return double_list


def which_are_constant(data_set, verbose):
    if verbose:
        logger.info("filter_cdb: Start to look for constant.")
    constant_list = []
    for col in tqdm(data_set.columns):
        if fast_is_constant(data_set[col]):
            constant_list.append(col)
            if verbose:
                logger.info("filter_cdb: " + col + " is constant. Will drop it.")
    if verbose:
        logger.info("filter_cdb: found " + str(len(constant_list)) + " constant dropping them.")
    return constant_list
