#!/usr/bin/env python3
"""Remove the customerID column from a DataFrame."""


def drop_customerID(df):
    """Remove the customerID column."""
    return df.drop(columns=['customerID'])
