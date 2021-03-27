"""Module for describing the schema of features"""
# Standard libraries
from pydantic import BaseModel


class BankNote(BaseModel):
    """
    Scheme for Input features
    """

    variance: float
    skewness: float
    curtosis: float
    entropy: float
