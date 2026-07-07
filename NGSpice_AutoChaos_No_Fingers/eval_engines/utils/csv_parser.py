import numpy as np
import pandas as pd
from typing import Tuple, Dict

def parse_cadence_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    df = pd.read_csv(csv_path)
    if 'Vin' in df.columns:
        Vin = df['Vin'].values
    else:
        Vin = df.iloc[:, 0].values
    Vout_cols = [col for col in df.columns if 'Vout' in col or 'V(' in col]
    if Vout_cols:
        Vout = df[Vout_cols].values
    else:
        Vout = df.iloc[:, 1:].values
    Vc = np.linspace(0, 1.1, Vout.shape[1])
    return Vin, Vc, Vout

def export_to_matlab_format(Vin: np.ndarray, Vc: np.ndarray, Vout: np.ndarray, output_path: str):

    import scipy.io
    scipy.io.savemat(output_path, {
        'Vin': Vin,
        'Vc': Vc,
        'Vout': Vout
    })
