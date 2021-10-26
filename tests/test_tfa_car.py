import numpy as np
import pandas as pd
import pytest

from tfa_car import TFAOutDict, tfa_car

tfa_out: TFAOutDict = {}
with open("tfa_sample_output.txt", "r") as f:
    x = f.read()
data = x.split("\n\n\n")
filt_data = [d for d in data if len(d) > 2]
filt_data[0] = "\n".join(filt_data[0].split("\n")[-3:])


def extract(d):
    metadata = {}
    dat = d.split("\n")
    i = 0
    da = dat[i]
    while da.startswith("#"):
        me_arr = da.replace("# ", "").split(": ")
        metadata[me_arr[0]] = me_arr[1]
        i += 1
        da = dat[i]
    if metadata["type"] == "scalar":
        return metadata["name"], float(da)
    if metadata["type"] == "complex matrix":
        da_arr = np.array(
            [
                [float(x) for x in da.replace("(", "").replace(")", "").split(",")]
                for da in dat[i:]
            ]
        )
        da_out = np.empty([da_arr.shape[0], 1], dtype=np.complex128)
        da_out.real = np.expand_dims(da_arr[:, 0], -1)
        da_out.imag = np.expand_dims(da_arr[:, 1], -1)
        return metadata["name"], da_out
    if metadata["type"] == "matrix":
        return metadata["name"], np.array([float(x) for x in dat[i:]])


for d in filt_data:
    n, x = extract(d)
    tfa_out[n] = x

df = pd.read_csv("tfa_sample_data.txt", sep="\t")
fs = 1 / np.mean(np.array(df["Time"][1:].tolist()) - np.array(df["Time"][:-1].tolist()))
tfa_test = tfa_car(df["MABP [mmHg]"].tolist(), df["CBFV-L [cm/s]"].tolist(), fs)
print("Pxx",tfa_out["Pxx"][0])
print("Pxy",tfa_out["Pxy"][0])
print("Pyy",tfa_out["Pyy"][0])

@pytest.mark.parametrize("key", [k for k in tfa_test.keys()])
def test_tfa_car(key):
    if isinstance(tfa_out[key], np.ndarray):
        # Test shapes
        assert tfa_out[key].shape[0] == tfa_test[key].shape[0]
        # Test first element
        assert tfa_out[key][0] == pytest.approx(tfa_test[key][0], 0.001)
        # Test middle element
        mid = len(tfa_out[key]) // 2
        assert tfa_out[key][mid] == pytest.approx(tfa_test[key][mid], 0.001)
        # Test last element
        assert tfa_out[key][-1] == pytest.approx(tfa_test[key][-1], 0.001)
    else:
        assert tfa_out[key] == pytest.approx(tfa_test[key], 0.001)
