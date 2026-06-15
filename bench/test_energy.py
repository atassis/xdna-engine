import pytest

from bench.energy import EnergyMeter, readable


@pytest.mark.skipif(not readable(), reason="RAPL energy counter not user-readable")
def test_energy_meter_measures_positive():
    with EnergyMeter() as m:
        acc = 0
        for i in range(2_000_000):
            acc += i
    assert m.t > 0
    assert m.joules > 0
    assert m.watts > 0
