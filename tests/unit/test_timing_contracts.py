from contracts.codesign.timing import TimingContract


def test_control_loop_contract_satisfaction():
    c = TimingContract.control_loop_contract(
        name="loop",
        sensing_latency_ms=15.0,
        computation_latency_ms=5.0,
        actuation_latency_ms=10.0,
        loop_period_ms=50.0,
    )

    ok = {"sensing_latency_ms": 10.0, "computation_latency_ms": 5.0, "actuation_latency_ms": 10.0}
    violate = {"sensing_latency_ms": 10.0, "computation_latency_ms": 40.0, "actuation_latency_ms": 10.0}
    vacuous = {"sensing_latency_ms": 25.0, "computation_latency_ms": 20.0, "actuation_latency_ms": 10.0}

    ok_result = c.check_satisfaction([ok])
    assert ok_result.satisfied is True
    assert ok_result.vacuous_steps == 0

    violate_result = c.check_satisfaction([violate])
    assert violate_result.satisfied is False
    assert violate_result.vacuous_steps == 0
    assert len(violate_result.violations) >= 1

    vacuous_result = c.check_satisfaction([vacuous])
    assert vacuous_result.satisfied is True
    assert vacuous_result.vacuous_steps == 1

