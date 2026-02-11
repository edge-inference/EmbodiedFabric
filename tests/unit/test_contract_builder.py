from contracts.codesign.contract import ContractBuilder


def test_contract_builder_vacuity_and_violation():
    c = (
        ContractBuilder("c")
        .assume("a", "a == true", evaluator=lambda s: bool(s.get("a")))
        .guarantee("g", "g == true", evaluator=lambda s: bool(s.get("g")))
        .build()
    )

    trace = [
        {"a": True, "g": True},
        {"a": True, "g": False},  # violation
        {"a": False, "g": False},  # vacuous
    ]

    result = c.check_satisfaction(trace)
    assert result.satisfied is False
    assert len(result.violations) == 1
    assert result.vacuous_steps == 1

