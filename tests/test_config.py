import pytest

from chord import ChordConfig, UserKnobs


def test_userknobs_validate_ranges():
    UserKnobs(M=0.5, rho=0.5, epsilon=0.1).validate()
    with pytest.raises(ValueError):
        UserKnobs(M=1.5).validate()
    with pytest.raises(ValueError):
        UserKnobs(rho=-0.1).validate()
    with pytest.raises(ValueError):
        UserKnobs(epsilon=-0.01).validate()


def test_normalized_theta_sums_to_one():
    k = UserKnobs(theta={"bridge": 3.0, "trend": 1.0})
    nt = k.normalized_theta()
    assert abs(sum(nt.values()) - 1.0) < 1e-12
    assert abs(nt["bridge"] - 0.75) < 1e-12


def test_normalized_theta_rejects_negative():
    with pytest.raises(ValueError):
        UserKnobs(theta={"bridge": -1.0}).validate()


def test_config_validate_rejects_bad_delta():
    with pytest.raises(ValueError):
        ChordConfig(eigentrust_delta=1.0).validate()
    with pytest.raises(ValueError):
        ChordConfig(eigentrust_delta=0.0).validate()


def test_config_rejects_zero_epsilon_min():
    # epsilon_min > 0 is the identifiability anchor invariant (§6.2).
    with pytest.raises(ValueError):
        ChordConfig(epsilon_min=0.0).validate()


def test_clamp_epsilon():
    cfg = ChordConfig(epsilon_min=0.05, epsilon_max=0.5)
    assert cfg.clamp_epsilon(0.0) == 0.05
    assert cfg.clamp_epsilon(0.9) == 0.5
    assert cfg.clamp_epsilon(0.2) == 0.2


def test_exposed_no_reaction_c_bounds():
    with pytest.raises(ValueError):
        ChordConfig(exposed_no_reaction_c=1.0).validate()
    with pytest.raises(ValueError):
        ChordConfig(exposed_no_reaction_c=0.0).validate()
