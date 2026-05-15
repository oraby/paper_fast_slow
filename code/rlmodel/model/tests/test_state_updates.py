import numpy as np

from .. import state_updates


def test_initialize_latent_state_defaults_to_half_values():
    state = state_updates.initialize_latent_state(
        include_Q=True,
        include_RewardRate=True,
    )

    assert state.q_left == 0.5
    assert state.q_right == 0.5
    assert state.reward_rate == 0.5
    assert state.include_Q is True
    assert state.include_RewardRate is True


def test_q_update_changes_only_left_when_left_chosen():
    q_left, q_right = state_updates.update_q_values(
        q_left=0.5,
        q_right=0.25,
        observed_choice_left=1.0,
        observed_reward=1.0,
        alpha=0.2,
    )

    assert np.isclose(q_left, 0.6)
    assert q_right == 0.25


def test_q_update_changes_only_right_when_right_chosen():
    q_left, q_right = state_updates.update_q_values(
        q_left=0.5,
        q_right=0.25,
        observed_choice_left=0.0,
        observed_reward=1.0,
        alpha=0.2,
    )

    assert q_left == 0.5
    assert np.isclose(q_right, 0.4)


def test_q_update_leaves_values_unchanged_for_no_choice():
    q_left, q_right = state_updates.update_q_values(
        q_left=0.5,
        q_right=0.25,
        observed_choice_left=np.nan,
        observed_reward=np.nan,
        alpha=0.2,
    )

    assert q_left == 0.5
    assert q_right == 0.25


def test_q_update_accepts_none_as_no_choice():
    q_left, q_right = state_updates.update_q_values(
        q_left=0.5,
        q_right=0.25,
        observed_choice_left=None,
        observed_reward=None,
        alpha=0.2,
    )

    assert q_left == 0.5
    assert q_right == 0.25


def test_reward_rate_update_follows_rescorla_wagner():
    reward_rate = state_updates.update_reward_rate(
        reward_rate=0.5,
        observed_reward=1.0,
        beta=0.2,
    )

    assert np.isclose(reward_rate, 0.6)


def test_starting_point_z_returns_zero_without_q():
    assert state_updates.compute_starting_point_z(
        q_left=0.8,
        q_right=0.2,
        delta=1.0,
        offset=0.5,
        include_Q=False,
    ) == 0.0


def test_starting_point_z_is_clipped_with_q():
    z = state_updates.compute_starting_point_z(
        q_left=1.0,
        q_right=0.01,
        delta=2.0,
        offset=0.0,
        include_Q=True,
    )

    assert z == 1.0


def test_trial_sigma_uses_reward_rate_only_when_enabled():
    assert state_updates.compute_trial_sigma(2.0, 0.25, True) == 0.5
    assert state_updates.compute_trial_sigma(2.0, 0.25, False) == 2.0


def test_trial_mu_returns_scalar_or_time_grid_array():
    assert np.isclose(state_updates.compute_trial_mu(0.4, 3.0), 1.2)

    time_grid = np.array([0.0, 0.1, 0.2])
    mu = state_updates.compute_trial_mu(0.4, 3.0, time_grid=time_grid)
    np.testing.assert_allclose(mu, np.array([1.2, 1.2, 1.2]))
