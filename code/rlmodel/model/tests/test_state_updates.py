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


def test_q_update_asymmetric_uses_alpha_unrewarded_when_unrewarded():
    # Reward == 0 on a left-choice trial: the asymmetric branch should
    # apply alpha_unrewarded (0.5) to Q_L, leaving Q_R untouched.
    q_left, q_right = state_updates.update_q_values(
        q_left=0.5,
        q_right=0.25,
        observed_choice_left=1.0,
        observed_reward=0.0,
        alpha=0.2,
        alpha_unrewarded=0.5,
    )

    assert np.isclose(q_left, 0.25)  # 0.5 + 0.5 * (0 - 0.5)
    assert q_right == 0.25


def test_q_update_asymmetric_uses_alpha_when_rewarded():
    # Reward == 1: the rewarded rate fires regardless of the asymmetric value.
    q_left, q_right = state_updates.update_q_values(
        q_left=0.5,
        q_right=0.25,
        observed_choice_left=1.0,
        observed_reward=1.0,
        alpha=0.2,
        alpha_unrewarded=0.9,
    )

    assert np.isclose(q_left, 0.6)  # 0.5 + 0.2 * (1 - 0.5)
    assert q_right == 0.25


def test_q_update_asymmetric_none_falls_back_to_symmetric():
    # Explicit None on alpha_unrewarded behaves like the legacy single-rate
    # call. Same result whether reward == 0 or 1.
    q_left, q_right = state_updates.update_q_values(
        q_left=0.5,
        q_right=0.25,
        observed_choice_left=1.0,
        observed_reward=0.0,
        alpha=0.2,
        alpha_unrewarded=None,
    )

    assert np.isclose(q_left, 0.4)  # 0.5 + 0.2 * (0 - 0.5)
    assert q_right == 0.25


def test_q_update_asymmetric_no_choice_picks_unrewarded():
    # no-choice currently leaves Q values unchanged regardless of which
    # learning rate would be selected. Still pin the no-change invariant
    # to make sure asymmetric wiring didn't perturb it.
    q_left, q_right = state_updates.update_q_values(
        q_left=0.5,
        q_right=0.25,
        observed_choice_left=np.nan,
        observed_reward=np.nan,
        alpha=0.2,
        alpha_unrewarded=0.9,
    )

    assert q_left == 0.5
    assert q_right == 0.25


def test_reward_rate_update_asymmetric_uses_beta_unrewarded_when_unrewarded():
    reward_rate = state_updates.update_reward_rate(
        reward_rate=0.5,
        observed_reward=0.0,
        beta=0.2,
        beta_unrewarded=0.5,
    )

    assert np.isclose(reward_rate, 0.25)  # 0.5 + 0.5 * (0 - 0.5)


def test_reward_rate_update_asymmetric_uses_beta_when_rewarded():
    reward_rate = state_updates.update_reward_rate(
        reward_rate=0.5,
        observed_reward=1.0,
        beta=0.2,
        beta_unrewarded=0.9,
    )

    assert np.isclose(reward_rate, 0.6)  # 0.5 + 0.2 * (1 - 0.5)


def test_reward_rate_update_asymmetric_none_falls_back_to_symmetric():
    reward_rate = state_updates.update_reward_rate(
        reward_rate=0.5,
        observed_reward=0.0,
        beta=0.2,
        beta_unrewarded=None,
    )

    assert np.isclose(reward_rate, 0.4)


# --- xp=np vectorized tests pinning the population-shape semantics ---
# The MLE population path calls these functions with
# ``xp=backend.xp`` and (n_candidates, n_sessions) shaped Q-state +
# (1, n_sessions) shaped trial inputs. Pin the broadcast semantics so
# the population path stays compatible when state_updates evolves.


def test_q_update_vectorized_broadcasts_population_shape():
    # 2 candidates × 3 sessions of Q-state, one trial of (1, 3) inputs.
    q_left = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    q_right = np.array([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]])
    # Session 0: rewarded left; session 1: unrewarded left; session 2: no choice.
    observed_choice_left = np.array([[1.0, 1.0, np.nan]])
    observed_reward = np.array([[1.0, 0.0, np.nan]])
    # Per-candidate broadcast: candidate 0 fast learner, candidate 1 slow.
    alpha = np.array([[0.5], [0.1]])
    alpha_unrewarded = np.array([[0.8], [0.2]])

    new_q_left, new_q_right = state_updates.update_q_values(
        q_left, q_right, observed_choice_left, observed_reward,
        alpha, alpha_unrewarded=alpha_unrewarded, xp=np)

    # Candidate 0, session 0: rewarded ⇒ 0.5 + 0.5*(1-0.5) = 0.75
    # Candidate 0, session 1: unrewarded ⇒ 0.5 + 0.8*(0-0.5) = 0.1
    # Candidate 0, session 2: no choice ⇒ 0.5 unchanged
    np.testing.assert_allclose(new_q_left[0], [0.75, 0.1, 0.5])
    # Candidate 1: slower rates.
    np.testing.assert_allclose(new_q_left[1], [0.55, 0.4, 0.5])
    # q_right is unchanged everywhere (left was chosen / no choice).
    np.testing.assert_allclose(new_q_right, q_right)


def test_compute_q_value_vectorized_matches_inlined_formula():
    # Population shape Q-state.
    q_left = np.array([[0.8, 0.2], [0.5, 0.5]])
    q_right = np.array([[0.2, 0.8], [0.5, 0.5]])
    result = state_updates.compute_q_value(q_left, q_right, xp=np)

    # Manual formula (replicates the pre-refactor inlined block):
    LOG_CIEL, LOG_CEIL_MAX = state_updates.LOG_CIEL, state_updates.LOG_CEIL_MAX
    expected = np.log(
        np.clip(q_left, LOG_CIEL, 1) / np.clip(q_right, LOG_CIEL, 1)
    ) / LOG_CEIL_MAX
    np.testing.assert_allclose(result, expected)


def test_compute_starting_point_z_returns_zero_without_q_array_form():
    # include_Q=False short-circuits to scalar 0 regardless of input shape.
    z = state_updates.compute_starting_point_z(
        np.array([0.8, 0.2]), np.array([0.2, 0.8]),
        delta=1.0, offset=0.0, include_Q=False, xp=np)
    assert z == 0.0


def test_compute_starting_point_z_clipped_array_form():
    z = state_updates.compute_starting_point_z(
        np.array([1.0, 1.0]), np.array([0.01, 0.01]),
        delta=2.0, offset=0.0, include_Q=True, xp=np)
    # Both entries clip to 1.0 (positive q_value × delta=2 saturates).
    np.testing.assert_allclose(z, [1.0, 1.0])


