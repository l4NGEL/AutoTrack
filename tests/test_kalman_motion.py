from tracking.kalman_motion import KalmanBoxTracker


def test_predict_without_update_holds_initial_box():
    kf = KalmanBoxTracker((0, 0, 10, 10))
    x1, y1, x2, y2 = kf.predict()
    assert abs(x1 - 0) < 1e-6
    assert abs(y1 - 0) < 1e-6
    assert abs((x2 - x1) - 10) < 1e-6
    assert abs((y2 - y1) - 10) < 1e-6


def test_velocity_zero_before_any_update():
    kf = KalmanBoxTracker((0, 0, 10, 10))
    assert kf.velocity_magnitude == 0.0


def test_tracker_learns_constant_velocity_and_predicts_ahead():
    # Box moves +5px in x every frame, starting at x1=0.
    kf = KalmanBoxTracker((0, 0, 10, 10))
    for step in range(1, 15):
        kf.predict()
        x1 = step * 5.0
        kf.update((x1, 0, x1 + 10, 10))

    # After many consistent updates the filter should have learned vx ~= 5.
    vx, _ = kf.velocity
    assert 3.5 < vx < 6.5

    predicted = kf.predict()
    expected_next_x1 = 15 * 5.0
    assert abs(predicted[0] - expected_next_x1) < 5.0


def test_innovation_norm_small_for_consistent_motion():
    kf = KalmanBoxTracker((0, 0, 10, 10))
    for step in range(1, 10):
        kf.predict()
        x1 = step * 5.0
        kf.update((x1, 0, x1 + 10, 10))
    # By now motion is predictable, innovation (prediction error) should be small.
    assert kf.last_innovation_norm < 2.0
