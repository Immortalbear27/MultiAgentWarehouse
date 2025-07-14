from model import WarehouseEnvModel

# Testing Framework for model.py:
def _test_fallback_consistency():
    # 1. Set up a tiny open grid with one drop‐zone
    model = WarehouseEnvModel(width=10, height=10, drop_coords=[(9,9)])
    start, goal = (0, 0), (9, 9)
    # Clear any stray reservations
    model.reservations.clear()

    # 2. Compare pure A* vs greedy→A* fallback when unblocked
    p_astar = model.compute_path(start, goal)
    p_greedy = model.compute_path_to_drop(start, goal)
    assert p_greedy == p_astar, "Greedy path must match A* when no blockage"

    # 3. Force a block on the first greedy step
    first_step = p_greedy[0]
    t1 = model.schedule.time + 1
    model.reservations[(first_step[0], first_step[1], t1)] = None

    p_fallback = model.compute_path_to_drop(start, goal)
    assert p_fallback == p_astar, "Fallback must invoke A* and find the same route"
    print("✔ Fallback consistency tests passed")


def _test_reset_clears_state():
    model = WarehouseEnvModel(width=10, height=10, drop_coords=[(9,9)])
    # simulate some steps to populate caches/reservations
    model.step()
    assert model.path_cache,    "path_cache should be non-empty after steps"
    assert model.reservations,  "reservations should be non-empty after steps"

    # Now “reset” by making a brand-new instance
    model2 = WarehouseEnvModel(width=10, height=10, drop_coords=[(9,9)])
    assert not model2.path_cache,    "New model must start with empty path_cache"
    assert not model2.reservations,  "New model must start with no reservations"
    assert hasattr(model2, "static_dist"), "static_dist must be built on init"
    print("✔ Reset state-cleanup tests passed")


if __name__ == "__main__":
    _test_fallback_consistency()
    _test_reset_clears_state()