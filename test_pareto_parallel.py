# test_pareto_parallel.py

from common import (
    pareto_rangovi_i_crowding,
    spoji_i_azuriraj,
)

from pareto_parallel import (
    pareto_rangovi_i_crowding_pool,
    spoji_i_azuriraj_pool,
)


def assert_crowding_equal(crowd1, crowd2, seqs):
    for s in seqs:
        a = crowd1[s]
        b = crowd2[s]

        if a == float("inf"):
            assert b == float("inf"), f"Crowding mismatch za {s}: {a} vs {b}"
        else:
            assert abs(a - b) < 1e-12, f"Crowding mismatch za {s}: {a} vs {b}"


def test_mali_skup():
    scores = {
        "A": (5.0, 4.0, 4.0),
        "B": (3.0, 4.0, 4.0),
        "C": (6.0, 2.0, 2.0),
    }

    seqs = ["A", "B", "C"]

    r1, c1, l1 = pareto_rangovi_i_crowding(seqs, scores)

    r2, c2, l2 = pareto_rangovi_i_crowding_pool(
        seqs,
        scores,
        max_workers=4,
        executor=None,
        min_size=0,
    )

    assert r1 == r2
    assert l1 == l2
    assert_crowding_equal(c1, c2, seqs)


def test_veci_skup():
    n = 120

    seqs = [f"SEQ{i:03d}" for i in range(n)]

    scores = {}

    for i, s in enumerate(seqs):
        scores[s] = (
            float((i * 7) % 10),
            float((i * 3) % 10),
            float((i * 5 + 1) % 10),
        )

    r1, c1, l1 = pareto_rangovi_i_crowding(seqs, scores)

    r2, c2, l2 = pareto_rangovi_i_crowding_pool(
        seqs,
        scores,
        max_workers=4,
        executor=None,
        min_size=0,
    )

    assert r1 == r2
    assert l1 == l2
    assert_crowding_equal(c1, c2, seqs)


def test_update_populacije():
    n = 120

    seqs = [f"SEQ{i:03d}" for i in range(n)]

    scores = {}

    for i, s in enumerate(seqs):
        scores[s] = (
            float((i * 7) % 10),
            float((i * 3) % 10),
            float((i * 5 + 1) % 10),
        )

    populacija = seqs[:40]
    arhiva = seqs[40:60]
    novi = seqs[60:90]

    N = 50

    p1, a1, _, _, _ = spoji_i_azuriraj(
        populacija,
        arhiva,
        novi,
        scores,
        N,
    )

    p2, a2, _, _, _ = spoji_i_azuriraj_pool(
        populacija,
        arhiva,
        novi,
        scores,
        N,
        max_workers=4,
        executor=None,
        min_size=0,
    )

    assert p1 == p2
    assert a1 == a2


def test_rank_fn_se_koristi():
    pozivi = []

    def spy(seqs, scores):
        pozivi.append(len(seqs))
        return pareto_rangovi_i_crowding(seqs, scores)

    scores = {"A": (1.0, 2.0, 3.0), "B": (3.0, 2.0, 1.0)}
    spoji_i_azuriraj(["A"], [], ["B"], scores, 2, rank_fn=spy)

    assert pozivi == [2]


if __name__ == "__main__":
    test_mali_skup()
    test_veci_skup()
    test_update_populacije()
    test_rank_fn_se_koristi()
    print("Pareto parallel testovi su prošli.")