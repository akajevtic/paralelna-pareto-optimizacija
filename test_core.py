# test_core.py

import numpy as np

from common import (
    AA,
    POCETNA_SEKVENCA,
    dominira,
    pareto_rangovi_i_crowding,
    napravi_pocetnu_populaciju,
    generisi_kandidata,
    proveri_sekvencu,
    crowding_distance,
    optimizuj_sekvencijalno,
)


def test_dominacija():
    assert dominira((5, 4, 4), (3, 4, 4)) is True
    assert dominira((5, 4, 4), (5, 4, 4)) is False  # jednako nije dominacija
    assert dominira((5, 4, 4), (6, 2, 2)) is False
    assert dominira((6, 2, 2), (5, 4, 4)) is False
    assert dominira((5, 5, 5), (5, 4, 5)) is True
    assert dominira((5, 4, 5), (5, 5, 5)) is False


def test_pareto_rangovi():
    scores = {
        "A": (5.0, 4.0, 4.0),
        "B": (3.0, 4.0, 4.0),
        "C": (6.0, 2.0, 2.0),
    }

    ranks, crowd, layers = pareto_rangovi_i_crowding(["A", "B", "C"], scores)

    # A dominira B, ali A i C se međusobno ne dominiraju.
    assert ranks["A"] == 1
    assert ranks["C"] == 1
    assert ranks["B"] == 2

    assert len(layers[0]) == 2
    assert set(layers[0]) == {"A", "C"}
    assert layers[1] == ["B"]


def test_pocetna_populacija():
    rng = np.random.default_rng(1)

    N = 10
    k = 5

    populacija, evidencija = napravi_pocetnu_populaciju(POCETNA_SEKVENCA, N, k, rng)

    assert len(populacija) == N
    assert len(set(populacija)) == N
    assert populacija[0] == POCETNA_SEKVENCA

    for s in populacija:
        assert proveri_sekvencu(s)
        assert s in evidencija


def test_generisanje_kandidata():
    rng = np.random.default_rng(2)

    p1 = POCETNA_SEKVENCA
    p2 = "A" * len(POCETNA_SEKVENCA)

    populacija = [p1, p2]

    for _ in range(500):
        child = generisi_kandidata(populacija, rng, 2.0)

        assert proveri_sekvencu(child)

        # Dužina mora ostati ista.
        assert len(child) == len(POCETNA_SEKVENCA)

        # Samo dozvoljena slova.
        assert set(child) <= set(AA)


def test_crowding_vrednosti():
    scores = {
        "A": (0.0, 4.0, 0.0),
        "B": (1.0, 2.0, 0.0),
        "C": (3.0, 1.0, 0.0),
        "D": (4.0, 0.0, 0.0),
    }
    c = crowding_distance(["A", "B", "C", "D"], scores)

    assert c["A"] == float("inf") and c["D"] == float("inf")

    # B: (3-0)/4 + (4-1)/4 = 1.5 ; C: (4-1)/4 + (2-0)/4 = 1.25 ; treća osa ima raspon 0
    assert abs(c["B"] - 1.5) < 1e-12
    assert abs(c["C"] - 1.25) < 1e-12


def _brojac():
    pozivi = []

    def f(seq):
        pozivi.append(seq)
        x = sum(map(ord, seq)) % 7
        return (float(x), float((x + 3) % 10), float((2 * x + 1) % 10))

    return f, pozivi


def test_preskakanje_duplikata():
    f, pozivi = _brojac()
    _, evidencija, stats = optimizuj_sekvencijalno(f, B=300, N=10, k=3, M=5, lam=1.0, G=1000, seed=3)

    # Nijedan string nije ocenjen dvaput.
    assert len(pozivi) == len(set(pozivi))
    assert len(pozivi) == stats["evaluacije"]

    # Duplikati su se pojavili i preskočeni su.
    assert stats["neuspesi_ukupno"] > 0


def test_mali_budzet():
    f, pozivi = _brojac()
    arhiva, _, stats = optimizuj_sekvencijalno(f, B=5, N=10, k=3, M=5, lam=1.0, G=100, seed=1)

    assert len(pozivi) == 5
    assert stats["status"] == "budzet_potrosen_posle_pocetne_populacije"
    assert len(arhiva) >= 1


def test_zaustavljanje_po_G():
    # Dva roditelja se razlikuju na jednoj poziciji, lam=0: svako dete je jedan od roditelja.
    f, _ = _brojac()
    _, _, stats = optimizuj_sekvencijalno(f, B=1000, N=2, k=1, M=5, lam=0.0, G=10,
                                          seed=1, pocetna="ACDE")

    assert stats["status"] == "prag_neuspesnih_pokusaja_G"
    assert stats["evaluacije"] == 2


if __name__ == "__main__":
    test_dominacija()
    test_pareto_rangovi()
    test_pocetna_populacija()
    test_generisanje_kandidata()
    test_crowding_vrednosti()
    test_preskakanje_duplikata()
    test_mali_budzet()
    test_zaustavljanje_po_G()
    print("Svi osnovni testovi su prošli.")