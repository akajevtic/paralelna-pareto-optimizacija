# test_processes.py

import time

import numpy as np

from common import spoji_i_azuriraj
from processes import (
    izaberi_susede,
    lokalni_budzet,
    _primi_poruke,
    _spoji_rezultate,
    pokreni_procese,
)


def test_topologije():
    rng = np.random.default_rng(0)
    assert izaberi_susede(3, 4, "ring", rng) == [0]
    assert izaberi_susede(0, 4, "star", rng) == [1, 2, 3]
    assert izaberi_susede(2, 4, "star", rng) == [0]
    assert izaberi_susede(1, 4, "all2all", rng) == [0, 2, 3]
    assert izaberi_susede(0, 1, "ring", rng) == []


def test_spajanje_arhiva():
    results = {
        0: {"stats": {}, "archive": [("A" * 32, (5, 5, 5), 0), ("C" * 32, (1, 1, 1), 0)]},
        1: {"stats": {}, "archive": [("A" * 32, (5, 5, 5), 1), ("D" * 32, (9, 0, 0), 1)]},
    }
    rez = _spoji_rezultate(results)
    assert sorted(rez["final_archive"]) == sorted(["A" * 32, "D" * 32])


def test_mali_pokret():
    rez = pokreni_procese(B=81, N=10, k=3, M=5, lam_list=[1.0], G=50, K=2, E=1,
                          topology="ring", send_selection="archive", P=2, seed=1,
                          timeout=120)
    assert rez["missing_ranks"] == []
    assert rez["agg"]["evaluacije_zavrsene"] <= 81
    assert rez["agg"]["velicina_arhive"] >= 1


def test_podela_budzeta():
    for B, P in [(2000, 4), (81, 2), (10, 3), (3, 4)]:
        delovi = [lokalni_budzet(r, P, B) for r in range(P)]
        assert sum(delovi) == B
        assert max(delovi) - min(delovi) <= 1


def test_random_topologija():
    rng = np.random.default_rng(0)
    for rank in range(4):
        vidjeni = set()
        for _ in range(100):
            s = izaberi_susede(rank, 4, "random", rng)
            assert len(s) == 1 and s[0] != rank and 0 <= s[0] < 4
            vidjeni.add(s[0])
        assert vidjeni == set(range(4)) - {rank}


def test_prijem_poznat_dominiran_nedominiran():
    poznat, dom, nedom, lok = "A" * 32, "C" * 32, "D" * 32, "E" * 32
    evidencija = {poznat: (1.0, 1.0, 1.0), lok: (5.0, 5.0, 5.0)}
    origin = {poznat: 0, lok: 0}
    stats = {"primljeno_poruka": 0, "primljeno_novih": 0}
    msg = {"tip": "CANDS", "from": 1, "iter": 1, "kandidati": [
        (poznat, (9.0, 9.0, 9.0), 1),   # poznat: ne sme da prepiše ocene
        (dom, (1.0, 1.0, 1.0), 1),      # dominiran od lok
        (nedom, (9.0, 0.0, 0.0), 1),    # nedominiran
    ]}

    primljeni = _primi_poruke([msg], evidencija, origin, stats)

    assert primljeni == [dom, nedom]
    assert evidencija[poznat] == (1.0, 1.0, 1.0)
    assert stats == {"primljeno_poruka": 1, "primljeno_novih": 2}

    _, arhiva, *_ = spoji_i_azuriraj([lok, poznat], [lok], primljeni, evidencija, N=4)
    assert nedom in arhiva and dom not in arhiva


def test_proces_zavrsava_dok_poruke_stizu():
    # all2all, E=1: svi šalju svima posle svake iteracije; procesi završavaju u različito vreme.
    t0 = time.perf_counter()
    rez = pokreni_procese(B=60, N=8, k=3, M=3, lam_list=[0.5, 1.0, 2.0, 4.0], G=50, K=5, E=1,
                          topology="all2all", send_selection="random-evidence", P=4, seed=2,
                          timeout=120)

    assert rez["missing_ranks"] == []

    # Niko nije blokiran na slanju prema procesu koji je završio.
    assert time.perf_counter() - t0 < 60

    # Kasne poruke su odbačene u drain fazi, ne broje se kao primljene.
    agg = rez["agg"]
    assert agg["poslato_poruka"] > 0
    assert agg["primljeno_poruka"] <= agg["poslato_poruka"]


if __name__ == "__main__":
    test_topologije()
    test_spajanje_arhiva()
    test_mali_pokret()
    test_podela_budzeta()
    test_random_topologija()
    test_prijem_poznat_dominiran_nedominiran()
    test_proces_zavrsava_dok_poruke_stizu()
    print("OK")