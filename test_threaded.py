# test_threaded.py

import threading
import time
from collections import Counter

import common
from threaded import optimizuj_nitima


def lazi_oceni(seq):
    """
    Brza lažna funkcija za testiranje logike.
    Ne koristi pravi evaluator, da test ne bi trajao dugo.
    """
    x = sum(ord(c) for c in seq) % 7

    return (
        float(x),
        float((x + 3) % 10),
        float((2 * x + 1) % 10),
    )


def test_threaded_basic():
    B = 120
    N = 20
    k = 5
    M = 10
    lam = 2.0
    G = 100
    T = 8
    seed = 42

    arhiva, evidencija, stats = optimizuj_nitima(
        oceni_func=lazi_oceni,
        B=B,
        N=N,
        k=k,
        M=M,
        lam=lam,
        G=G,
        T=T,
        seed=seed,
    )

    # Budžet ne sme biti prekoračen.
    assert stats["evaluacije_zapocete"] <= B

    # Niti moraju zaista generisati kandidate posle početne populacije.
    assert stats["evaluacije_zavrsene"] > N, stats
    assert stats["iteracije"] > 0, stats

    # Pošto sve niti završavaju pre povratka, započete i završene moraju biti jednake.
    assert stats["evaluacije_zavrsene"] == stats["evaluacije_zapocete"]

    # Svaka sekvenca u evidenciji mora imati ocene, ne None.
    for seq, ocene in evidencija.items():
        assert ocene is not None, f"Sekvenca {seq} nema ocene"

    # Broj jedinstvenih sekvenci u evidenciji mora biti jednak broju evaluacija.
    assert len(evidencija) == stats["evaluacije_zavrsene"]

    # Arhiva ne sme biti prazna ako ima evaluiranih kandidata.
    if stats["evaluacije_zavrsene"] > 0:
        assert len(arhiva) > 0


def test_bez_duplikata_mnogo_niti():
    pozivi = Counter()
    lk = threading.Lock()

    def f(seq):
        # sleep pušta GIL, pa se niti zaista prepliću.
        time.sleep(0.001)
        with lk:
            pozivi[seq] += 1
        return lazi_oceni(seq)

    # Alfabet {A, C} i dužina 4: prostor ima samo 2^4 = 16 sekvenci.
    staro = common.ALT_AA
    common.ALT_AA = {"A": ("C",), "C": ("A",)}
    try:
        _, evidencija, stats = optimizuj_nitima(f, B=100, N=4, k=2, M=3, lam=1.0,
                                                G=200, T=16, seed=7, pocetna="ACAC")
    finally:
        common.ALT_AA = staro

    assert max(pozivi.values()) == 1
    assert sum(pozivi.values()) == stats["evaluacije_zapocete"] <= 16
    assert all(v is not None for v in evidencija.values())


def test_iteracija_ne_prelazi_M():
    M = 7
    N = 20
    _, _, stats = optimizuj_nitima(lazi_oceni, B=200, N=N, k=5, M=M, lam=2.0,
                                   G=100, T=8, seed=1)

    assert stats["zapocete_po_iteraciji"]
    assert all(z <= M for z in stats["zapocete_po_iteraciji"])
    assert sum(stats["zapocete_po_iteraciji"]) == stats["evaluacije_zapocete"] - N


if __name__ == "__main__":
    test_threaded_basic()
    test_bez_duplikata_mnogo_niti()
    test_iteracija_ne_prelazi_M()
    print("Threaded test je prošao.")