# threaded.py

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from common import (
    POCETNA_SEKVENCA,
    napravi_pocetnu_populaciju,
    generisi_kandidata,
    pareto_rangovi_i_crowding,
    spoji_i_azuriraj,
)

from pareto_parallel import pareto_rangovi_i_crowding_pool


class DeljenoStanje:
    """
    Deljeno stanje cele pretrage.
    populacija, arhiva, evidencija i brojači se menjaju samo pod self.lock.
    Jedini RNG programa se koristi samo pod self.rng_lock.
    """

    def __init__(self, populacija, arhiva, evidencija, budzet, M, G, rng):
        self.lock = threading.Lock()
        self.rng_lock = threading.Lock()
        self.rng = rng

        self.populacija = populacija
        self.arhiva = arhiva
        self.evidencija = evidencija

        self.budzet = budzet
        self.M = M
        self.G = G

        self.neuspesi = 0
        self.neuspesi_ukupno = 0
        self.zapocete = 0
        self.zavrsene = 0
        self.in_flight = 0
        self.stop = False

        # Iteracija je samo brojač: nova počinje na svakih M rezervisanih kandidata.
        self.iteracije = 0
        self.zapocete_u_iteraciji = 0
        self.zapocete_po_iteraciji = []
        self.trajanja_iteracija = []
        self.iter_start = time.perf_counter()

    def treba_stati(self):
        # Poziva se samo pod self.lock.
        return self.stop or self.budzet <= 0 or self.neuspesi >= self.G


def _novi_stats(T):
    return {
        "T": T,
        "evaluacije": 0,
        "evaluacije_zapocete": 0,
        "evaluacije_zavrsene": 0,
        "iteracije": 0,
        "zapocete_po_iteraciji": [],
        "neuspesi_ukupno": 0,
        "status": "",
        "ukupno_vreme": 0.0,
        "ukupno_vreme_iteracija": 0.0,
        "prosecno_vreme_iteracije": 0.0,
        "parallel_rank": False,
    }


def optimizuj_nitima(
    oceni_func,
    B: int,
    N: int,
    k: int,
    M: int,
    lam: float,
    G: int,
    T: int,
    seed: int,
    pocetna: str = POCETNA_SEKVENCA,
    parallel_rank: bool = False,
):
    """
    Verzija optimizacije sa T radnih niti u jednom procesu.

    Propisana verzija (P2) rangira sekvencijalno.
    parallel_rank=True je dodatni eksperiment (P2-pool): rangiranje ide
    kroz odvojeni ThreadPoolExecutor iz pareto_parallel.py.
    """

    if B <= 0 or N <= 0 or M <= 0 or T <= 0:
        stats = _novi_stats(T if T > 0 else 0)
        stats["status"] = "losi_parametri"
        return [], {}, stats

    if T > 1 and parallel_rank:
        with ThreadPoolExecutor(max_workers=T) as rank_pool:
            rank_fn = partial(
                pareto_rangovi_i_crowding_pool,
                max_workers=T,
                executor=rank_pool,
            )
            return _optimizuj_nitima_impl(
                oceni_func, B, N, k, M, lam, G, T, seed, pocetna, rank_fn
            )

    return _optimizuj_nitima_impl(
        oceni_func, B, N, k, M, lam, G, T, seed, pocetna, None
    )


def _radna_nit(st, oceni_func, lam, N, rank_fn):
    while True:
        # Snimak populacije pod Lock-om; generisanje je van glavnog Lock-a.
        with st.lock:
            if st.treba_stati():
                st.stop = True
                return
            roditelji = list(st.populacija)

        # Jedini RNG programa, pod sopstvenim lock-om.
        with st.rng_lock:
            kandidat = generisi_kandidata(roditelji, st.rng, lam)

        # Kritična sekcija: provera duplikata, budžeta i upis u evidenciju.
        with st.lock:
            if st.treba_stati():
                st.stop = True
                return

            if kandidat in st.evidencija:
                st.neuspesi += 1
                st.neuspesi_ukupno += 1
                continue

            # Kandidat se prvo upisuje kao poznat / u obradi.
            st.evidencija[kandidat] = None

            # Rezervišemo budžet pre poziva evaluatora.
            st.budzet -= 1
            st.zapocete += 1
            st.in_flight += 1

            # Uspešan kandidat resetuje brojač neuspešnih pokušaja.
            st.neuspesi = 0

            # Iteracija je gotova čim je M rezervisano; niko ne čeka.
            st.zapocete_u_iteraciji += 1
            if st.zapocete_u_iteraciji == st.M:
                sad = time.perf_counter()
                st.trajanja_iteracija.append(sad - st.iter_start)
                st.zapocete_po_iteraciji.append(st.zapocete_u_iteraciji)
                st.iteracije += 1
                st.iter_start = sad
                st.zapocete_u_iteraciji = 0

        # Evaluacija je van lock-a, da ne blokira ostale niti.
        ocene = oceni_func(kandidat)

        # Nit sama spaja svog kandidata, tek kada ima ocene.
        # Zato u rangiranje nikad ne ulazi kandidat sa None ocenama.
        with st.lock:
            st.evidencija[kandidat] = ocene
            st.zavrsene += 1
            st.in_flight -= 1
            st.populacija, st.arhiva, _, _, _ = spoji_i_azuriraj(
                st.populacija,
                st.arhiva,
                [kandidat],
                st.evidencija,
                N,
                rank_fn=rank_fn,
            )


def _optimizuj_nitima_impl(
    oceni_func,
    B,
    N,
    k,
    M,
    lam,
    G,
    T,
    seed,
    pocetna,
    rank_fn,
):
    start = time.perf_counter()

    stats = _novi_stats(T)
    stats["parallel_rank"] = rank_fn is not None

    # Jedan RNG za ceo program.
    rng = np.random.default_rng(seed)

    pocetna_populacija, evidencija = napravi_pocetnu_populaciju(
        pocetna,
        N,
        k,
        rng,
    )

    budzet = B
    populacija = []

    # Početna populacija se evaluira sekvencijalno, pre pokretanja radnih niti.
    for s in pocetna_populacija:
        if budzet <= 0:
            break

        # Kandidat se označava u evidenciji pre evaluacije.
        evidencija[s] = None

        # Budžet se troši kada poziv evaluatora počne.
        budzet -= 1
        stats["evaluacije_zapocete"] += 1

        ocene = oceni_func(s)

        evidencija[s] = ocene
        stats["evaluacije_zavrsene"] += 1

        populacija.append(s)

    if not populacija:
        stats["status"] = "nema_evaluiranih_pocetnih"
        stats["ukupno_vreme"] = time.perf_counter() - start
        return [], evidencija, stats

    # Početni Pareto rangovi i arhiva.
    ranks, crowd, layers = (rank_fn or pareto_rangovi_i_crowding)(populacija, evidencija)
    arhiva = layers[0] if layers else []

    # Neevaluirani početni mutanti (mali budžet) ne smeju ostati sa None.
    for s in pocetna_populacija[len(populacija):]:
        del evidencija[s]

    if budzet <= 0:
        stats["status"] = "budzet_potrosen_posle_pocetne_populacije"
        stats["evaluacije"] = stats["evaluacije_zavrsene"]
        stats["ukupno_vreme"] = time.perf_counter() - start
        return arhiva, evidencija, stats

    # Spajanje dodaje kandidate, pa populacija posle ove provere ne pada ispod 2.
    if len(populacija) < 2:
        stats["status"] = "nema_dovoljno_roditelja"
        stats["evaluacije"] = stats["evaluacije_zavrsene"]
        stats["ukupno_vreme"] = time.perf_counter() - start
        return arhiva, evidencija, stats

    st = DeljenoStanje(populacija, arhiva, evidencija, budzet, M, G, rng)

    # T niti se prave jednom i rade do iscrpljenja budžeta ili praga G.
    niti = [
        threading.Thread(target=_radna_nit, args=(st, oceni_func, lam, N, rank_fn))
        for _ in range(T)
    ]
    for t in niti:
        t.start()
    for t in niti:
        t.join()

    # Proverava se samo jednom, posle join-a svih niti.
    if st.in_flight != 0:
        raise RuntimeError(
            "Nakon join-a svih niti postoje evaluacije koje još nisu završene."
        )

    # Nepotpuna poslednja iteracija se broji ako je počela.
    if st.zapocete_u_iteraciji > 0:
        st.trajanja_iteracija.append(time.perf_counter() - st.iter_start)
        st.zapocete_po_iteraciji.append(st.zapocete_u_iteraciji)
        st.iteracije += 1

    arhiva = st.arhiva

    stats["evaluacije_zapocete"] += st.zapocete
    stats["evaluacije_zavrsene"] += st.zavrsene
    stats["neuspesi_ukupno"] = st.neuspesi_ukupno
    stats["iteracije"] = st.iteracije
    stats["zapocete_po_iteraciji"] = st.zapocete_po_iteraciji
    stats["ukupno_vreme_iteracija"] = sum(st.trajanja_iteracija)

    if st.neuspesi >= G:
        stats["status"] = "prag_neuspesnih_pokusaja_G"
    else:
        stats["status"] = "budzet_potrosen"

    end = time.perf_counter()

    stats["ukupno_vreme"] = end - start
    stats["prosecno_vreme_iteracije"] = (
        stats["ukupno_vreme_iteracija"] / max(1, stats["iteracije"])
    )

    # Posle join-a započete == završene.
    stats["evaluacije"] = stats["evaluacije_zavrsene"]

    return arhiva, evidencija, stats
