# common.py

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
ALT_AA = {a: tuple(x for x in AA if x != a) for a in AA}

POCETNA_SEKVENCA = "TEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTN"

# Prag uzastopnih neuspešnih generisanja; postavka: G se zadaje u kodu.
G = 100


def proveri_sekvencu(seq: str, duzina: Optional[int] = None) -> bool:
    """Proveri da sekvenca ima dobru dužinu i samo dozvoljene aminokiseline."""
    if duzina is None:
        duzina = len(POCETNA_SEKVENCA)
    return len(seq) == duzina and set(seq) <= AA_SET


def dominira(a: Sequence[float], b: Sequence[float]) -> bool:
    """
    A dominira B ako A nije lošiji od B ni po jednom kriterijumu,
    a bolji je bar po jednom.
    Maksimizujemo sve ocene.
    """
    bolji = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            bolji = True
    return bolji


def crowding_distance(front: Sequence[str], scores: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
    """
    Mera razređenosti za jedan Pareto sloj.
    Veća vrednost = kandidat je u ređe zastupljenom delu prostora ocena.
    """
    crowd = {s: 0.0 for s in front}
    n = len(front)

    if n == 0:
        return crowd

    # Ako ima 1 ili 2 kandidata, svi su granični.
    if n <= 2:
        for s in front:
            crowd[s] = float("inf")
        return crowd

    for j in range(3):
        # Sortiramo po j-toj oceni; sekvenca kao tie-breaker zbog determinizma.
        sorted_front = sorted(front, key=lambda s: (scores[s][j], s))

        lo = scores[sorted_front[0]][j]
        hi = scores[sorted_front[-1]][j]

        # Krajnji kandidati dobijaju beskonačnu meru.
        crowd[sorted_front[0]] = float("inf")
        crowd[sorted_front[-1]] = float("inf")

        # Ako je raspon nula, preskačemo ovaj kriterijum.
        if hi - lo <= 1e-12:
            continue

        for i in range(1, n - 1):
            s = sorted_front[i]

            # Ako je već beskonačan (npr. bio krajnji u nekom drugom kriterijumu), ne diramo.
            if crowd[s] == float("inf"):
                continue

            crowd[s] += (
                scores[sorted_front[i + 1]][j] - scores[sorted_front[i - 1]][j]
            ) / (hi - lo)

    return crowd


def pareto_rangovi_i_crowding(
    seqs: Sequence[str],
    scores: Dict[str, Tuple[float, float, float]],
) -> Tuple[Dict[str, int], Dict[str, float], List[List[str]]]:
    """
    Deli kandidate u Pareto slojeve.
    Vraća:
      - ranks: seq -> rank
      - crowd_all: seq -> crowding distance
      - layers: lista slojeva, layers[0] je Pareto front
    """
    unique = list(dict.fromkeys(seqs))
    preostali = unique[:]

    ranks: Dict[str, int] = {}
    crowd_all: Dict[str, float] = {}
    layers: List[List[str]] = []

    rank = 1

    while preostali:
        front = []

        for s in preostali:
            s_scores = scores[s]
            if s_scores is None:
                raise ValueError(f"Kandidat {s} nema ocene prilikom Pareto rangiranja.")

            dominated = False

            for t in preostali:
                if s == t:
                    continue

                t_scores = scores[t]
                if t_scores is None:
                    raise ValueError(f"Kandidat {t} nema ocene prilikom Pareto rangiranja.")

                if dominira(t_scores, s_scores):
                    dominated = True
                    break

            if not dominated:
                front.append(s)

        if not front:
            raise RuntimeError("Pareto rangiranje nije pronašlo nijedan nedominirani kandidat.")

        layer_crowd = crowding_distance(front, scores)

        for s in front:
            ranks[s] = rank
            crowd_all[s] = layer_crowd[s]

        layers.append(front)

        front_set = set(front)
        preostali = [s for s in preostali if s not in front_set]

        rank += 1

    return ranks, crowd_all, layers


def napravi_pocetnog_mutanta(seq: str, k: int, rng) -> str:
    """
    Početni mutant: menja između 1 i k pozicija u odnosu na seq.
    """
    L = len(seq)

    if k <= 0:
        return seq

    max_promena = min(k, L)
    broj_promena = int(rng.integers(1, max_promena + 1))

    pozicije = rng.choice(L, size=broj_promena, replace=False)

    lista = list(seq)

    for p in pozicije:
        p = int(p)
        stari = lista[p]
        lista[p] = str(rng.choice(ALT_AA[stari]))

    return "".join(lista)


def napravi_pocetnu_populaciju(
    pocetna: str,
    N: int,
    k: int,
    rng,
) -> Tuple[List[str], Dict[str, Optional[Tuple[float, float, float]]]]:
    """
    Početna populacija: početna sekvenca + N-1 jedinstvenih početnih mutanata.
    Ovde još ne evaluiramo; evidencija samo označava sekvence.
    """
    if N <= 0:
        return [], {}

    if k <= 0 and N > 1:
        raise ValueError("Za k <= 0 nema mutanata; N mora biti 1.")

    populacija = [pocetna]
    evidencija: Dict[str, Optional[Tuple[float, float, float]]] = {pocetna: None}

    pokusaji = 0
    max_pokusaja = 100 * N

    while len(populacija) < N:
        pokusaji += 1
        if pokusaji > max_pokusaja:
            raise RuntimeError(
                f"Napravljeno samo {len(populacija)} od {N} jedinstvenih mutanata; povećaj k ili smanji N."
            )

        mutant = napravi_pocetnog_mutanta(pocetna, k, rng)

        if mutant not in evidencija:
            evidencija[mutant] = None
            populacija.append(mutant)

    return populacija, evidencija


def generisi_kandidata(populacija: Sequence[str], rng, lam: float) -> str:
    """
    Generiše jednog kandidata:
      1. bira dva različita roditelja iz populacije
      2. ukršta ih na jednom mestu
      3. mutira broj pozicija iz Poissonove raspodele sa parametrom lam
    """
    if len(populacija) < 2:
        raise ValueError("Populacija mora imati bar 2 kandidata za generisanje.")

    idx = rng.choice(len(populacija), size=2, replace=False)
    p1 = populacija[int(idx[0])]
    p2 = populacija[int(idx[1])]

    L = len(p1)

    # Mesto preseka između dve aminokiseline: 1..L-1
    cut = int(rng.integers(1, L))

    child = p1[:cut] + p2[cut:]

    n_mut = int(rng.poisson(lam))

    if n_mut > 0:
        n_mut = min(n_mut, L)
        pozicije = rng.choice(L, size=n_mut, replace=False)

        lista = list(child)

        for p in pozicije:
            p = int(p)
            stari = lista[p]
            lista[p] = str(rng.choice(ALT_AA[stari]))

        child = "".join(lista)

    return child


def spoji_i_azuriraj(
    populacija: Sequence[str],
    arhiva: Sequence[str],
    novi: Sequence[str],
    scores: Dict[str, Tuple[float, float, float]],
    N: int,
    rank_fn=None,
) -> Tuple[List[str], List[str], Dict[str, int], Dict[str, float], List[List[str]]]:
    """
    Spaja populaciju, arhivu i nove kandidate, zatim:
      - rangira ih Pareto slojevima
      - pravi novu arhivu od prvog sloja
      - puni novu populaciju do N kandidata
    """
    skup = list(dict.fromkeys(populacija + arhiva + novi))

    if rank_fn is None:
        rank_fn = pareto_rangovi_i_crowding

    ranks, crowd, layers = rank_fn(skup, scores)

    nova_arhiva = layers[0] if layers else []

    nova_populacija: List[str] = []

    for layer in layers:
        # Deterministički poredak: veći crowding ranije, onda leksikografski.
        ordered = sorted(layer, key=lambda s: (-crowd[s], s))

        if len(nova_populacija) + len(ordered) <= N:
            nova_populacija.extend(ordered)
        else:
            need = N - len(nova_populacija)
            if need > 0:
                nova_populacija.extend(ordered[:need])
            break

    return nova_populacija, nova_arhiva, ranks, crowd, layers


def optimizuj_sekvencijalno(
    oceni_func,
    B: int,
    N: int,
    k: int,
    M: int,
    lam: float,
    G: int,
    seed: int,
    pocetna: str = POCETNA_SEKVENCA,
) -> Tuple[List[str], Dict[str, Optional[Tuple[float, float, float]]], Dict]:
    """
    Sekvencijalna verzija optimizacije.
    """
    start = time.perf_counter()

    stats = {
        "evaluacije": 0,
        "iteracije": 0,
        "neuspesi_ukupno": 0,
        "status": "",
        "ukupno_vreme": 0.0,
        "prosecno_vreme_iteracije": 0.0,
    }

    if B <= 0 or N <= 0 or M <= 0:
        stats["status"] = "losi_parametri"
        return [], {}, stats

    rng = np.random.default_rng(seed)

    pocetna_populacija, evidencija = napravi_pocetnu_populaciju(pocetna, N, k, rng)

    populacija: List[str] = []

    # Evaluacija početne populacije, uz budžet.
    for s in pocetna_populacija:
        if stats["evaluacije"] >= B:
            break

        evidencija[s] = oceni_func(s)
        stats["evaluacije"] += 1
        populacija.append(s)

    if not populacija:
        stats["status"] = "nema_evaluiranih_pocetnih"
        stats["ukupno_vreme"] = time.perf_counter() - start
        return [], evidencija, stats

    # Početni Pareto rangovi i arhiva.
    ranks, crowd, layers = pareto_rangovi_i_crowding(populacija, evidencija)
    arhiva = layers[0] if layers else []

    if stats["evaluacije"] >= B:
        stats["status"] = "budzet_potrosen_posle_pocetne_populacije"
        stats["ukupno_vreme"] = time.perf_counter() - start
        return arhiva, evidencija, stats

    neuspesi = 0

    while stats["evaluacije"] < B:
        if len(populacija) < 2:
            stats["status"] = "nema_dovoljno_roditelja"
            break

        novi: List[str] = []

        # Prikupljamo M novih ocenjenih kandidata.
        while len(novi) < M and stats["evaluacije"] < B and neuspesi < G:
            kandidat = generisi_kandidata(populacija, rng, lam)

            if kandidat in evidencija:
                neuspesi += 1
                stats["neuspesi_ukupno"] += 1
                continue

            # Označavamo da je kandidat poznat / u obradi.
            # U sekvencijalnoj verziji odmah evaluiramo, pa None traje zanemarljivo.
            evidencija[kandidat] = None

            ocene = oceni_func(kandidat)
            evidencija[kandidat] = ocene

            stats["evaluacije"] += 1
            novi.append(kandidat)

            # Uspešan kandidat resetuje brojač neuspešnih pokušaja.
            neuspesi = 0

        # Ažuriranje populacije i arhive ako ima novih kandidata.
        if novi:
            populacija, arhiva, ranks, crowd, layers = spoji_i_azuriraj(
                populacija,
                arhiva,
                novi,
                evidencija,
                N,
            )
            stats["iteracije"] += 1

        # Uslovi zaustavljanja.
        if neuspesi >= G:
            stats["status"] = "prag_neuspesnih_pokusaja_G"
            break

        if stats["evaluacije"] >= B:
            stats["status"] = "budzet_potrosen"
            break

        if not novi:
            stats["status"] = "nema_novih_kandidata"
            break

    if not stats["status"]:
        stats["status"] = "budzet_potrosen"

    end = time.perf_counter()
    stats["ukupno_vreme"] = end - start
    stats["prosecno_vreme_iteracije"] = stats["ukupno_vreme"] / max(1, stats["iteracije"])

    return arhiva, evidencija, stats