# pareto_parallel.py

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Sequence, Tuple

from common import (
    dominira,
    crowding_distance,
    pareto_rangovi_i_crowding,
    spoji_i_azuriraj,
)


# Za male skupove paralelizacija se obično ne isplati.
# Ovo je prag koji možeš menjati.
MIN_PARALELNA_VELICINA = 32


def _dominirani_chunk(chunk: Sequence[str], preostali: Sequence[str], scores: Dict[str, Tuple[float, float, float]]) -> List[bool]:
    """
    Za chunk kandidata proverava da li je svaki kandidat dominiran
    od strane nekog drugog kandidata iz skupa preostali.
    """
    rezultat = []

    for s in chunk:
        s_scores = scores[s]

        dominiran = False

        for t in preostali:
            if s == t:
                continue

            if dominira(scores[t], s_scores):
                dominiran = True
                break

        rezultat.append(dominiran)

    return rezultat


def _napravi_chunkove(lista: Sequence[str], max_workers: int) -> List[List[str]]:
    """
    Deli listu na približno max_workers chunk-ova.
    """
    n = len(lista)

    if n == 0:
        return []

    if max_workers <= 1:
        return [list(lista)]

    chunk_size = max(1, (n + max_workers - 1) // max_workers)

    chunkovi = []

    for i in range(0, n, chunk_size):
        chunkovi.append(list(lista[i:i + chunk_size]))

    return chunkovi


def _pareto_pool(
    unique: Sequence[str],
    scores: Dict[str, Tuple[float, float, float]],
    executor: ThreadPoolExecutor,
    max_workers: int,
) -> Tuple[Dict[str, int], Dict[str, float], List[List[str]]]:
    """
    Pareto rangiranje koje koristi thread pool za proveru dominacije.
    """
    preostali = list(unique)

    ranks: Dict[str, int] = {}
    crowd_all: Dict[str, float] = {}
    layers: List[List[str]] = []

    rank = 1

    while preostali:
        chunkovi = _napravi_chunkove(preostali, max_workers)

        # executor.map čuva redosled rezultata po chunk-ovima.
        chunk_rezultati = executor.map(
            lambda chunk: _dominirani_chunk(chunk, preostali, scores),
            chunkovi,
        )

        # Spajamo rezultate iz svih chunk-ova u jednu listu istog redosleda kao preostali.
        flags: List[bool] = []
        for deo in chunk_rezultati:
            flags.extend(deo)

        front = [
            s
            for s, dominiran in zip(preostali, flags)
            if not dominiran
        ]

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


def pareto_rangovi_i_crowding_pool(
    seqs: Sequence[str],
    scores: Dict[str, Tuple[float, float, float]],
    max_workers: int = 1,
    executor: ThreadPoolExecutor = None,
    min_size: int = MIN_PARALELNA_VELICINA,
) -> Tuple[Dict[str, int], Dict[str, float], List[List[str]]]:
    """
    Verzija Pareto rangiranja koja može koristiti thread pool.

    Ako je max_workers <= 1 ili je skup premali, koristi se sekvencijalna verzija.
    Ako executor nije dat, funkcija sama pravi privremeni thread pool.
    """
    unique = list(dict.fromkeys(seqs))

    if max_workers <= 1 or len(unique) < min_size:
        return pareto_rangovi_i_crowding(unique, scores)

    if executor is None:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return _pareto_pool(unique, scores, ex, max_workers)

    return _pareto_pool(unique, scores, executor, max_workers)


def spoji_i_azuriraj_pool(
    populacija: Sequence[str],
    arhiva: Sequence[str],
    novi: Sequence[str],
    scores: Dict[str, Tuple[float, float, float]],
    N: int,
    max_workers: int = 1,
    executor: ThreadPoolExecutor = None,
    min_size: int = MIN_PARALELNA_VELICINA,
):
    """Verzija spoji_i_azuriraj koja koristi paralelno Pareto rangiranje."""
    rank_fn = partial(
        pareto_rangovi_i_crowding_pool,
        max_workers=max_workers,
        executor=executor,
        min_size=min_size,
    )
    return spoji_i_azuriraj(
        populacija, arhiva, novi, scores, N, rank_fn=rank_fn
    )