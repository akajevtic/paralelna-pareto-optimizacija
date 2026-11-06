# processes.py

import time
from queue import Empty, Full

import numpy as np

from common import (
    POCETNA_SEKVENCA,
    napravi_pocetnu_populaciju,
    generisi_kandidata,
    pareto_rangovi_i_crowding,
    spoji_i_azuriraj,
)


# Topologija razmene; argument komandne linije --topology je prepisuje.
TOPOLOGIJA = "ring"


def lokalni_budzet(rank, P, B):
    """Budžet procesa: B // P, ostatak ide prvim procesima."""
    return B // P + (1 if rank < B % P else 0)


def _primi_poruke(poruke, evidencija, origin_map, stats):
    """
    Upisuje nove kandidate iz poruka u evidenciju, sa ocenama.
    Ne troši budžet. Vraća listu novih kandidata.
    """
    primljeni = []

    for msg in poruke:
        if not isinstance(msg, dict):
            continue

        if msg.get("tip") != "CANDS":
            continue

        stats["primljeno_poruka"] += 1

        for seq, ocene, origin in msg.get("kandidati", []):
            if seq in evidencija:
                continue

            evidencija[seq] = tuple(ocene)
            origin_map[seq] = origin
            primljeni.append(seq)

            stats["primljeno_novih"] += 1

    return primljeni


def _prazni_red(q):
    """
    Bez čekanja prazni red. Vraća listu poruka.
    """
    poruke = []

    while True:
        try:
            poruke.append(q.get_nowait())
        except Empty:
            break
        except Exception:
            break

    return poruke


def _bezbedno_put(q, msg):
    """
    Slanje poruke sa kratkim timeout-om da ne bi došlo do beskonačnog blokiranja.
    """
    try:
        q.put(msg, timeout=0.5)
        return True
    except Full:
        return False
    except Exception:
        return False


def izaberi_susede(rank, P, topology, rng):
    """
    Vraća listu procesa kojima rank šalje poruke.
    """
    if P <= 1:
        return []

    if topology == "ring":
        return [(rank + 1) % P]

    if topology == "star":
        if rank == 0:
            return [i for i in range(P) if i != 0]
        return [0]

    if topology == "all2all":
        return [i for i in range(P) if i != rank]

    if topology == "random":
        candidates = [i for i in range(P) if i != rank]
        return [int(rng.choice(candidates))]

    raise ValueError(f"Nepoznata topologija: {topology}")


def _izaberi_kandidate_za_slanje(arhiva, crowd, evidencija, K, send_selection, rng):
    """
    Bira do K kandidata za slanje.

    - archive: najbolji kandidati iz arhive po crowding distance
    - random-evidence: nasumični kandidati iz evidencije
    """
    if K <= 0:
        return []

    if send_selection == "archive":
        if not arhiva:
            return []

        if len(arhiva) <= K:
            return list(arhiva)

        ordered = sorted(arhiva, key=lambda s: (-crowd.get(s, 0.0), s))
        return ordered[:K]

    if send_selection == "random-evidence":
        seqs = [s for s, v in evidencija.items() if v is not None]

        if not seqs:
            return []

        if len(seqs) <= K:
            return seqs

        idx = rng.choice(len(seqs), size=K, replace=False)
        return [seqs[int(i)] for i in idx]

    return []


def _novi_stats(rank, P, budzet_dodeljen, lam, topology):
    return {
        "rank": rank,
        "P": P,
        "lam": lam,
        "topology": topology,
        "budzet_dodeljen": budzet_dodeljen,
        "budzet_preostalo": budzet_dodeljen,
        "evaluacije_zapocete": 0,
        "evaluacije_zavrsene": 0,
        "iteracije": 0,
        "neuspesi_ukupno": 0,
        "poslato_poruka": 0,
        "poslato_kandidata": 0,
        "primljeno_poruka": 0,
        "primljeno_novih": 0,
        "strani_u_arhivi": 0,
        "status": "",
        "ukupno_vreme": 0.0,
        "ukupno_vreme_iteracija": 0.0,
        "prosecno_vreme_iteracije": 0.0,
    }


def process_worker(
    rank,
    P,
    B,
    N,
    k,
    M,
    lam,
    G,
    K,
    E,
    topology,
    send_selection,
    seed,
    received_resets_G,
    input_queues,
    result_queue,
    stop_event,
):
    """
    Jedan proces radi lokalnu pretragu i razmenjuje kandidate.
    """
    start = time.perf_counter()

    local_budget = lokalni_budzet(rank, P, B)

    stats = _novi_stats(rank, P, local_budget, lam, topology)

    # Lokalno stanje
    rng = None
    evidencija = {}
    origin_map = {}
    populacija = []
    arhiva = []
    neuspesi = 0
    local_iter = 0

    try:
        # Evaluator se učitava u svakom procesu.
        # oceni.py ima printove na dnu, pa će svaki proces nešto odštampati.
        from oceni import oceni as oceni_func

        # Jedan slučajni tok po procesu, i za pretragu i za komunikaciju.
        rng = np.random.default_rng(seed * 10000 + rank)

        # Početna populacija
        pocetna_populacija, evidencija = napravi_pocetnu_populaciju(
            POCETNA_SEKVENCA,
            N,
            k,
            rng,
        )

        for s in pocetna_populacija:
            if local_budget <= 0:
                break

            evidencija[s] = None

            local_budget -= 1
            stats["evaluacije_zapocete"] += 1

            ocene = oceni_func(s)

            evidencija[s] = ocene
            stats["evaluacije_zavrsene"] += 1

            populacija.append(s)
            origin_map[s] = rank

        if populacija:
            ranks, crowd, layers = pareto_rangovi_i_crowding(populacija, evidencija)
            arhiva = layers[0] if layers else []
        else:
            ranks = {}
            crowd = {}
            layers = []
            arhiva = []

        # Glavna lokalna petlja
        while (
            local_budget > 0
            and neuspesi < G
            and len(populacija) >= 2
            and not stop_event.is_set()
        ):
            # Prijem poruka na početku iteracije.
            primljeni = _primi_poruke(
                _prazni_red(input_queues[rank]), evidencija, origin_map, stats
            )

            if primljeni:
                if received_resets_G:
                    neuspesi = 0

                populacija, arhiva, ranks, crowd, layers = spoji_i_azuriraj(
                    populacija,
                    arhiva,
                    primljeni,
                    evidencija,
                    N,
                )

            # Lokalno prikupljanje M kandidata.
            iter_start = time.perf_counter()

            lokalni_novi = []

            while len(lokalni_novi) < M and local_budget > 0 and neuspesi < G:
                kandidat = generisi_kandidata(populacija, rng, lam)

                if kandidat in evidencija:
                    neuspesi += 1
                    stats["neuspesi_ukupno"] += 1
                    continue

                evidencija[kandidat] = None

                local_budget -= 1
                stats["evaluacije_zapocete"] += 1

                ocene = oceni_func(kandidat)

                evidencija[kandidat] = ocene
                stats["evaluacije_zavrsene"] += 1

                lokalni_novi.append(kandidat)
                origin_map[kandidat] = rank

                neuspesi = 0

                # Ažuriranje populacije i arhive posle svake evaluacije.
                populacija, arhiva, ranks, crowd, layers = spoji_i_azuriraj(
                    populacija,
                    arhiva,
                    [kandidat],
                    evidencija,
                    N,
                )

            if lokalni_novi or primljeni:
                local_iter += 1
                stats["iteracije"] += 1
                stats["ukupno_vreme_iteracija"] += time.perf_counter() - iter_start

                # Slanje nakon svake E-te lokalne iteracije.
                if E > 0 and local_iter % E == 0:
                    susedi = izaberi_susede(rank, P, topology, rng)

                    if susedi:
                        izabrani = _izaberi_kandidate_za_slanje(
                            arhiva,
                            crowd,
                            evidencija,
                            K,
                            send_selection,
                            rng,
                        )

                        if izabrani:
                            payload = []

                            for s in izabrani:
                                payload.append(
                                    (
                                        s,
                                        tuple(evidencija[s]),
                                        origin_map.get(s, rank),
                                    )
                                )

                            msg = {
                                "tip": "CANDS",
                                "from": rank,
                                "iter": local_iter,
                                "kandidati": payload,
                            }

                            for dest in susedi:
                                if _bezbedno_put(input_queues[dest], msg):
                                    stats["poslato_poruka"] += 1
                                    stats["poslato_kandidata"] += len(payload)
            else:
                # Nema ni lokalnih ni primljenih kandidata.
                break

        if neuspesi >= G:
            stats["status"] = "prag_neuspesnih_pokusaja_G"
        elif local_budget <= 0:
            stats["status"] = "budzet_potrosen"
        elif len(populacija) < 2:
            stats["status"] = "nema_dovoljno_roditelja"
        elif stop_event.is_set():
            stats["status"] = "global_stop"
        else:
            stats["status"] = "zavrseno"

    except Exception as e:
        stats["status"] = f"greska: {e}"

    finally:
        stats["budzet_preostalo"] = local_budget
        stats["ukupno_vreme"] = time.perf_counter() - start
        stats["prosecno_vreme_iteracije"] = (
            stats["ukupno_vreme_iteracija"] / max(1, stats["iteracije"])
        )

        stats["strani_u_arhivi"] = sum(
            1 for s in arhiva if origin_map.get(s, rank) != rank
        )

        archive_payload = []

        for s in arhiva:
            if evidencija.get(s) is not None:
                archive_payload.append(
                    (
                        s,
                        tuple(evidencija[s]),
                        origin_map.get(s, rank),
                    )
                )

        try:
            result_queue.put(
                {
                    "rank": rank,
                    "stats": stats,
                    "archive": archive_payload,
                }
            )
        except Exception:
            pass

        # Proces posle svog rezultata još malo prazni svoj red,
        # da drugi procesi ne bi blokirali slanjem prema njemu.
        while not stop_event.wait(timeout=0.05):
            _prazni_red(input_queues[rank])


def _spoji_rezultate(results):
    """
    Spaja lokalne arhive procesa u jednu globalnu Pareto arhivu.
    """
    seq_scores = {}
    origins = {}
    per_process = {}

    agg = {
        "broj_procesa_sa_rezultatom": 0,
        "evaluacije_zapocete": 0,
        "evaluacije_zavrsene": 0,
        "iteracije": 0,
        "neuspesi_ukupno": 0,
        "poslato_poruka": 0,
        "poslato_kandidata": 0,
        "primljeno_poruka": 0,
        "primljeno_novih": 0,
        "strani_u_arhivi_sum": 0,
        "velicina_arhive": 0,
    }

    for rank in sorted(results.keys()):
        payload = results[rank]
        stats = payload.get("stats", {})
        per_process[rank] = stats

        agg["broj_procesa_sa_rezultatom"] += 1
        agg["evaluacije_zapocete"] += stats.get("evaluacije_zapocete", 0)
        agg["evaluacije_zavrsene"] += stats.get("evaluacije_zavrsene", 0)
        agg["iteracije"] += stats.get("iteracije", 0)
        agg["neuspesi_ukupno"] += stats.get("neuspesi_ukupno", 0)
        agg["poslato_poruka"] += stats.get("poslato_poruka", 0)
        agg["poslato_kandidata"] += stats.get("poslato_kandidata", 0)
        agg["primljeno_poruka"] += stats.get("primljeno_poruka", 0)
        agg["primljeno_novih"] += stats.get("primljeno_novih", 0)
        agg["strani_u_arhivi_sum"] += stats.get("strani_u_arhivi", 0)

        for seq, scores, origin in payload.get("archive", []):
            if seq not in seq_scores:
                seq_scores[seq] = tuple(scores)
                origins[seq] = origin

    if seq_scores:
        ranks, crowd, layers = pareto_rangovi_i_crowding(
            list(seq_scores.keys()),
            seq_scores,
        )
        final_archive = layers[0] if layers else []
    else:
        final_archive = []

    agg["velicina_arhive"] = len(final_archive)

    if agg["broj_procesa_sa_rezultatom"] > 0:
        agg["strani_u_arhivi_avg"] = (
            agg["strani_u_arhivi_sum"] / agg["broj_procesa_sa_rezultatom"]
        )
    else:
        agg["strani_u_arhivi_avg"] = 0.0

    return {
        "final_archive": final_archive,
        "seq_scores": seq_scores,
        "origins": origins,
        "per_process": per_process,
        "agg": agg,
    }


def pokreni_procese(
    B,
    N,
    k,
    M,
    lam_list,
    G,
    K,
    E,
    topology,
    send_selection,
    P,
    seed,
    received_resets_G=False,
    timeout=1200,
):
    """
    Pokreće P procesa, čeka rezultate i spaja konačne arhive.
    """
    import multiprocessing as mp

    if P <= 0:
        raise ValueError("P mora biti veći od 0.")

    topology = topology or TOPOLOGIJA

    if len(lam_list) == 1:
        lam_per_process = [lam_list[0] for _ in range(P)]
    elif len(lam_list) == P:
        lam_per_process = lam_list
    else:
        raise ValueError(
            "Lista lambda vrednosti mora imati 1 element ili tačno P elemenata."
        )

    input_queues = [mp.Queue() for _ in range(P)]
    result_queue = mp.Queue()
    stop_event = mp.Event()

    start = time.perf_counter()

    procs = []

    for rank in range(P):
        p = mp.Process(
            target=process_worker,
            args=(
                rank,
                P,
                B,
                N,
                k,
                M,
                lam_per_process[rank],
                G,
                K,
                E,
                topology,
                send_selection,
                seed,
                received_resets_G,
                input_queues,
                result_queue,
                stop_event,
            ),
        )
        p.start()
        procs.append(p)

    results = {}
    deadline = time.time() + timeout

    while len(results) < P and time.time() < deadline:
        try:
            payload = result_queue.get(timeout=1.0)
            results[payload["rank"]] = payload
        except Empty:
            # Ako nijedan proces više nije živ, pokušaj da pokupiš preostale poruke.
            if not any(p.is_alive() for p in procs):
                while True:
                    try:
                        payload = result_queue.get_nowait()
                        results[payload["rank"]] = payload
                    except Empty:
                        break
                break

    # Signal procesima da mogu da završe drain fazu.
    stop_event.set()

    for p in procs:
        p.join(timeout=5)

    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)

    total_time = time.perf_counter() - start

    merged = _spoji_rezultate(results)
    merged["ukupno_vreme"] = total_time
    merged["missing_ranks"] = [r for r in range(P) if r not in results]

    return merged