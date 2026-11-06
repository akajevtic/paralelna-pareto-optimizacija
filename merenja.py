# merenja.py

import argparse
import json
import os
import platform
import statistics
import time

import numpy

import sys

from oceni import oceni

from common import G, optimizuj_sekvencijalno
from threaded import optimizuj_nitima
from processes import pokreni_procese
from hypervolume import hypervolume_arhive


def default_config():
    return {
        "B": 2000,
        "N": 80,
        "k": 5,
        "M": 40,
        "G": G,
        "K": 5,
        "E": 2,
        "E_P6": 1,
        "P": 4,
        "T": 4,
        "lam_common": [2.0],
        "lam_diff": [0.5, 1.0, 2.0, 4.0],
        "topology_p5": "star",
    }


def quick_config(cfg):
    cfg = dict(cfg)
    cfg.update(
        {
            "B": 400,
            "N": 30,
            "M": 15,
            "K": 3,
            "E": 1,
            "E_P6": 2,
        }
    )
    return cfg


def _score_ranges(ocene):
    if not ocene:
        return {"min": None, "max": None}

    return {
        "min": [min(o[j] for o in ocene) for j in range(3)],
        "max": [max(o[j] for o in ocene) for j in range(3)],
    }


def run_p1(cfg, seed):
    arhiva, evidencija, stats = optimizuj_sekvencijalno(
        oceni_func=oceni,
        B=cfg["B"],
        N=cfg["N"],
        k=cfg["k"],
        M=cfg["M"],
        lam=cfg["lam_common"][0],
        G=cfg["G"],
        seed=seed,
    )

    ocene = [tuple(evidencija[s]) for s in arhiva if evidencija.get(s) is not None]

    return {
        "config": "P1",
        "seed": seed,
        "topology": None,
        "E": None,
        "lam": cfg["lam_common"][0],
        "vreme": stats.get("ukupno_vreme", 0.0),
        "evaluacije": stats.get("evaluacije", len(ocene)),
        "iteracije": stats.get("iteracije", 0),
        "neuspesi": stats.get("neuspesi_ukupno", 0),
        "vreme_iteracije": stats.get("prosecno_vreme_iteracije", 0.0),
        "velicina_arhive": len(arhiva),
        "hypervolume": hypervolume_arhive(ocene),
        "score_ranges": _score_ranges(ocene),
        "archive": [(s, tuple(evidencija[s])) for s in arhiva],
        "raw_stats": stats,
    }


def run_p2(cfg, seed, parallel_rank=False, config_name="P2"):
    arhiva, evidencija, stats = optimizuj_nitima(
        oceni_func=oceni,
        B=cfg["B"],
        N=cfg["N"],
        k=cfg["k"],
        M=cfg["M"],
        lam=cfg["lam_common"][0],
        G=cfg["G"],
        T=cfg["T"],
        seed=seed,
        parallel_rank=parallel_rank,
    )

    ocene = [tuple(evidencija[s]) for s in arhiva if evidencija.get(s) is not None]

    return {
        "config": config_name,
        "seed": seed,
        "topology": None,
        "E": None,
        "lam": cfg["lam_common"][0],
        "T": cfg["T"],
        "vreme": stats.get("ukupno_vreme", 0.0),
        "evaluacije": stats.get("evaluacije", len(ocene)),
        "iteracije": stats.get("iteracije", 0),
        "neuspesi": stats.get("neuspesi_ukupno", 0),
        "vreme_iteracije": stats.get("prosecno_vreme_iteracije", 0.0),
        "velicina_arhive": len(arhiva),
        "hypervolume": hypervolume_arhive(ocene),
        "score_ranges": _score_ranges(ocene),
        "archive": [(s, tuple(evidencija[s])) for s in arhiva],
        "raw_stats": stats,
    }


def run_process_config(cfg, seed, config_name, topology, lam_list, E):
    rez = pokreni_procese(
        B=cfg["B"],
        N=cfg["N"],
        k=cfg["k"],
        M=cfg["M"],
        lam_list=lam_list,
        G=cfg["G"],
        K=cfg["K"],
        E=E,
        topology=topology,
        send_selection="random-evidence",
        P=cfg["P"],
        seed=seed,
        received_resets_G=True,
        timeout=3600,
    )

    # Neispravno pokretanje ne sme ući u tabelu kao ispravno.
    if rez.get("missing_ranks"):
        raise RuntimeError(f"{config_name} seed={seed}: nema rezultata od procesa {rez['missing_ranks']}")
    for r, st in rez.get("per_process", {}).items():
        if str(st.get("status", "")).startswith("greska"):
            raise RuntimeError(f"{config_name} seed={seed}, proces {r}: {st['status']}")

    agg = rez.get("agg", {})
    final_archive = rez.get("final_archive", [])
    seq_scores = rez.get("seq_scores", {})
    origins = rez.get("origins", {})

    ocene = [tuple(seq_scores[s]) for s in final_archive]

    return {
        "config": config_name,
        "seed": seed,
        "topology": topology,
        "E": E,
        "lam": ",".join(str(x) for x in lam_list),
        "vreme": rez.get("ukupno_vreme", 0.0),
        "evaluacije": agg.get("evaluacije_zavrsene", 0),
        "iteracije": agg.get("iteracije", 0),
        "neuspesi": agg.get("neuspesi_ukupno", 0),
        "vreme_iteracije": _mean(
            [st.get("prosecno_vreme_iteracije", 0.0) for st in rez.get("per_process", {}).values()]
        ),
        "velicina_arhive": agg.get("velicina_arhive", len(ocene)),
        "hypervolume": hypervolume_arhive(ocene),
        "score_ranges": _score_ranges(ocene),
        "poslato_poruka": agg.get("poslato_poruka", 0),
        "poslato_kandidata": agg.get("poslato_kandidata", 0),
        "primljeno_poruka": agg.get("primljeno_poruka", 0),
        "primljeno_novih": agg.get("primljeno_novih", 0),
        "strani_avg": agg.get("strani_u_arhivi_avg", 0.0),
        "missing_ranks": rez.get("missing_ranks", []),
        "per_process": rez.get("per_process", {}),
        "archive": [
            (s, tuple(seq_scores[s]), origins.get(s))
            for s in final_archive
        ],
    }


def run_p3(cfg, seed):
    return run_process_config(
        cfg,
        seed,
        "P3",
        "ring",
        cfg["lam_common"],
        cfg["E"],
    )


def run_p4(cfg, seed):
    return run_process_config(
        cfg,
        seed,
        "P4",
        "ring",
        cfg["lam_diff"],
        cfg["E"],
    )


def run_p5(cfg, seed):
    return run_process_config(
        cfg,
        seed,
        "P5",
        cfg["topology_p5"],
        cfg["lam_diff"],
        cfg["E"],
    )


def run_p6(cfg, seed):
    return run_process_config(
        cfg,
        seed,
        "P6",
        "ring",
        cfg["lam_diff"],
        cfg["E_P6"],
    )

def run_p8(cfg, seed):
    return run_process_config(
        cfg,
        seed,
        "P8",
        "all2all",
        cfg["lam_diff"],
        cfg["E"],
    )

def _mean(vals):
    return statistics.mean(vals) if vals else 0.0


def _stdev(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


# Parovi iz odeljka „Poređenje“ postavke: (A, B, šta se menja).
POREDJENJA = [
    ("P2", "P1", "T niti u jednom procesu"),
    ("P3", "P2", "procesi umesto niti (isti λ)"),
    ("P4", "P3", "različite λ po procesima"),
    ("P5", "P4", "druga topologija"),
    ("P6", "P4", "druga učestalost razmene E"),
]

# Tumačenje se upisuje ručno, posle merenja.
TUMACENJE = {
    'P2-P1': 'Niti 2.65x brže: evaluacija (time.sleep) pušta GIL. Efikasnost 0.66 ograničavaju serijalizovan RNG i rangiranje pod Lock-om posle svake evaluacije. HV isti u okviru rasipanja.',
    'P3-P2': 'Procesi ~8% brži od niti: nema GIL-a ni zajedničkog Lock-a. Razlika u HV je manja od rasipanja kroz semena.',
    'P4-P3': 'Vreme isto. HV niži za 0.028, ali to je manje od rasipanja (~0.04); sa 3 semena efekat različitih λ se ne vidi.',
    'P5-P4': 'Star šalje 50% više poruka, vreme isto. HV veći za 0.035, u okviru rasipanja.',
    'P6-P4': 'E=1 udvostručuje razmenu, vreme isto. HV veći za 0.045, granično u odnosu na rasipanje.',
}


def summarize(results, cfg):
    configs = sorted({r["config"] for r in results}, key=lambda x: (int(x[1]), x))

    summary = {}

    for c in configs:
        rows = [r for r in results if r["config"] == c]

        summary[c] = {
            "broj_seedova": len(rows),
            "vreme_avg": _mean([r["vreme"] for r in rows]),
            "vreme_stdev": _stdev([r["vreme"] for r in rows]),
            "vreme_min": min(r["vreme"] for r in rows) if rows else 0.0,
            "vreme_max": max(r["vreme"] for r in rows) if rows else 0.0,
            "hv_avg": _mean([r["hypervolume"] for r in rows]),
            "hv_stdev": _stdev([r["hypervolume"] for r in rows]),
            "arhiva_avg": _mean([r["velicina_arhive"] for r in rows]),
            "evaluacije_avg": _mean([r["evaluacije"] for r in rows]),
            "iteracija_avg": _mean([r["vreme_iteracije"] for r in rows]),
        }

        if "poslato_kandidata" in rows[0]:
            summary[c]["poslato_kandidata_avg"] = _mean(
                [r.get("poslato_kandidata", 0) for r in rows]
            )
            summary[c]["primljeno_novih_avg"] = _mean(
                [r.get("primljeno_novih", 0) for r in rows]
            )
            summary[c]["strani_avg"] = _mean(
                [r.get("strani_avg", 0.0) for r in rows]
            )
            summary[c]["poslato_poruka_avg"] = _mean(
                [r.get("poslato_poruka", 0) for r in rows]
            )
            summary[c]["primljeno_poruka_avg"] = _mean(
                [r.get("primljeno_poruka", 0) for r in rows]
            )

    t1 = summary.get("P1", {}).get("vreme_avg")
    for c, s in summary.items():
        if c == "P1":
            jedinice = 1
        elif c.startswith("P2"):
            jedinice = cfg["T"]
        else:
            jedinice = cfg["P"]
        if t1 and s["vreme_avg"] > 0:
            s["ubrzanje"] = t1 / s["vreme_avg"]
            s["efikasnost"] = s["ubrzanje"] / jedinice
        else:
            s["ubrzanje"] = None
            s["efikasnost"] = None

    return summary


def _fmt(x, nd=3):
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def write_markdown(summary, results, cfg, seeds, filename="rezultati.md"):
    lines = []

    lines.append("# Rezultati P1-P6 (+ dodaci P2-pool, P8)")
    lines.append("")
    lines.append(
        "Propisani P2 rangira sekvencijalno. P2-pool (rangiranje kroz ThreadPoolExecutor) "
        "i P8 (all2all) su dodatni eksperimenti van P1-P6."
    )
    lines.append("")

    lines.append("## Okruženje")
    lines.append("")
    lines.append(f"- CPU: {platform.processor()}")
    lines.append(f"- Logičkih procesora: {os.cpu_count()}")
    lines.append(f"- Python: {platform.python_version()}")
    lines.append(f"- GIL uključen: {getattr(sys, '_is_gil_enabled', lambda: True)()}")
    lines.append(f"- numpy: {numpy.__version__}")
    lines.append(f"- OS: {platform.platform()}")
    lines.append("")

    lines.append("## Parametri")
    lines.append("")
    lines.append(f"- B: {cfg['B']}")
    lines.append(f"- N: {cfg['N']}")
    lines.append(f"- k: {cfg['k']}")
    lines.append(f"- M: {cfg['M']}")
    lines.append(f"- G: {cfg['G']}")
    lines.append(f"- K: {cfg['K']}")
    lines.append(f"- E za P3/P4/P5: {cfg['E']}")
    lines.append(f"- E za P6: {cfg['E_P6']}")
    lines.append(f"- P za P3-P6: {cfg['P']}")
    lines.append(f"- T za P2 i P2-pool: {cfg['T']}")
    lines.append("- Razmena u P3-P8: nasumični kandidati iz evidencije; primljen nov kandidat resetuje G")
    lines.append(f"- λ za P1/P2/P3: {cfg['lam_common']}")
    lines.append(f"- λ za P4/P5/P6: {cfg['lam_diff']}")
    lines.append(f"- Seeds: {seeds}")
    lines.append("")

    lines.append("## Prosek kroz semena")
    lines.append("")
    lines.append(
            "| Config | Vreme avg (s) | Vreme stdev |   Ubrzanje | Efikasnost | Iteracija avg (s) | HV avg | HV stdev | Arhiva avg | Evaluacije avg | Poslato poruka | Primljeno poruka | Poslato kand. | Primljeno novih | Strani avg |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for c, s in summary.items():
        poslato = s.get("poslato_kandidata_avg")
        primljeno = s.get("primljeno_novih_avg")
        strani = s.get("strani_avg")

        poslato_por = s.get("poslato_poruka_avg")
        primljeno_por = s.get("primljeno_poruka_avg")

        lines.append(
            f"| {c} "
            f"| {_fmt(s['vreme_avg'])} "
            f"| {_fmt(s['vreme_stdev'])} "
            f"| {_fmt(s.get('ubrzanje'), 2)} "
            f"| {_fmt(s.get('efikasnost'), 2)} "
            f"| {_fmt(s['iteracija_avg'], 4)} "
            f"| {_fmt(s['hv_avg'], 6)} "
            f"| {_fmt(s['hv_stdev'], 6)} "
            f"| {_fmt(s['arhiva_avg'], 2)} "
            f"| {_fmt(s['evaluacije_avg'], 1)} "
            f"| {_fmt(poslato_por, 1)} "
            f"| {_fmt(primljeno_por, 1)} "
            f"| {_fmt(poslato, 1)} "
            f"| {_fmt(primljeno, 1)} "
            f"| {_fmt(strani, 2)} |"
        )

    lines.append("")

    lines.append("## Po pokretanjima")
    lines.append("")
    lines.append(
        "| Config | Seed | Topology | E | λ | Vreme (s) | HV | Arhiva | Evaluacije | Poslato poruka | Primljeno poruka | Poslato kand. | Primljeno novih |"
    )
    lines.append("|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in results:
        topology = r.get("topology") or "-"
        e = r.get("E")
        e_txt = "-" if e is None else str(e)
        lam = r.get("lam", "-")
        poslato = r.get("poslato_kandidata")
        primljeno = r.get("primljeno_novih")

        poslato_txt = "-" if poslato is None else str(poslato)
        primljeno_txt = "-" if primljeno is None else str(primljeno)

        por_s = r.get("poslato_poruka")
        por_p = r.get("primljeno_poruka")

        lines.append(
            f"| {r['config']} "
            f"| {r['seed']} "
            f"| {topology} "
            f"| {e_txt} "
            f"| {lam} "
            f"| {_fmt(r['vreme'])} "
            f"| {_fmt(r['hypervolume'], 6)} "
            f"| {r['velicina_arhive']} "
            f"| {r['evaluacije']} "
            f"| {'-' if por_s is None else por_s} "
            f"| {'-' if por_p is None else por_p} "
            f"| {poslato_txt} "
            f"| {primljeno_txt} |"
        )

    lines.append("")

    lines.append("## Poređenje")
    lines.append("")
    lines.append("| Par | Šta se menja | Δ vreme (s) | Odnos vremena | Δ HV | Δ arhiva | Tumačenje |")
    lines.append("|---|---|---:|---:|---:|---:|---|")

    for a, b, sta in POREDJENJA:
        if a not in summary or b not in summary:
            continue
        sa, sb = summary[a], summary[b]
        lines.append(
            f"| {a}–{b} | {sta} "
            f"| {_fmt(sa['vreme_avg'] - sb['vreme_avg'])} "
            f"| {_fmt(sb['vreme_avg'] / sa['vreme_avg'], 2)} "
            f"| {_fmt(sa['hv_avg'] - sb['hv_avg'], 6)} "
            f"| {_fmt(sa['arhiva_avg'] - sb['arhiva_avg'], 2)} "
            f"| {TUMACENJE.get(a + '-' + b, '')} |"
        )

    lines.append("")

    lines.append("## Po procesima")
    lines.append("")
    lines.append(
        "| Config | Seed | Proces | λ | Evaluacije | Poslato kand. | Primljeno novih | Strani u arhivi | Iteracija avg (s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in results:
        for rank, st in sorted(r.get("per_process", {}).items()):
            lines.append(
                f"| {r['config']} | {r['seed']} | {rank} | {st['lam']} "
                f"| {st['evaluacije_zavrsene']} | {st['poslato_kandidata']} "
                f"| {st['primljeno_novih']} | {st['strani_u_arhivi']} "
                f"| {_fmt(st['prosecno_vreme_iteracije'], 4)} |"
            )

    lines.append("")

    lines.append("## Raspon ocena")
    lines.append("")
    lines.append("| Config | Seed | min | max |")
    lines.append("|---|---:|---|---|")

    for r in results:
        sr = r.get("score_ranges", {})
        lines.append(
            f"| {r['config']} | {r['seed']} | {sr.get('min')} | {sr.get('max')} |"
        )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_archive_markdown(results, filename="rezultati_arhive.md"):
    lines = []

    lines.append("# Konačne arhive po pokretanjima")
    lines.append("")

    for r in results:
        lines.append(f"## {r['config']} — seed {r['seed']}")
        lines.append("")
        lines.append(f"- Veličina arhive: {r['velicina_arhive']}")
        lines.append(f"- Hipervolumen: {r['hypervolume']:.6f}")
        lines.append(f"- Vreme: {r['vreme']:.3f} s")
        lines.append("")

        archive = r.get("archive", [])

        if not archive:
            lines.append("Arhiva je prazna.")
            lines.append("")
            continue

        for item in archive:
            if len(item) == 2:
                seq, scores = item
                lines.append(f"- {seq} | {scores}")
            else:
                seq, scores, origin = item
                lines.append(f"- origin={origin} | {seq} | {scores}")

        lines.append("")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Merenja P1-P6 i dodaci P2-pool, P8")

    parser.add_argument("--quick", action="store_true", help="brža, manja merenja")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument(
        "--configs",
        type=str,
        default="P1,P2,P3,P4,P5,P6,P2-pool,P8",
        help="npr. P1,P2 ili P3,P4,P5,P6,P2-pool,P8",
    )

    for ime, tip in [
        ("B", int), ("N", int), ("k", int), ("M", int),
        ("T", int), ("P", int), ("K", int), ("E", int),
    ]:
        parser.add_argument(f"--{ime}", type=tip, default=None)

    parser.add_argument("--E-P6", dest="E_P6", type=int, default=None)
    parser.add_argument("--lam", type=str, default=None, help="npr. 2.0")
    parser.add_argument(
        "--lam-diff",
        dest="lam_diff",
        type=str,
        default=None,
        help="npr. 0.5,1,2,4",
    )
    parser.add_argument(
        "--topology-p5",
        dest="topology_p5",
        default=None,
        choices=["ring", "star", "all2all", "random"],
    )

    args = parser.parse_args()

    cfg = default_config()

    if args.quick:
        cfg = quick_config(cfg)

    for ime in ["B", "N", "k", "M", "T", "P", "K", "E", "E_P6", "topology_p5"]:
        v = getattr(args, ime)
        if v is not None:
            cfg[ime] = v
    if args.lam is not None:
        cfg["lam_common"] = [float(args.lam)]
    if args.lam_diff is not None:
        cfg["lam_diff"] = [float(x) for x in args.lam_diff.split(",")]

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    wanted = {x.strip().upper() for x in args.configs.split(",") if x.strip()}

    results = []

    for seed in seeds:
        if "P1" in wanted:
            print(f"Pokrećem P1, seed={seed}")
            results.append(run_p1(cfg, seed))

        if "P2" in wanted:
            print(f"Pokrećem P2, seed={seed}")
            results.append(run_p2(cfg, seed))

        if "P2-POOL" in wanted:
            print(f"Pokrećem P2-pool (dodatak), seed={seed}")
            results.append(run_p2(cfg, seed, parallel_rank=True, config_name="P2-pool"))

        if "P3" in wanted:
            print(f"Pokrećem P3, seed={seed}")
            results.append(run_p3(cfg, seed))

        if "P4" in wanted:
            print(f"Pokrećem P4, seed={seed}")
            results.append(run_p4(cfg, seed))

        if "P5" in wanted:
            print(f"Pokrećem P5, seed={seed}")
            results.append(run_p5(cfg, seed))

        if "P6" in wanted:
            print(f"Pokrećem P6, seed={seed}")
            results.append(run_p6(cfg, seed))
        if "P8" in wanted:
            print(f"Pokrećem P8 (all2all), seed={seed}")
            results.append(run_p8(cfg, seed))  

    summary = summarize(results, cfg)

    write_markdown(summary, results, cfg, seeds, filename="rezultati.md")
    write_archive_markdown(results, filename="rezultati_arhive.md")

    with open("rezultati.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print("=== Rezime ===")

    for c, s in summary.items():
        print(
            f"{c}: vreme={s['vreme_avg']:.3f}s, "
            f"hv={s['hv_avg']:.6f}, "
            f"arhiva={s['arhiva_avg']:.2f}, "
            f"evaluacije={s['evaluacije_avg']:.1f}"
        )

    print()
    print("Sačuvani fajlovi:")
    print("- rezultati.md")
    print("- rezultati_arhive.md")
    print("- rezultati.json")


if __name__ == "__main__":
    main()