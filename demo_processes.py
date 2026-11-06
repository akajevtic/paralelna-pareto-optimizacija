# demo_processes.py

import argparse

from common import G
from processes import TOPOLOGIJA, pokreni_procese


def parse_lam(value):
    """
    Prima jednu vrednost ili listu vrednosti odvojenih zarezima.
    Primer:
      --lam 2.0
      --lam 0.5,1.0,2.0,4.0
    """
    if not value:
        return [2.0]

    parts = [x.strip() for x in value.split(",")]
    return [float(x) for x in parts]


def main():
    parser = argparse.ArgumentParser(
        description="Višeprocesna Pareto optimizacija sa razmenom kandidata."
    )

    parser.add_argument("--B", type=int, default=800, help="ukupan budžet evaluacija")
    parser.add_argument("--N", type=int, default=40, help="veličina lokalne populacije")
    parser.add_argument("--k", type=int, default=5, help="maksimalan broj promena u početnom mutantu")
    parser.add_argument("--M", type=int, default=25, help="broj novih lokalnih kandidata po iteraciji")
    parser.add_argument("--lam", type=str, default="2.0", help="lambda ili lista lambda vrednosti, npr. 0.5,1.0,2.0,4.0")
    parser.add_argument("--K", type=int, default=5, help="maksimalan broj kandidata po razmeni")
    parser.add_argument("--E", type=int, default=2, help="razmena nakon svake E-te lokalne iteracije")
    parser.add_argument("--P", type=int, default=4, help="broj procesa")
    parser.add_argument("--T", type=int, default=1, help="broj niti po procesu; trenutna verzija podržava T=1")
    parser.add_argument("--topology", type=str, default=TOPOLOGIJA, choices=["ring", "star", "all2all", "random"])
    parser.add_argument("--send-selection", type=str, default="archive", choices=["archive", "random-evidence"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--received-resets-g", action="store_true", help="ako je zadato, primljeni novi kandidat resetuje lokalni G")
    parser.add_argument("--timeout", type=float, default=1200.0)

    args = parser.parse_args()

    if args.T != 1:
        print("Upozorenje: trenutna procesna verzija je napravljena za T=1.")
        print("P3-P6 konfiguracije iz postavke koriste 4 procesa x 1 nit.")

    if args.B % args.P != 0:
        print("Upozorenje: B nije deljiv sa P. Višak budžeta neće biti iskorišćen.")

    lam_list = parse_lam(args.lam)

    rez = pokreni_procese(
        B=args.B,
        N=args.N,
        k=args.k,
        M=args.M,
        lam_list=lam_list,
        G=G,
        K=args.K,
        E=args.E,
        topology=args.topology,
        send_selection=args.send_selection,
        P=args.P,
        seed=args.seed,
        received_resets_G=args.received_resets_g,
        timeout=args.timeout,
    )

    agg = rez["agg"]

    print()
    print("=== Globalni rezultati ===")
    print("Ukupno vreme:", rez["ukupno_vreme"])
    print("Broj procesa sa rezultatom:", agg["broj_procesa_sa_rezultatom"])
    print("Procesi koji nedostaju:", rez["missing_ranks"])
    print("Veličina konačne arhive:", agg["velicina_arhive"])
    print("Ukupno evaluacija (završene):", agg["evaluacije_zavrsene"])
    print("Ukupno evaluacija (započete):", agg["evaluacije_zapocete"])
    print("Ukupno iteracija:", agg["iteracije"])
    print("Ukupno neuspešnih pokušaja:", agg["neuspesi_ukupno"])
    print("Ukupno poslato poruka:", agg["poslato_poruka"])
    print("Ukupno poslato kandidata:", agg["poslato_kandidata"])
    print("Ukupno primljeno poruka:", agg["primljeno_poruka"])
    print("Ukupno primljeno novih kandidata:", agg["primljeno_novih"])
    print("Prosek stranih kandidata u lokalnim arhivama:", agg["strani_u_arhivi_avg"])

    print()
    print("=== Po procesima ===")
    header = f"{'rank':>4} | {'status':<28} | {'eval':>5} | {'iter':>4} | {'poslato':>7} | {'primljeno':>9} | {'strani':>6} | {'vreme':>8}"
    print(header)
    print("-" * len(header))

    for rank in sorted(rez["per_process"].keys()):
        s = rez["per_process"][rank]

        print(
            f"{rank:>4} | "
            f"{s.get('status', ''):<28} | "
            f"{s.get('evaluacije_zavrsene', 0):>5} | "
            f"{s.get('iteracije', 0):>4} | "
            f"{s.get('poslato_kandidata', 0):>7} | "
            f"{s.get('primljeno_novih', 0):>9} | "
            f"{s.get('strani_u_arhivi', 0):>6} | "
            f"{s.get('ukupno_vreme', 0.0):>8.3f}"
        )

    print()
    print("=== Konačna arhiva ===")

    if not rez["final_archive"]:
        print("Konačna arhiva je prazna.")
        return

    print("Prvih nekoliko kandidata:")

    for s in rez["final_archive"][:10]:
        scores = rez["seq_scores"][s]
        origin = rez["origins"].get(s)
        print(f"origin={origin} | {s} | {scores}")

    print()
    print("Raspon ocena u konačnoj arhivi:")

    ocene = [rez["seq_scores"][s] for s in rez["final_archive"]]

    for j, ime in enumerate(["zaraznost", "virulentnost", "imunoevazivnost"]):
        vals = [o[j] for o in ocene]
        print(f"{ime}: min={min(vals)}, max={max(vals)}")


if __name__ == "__main__":
    main()