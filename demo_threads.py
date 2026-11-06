# demo_threads.py

import argparse

from oceni import oceni

from common import G, POCETNA_SEKVENCA
from threaded import optimizuj_nitima


def main():
    parser = argparse.ArgumentParser(
        description="Pareto optimizacija sekvenci sa više niti u jednom procesu."
    )

    parser.add_argument("--B", type=int, default=300, help="ukupan budžet evaluacija")
    parser.add_argument("--N", type=int, default=30, help="maksimalna veličina populacije")
    parser.add_argument("--k", type=int, default=5, help="maksimalan broj promena u početnom mutantu")
    parser.add_argument("--M", type=int, default=20, help="broj novih kandidata po iteraciji")
    parser.add_argument("--lam", type=float, default=2.0, help="lambda za Poissonovu mutaciju")
    parser.add_argument("--T", type=int, default=4, help="broj radnih niti")
    parser.add_argument("--seed", type=int, default=1, help="seed za RNG")

    args = parser.parse_args()

    arhiva, evidencija, stats = optimizuj_nitima(
        oceni_func=oceni,
        B=args.B,
        N=args.N,
        k=args.k,
        M=args.M,
        lam=args.lam,
        G=G,
        T=args.T,
        seed=args.seed,
        pocetna=POCETNA_SEKVENCA,
    )

    print("Status:", stats["status"])
    print("Broj niti:", stats["T"])
    print("Broj evaluacija (završene):", stats["evaluacije_zavrsene"])
    print("Broj evaluacija (započete):", stats["evaluacije_zapocete"])
    print("Broj iteracija:", stats["iteracije"])
    print("Ukupno neuspešnih pokušaja:", stats["neuspesi_ukupno"])
    print("Veličina konačne arhive:", len(arhiva))
    print("Ukupno vreme:", stats["ukupno_vreme"])
    print("Prosečno vreme iteracije:", stats["prosecno_vreme_iteracije"])

    if not arhiva:
        print("Arhiva je prazna.")
        return

    print("\nPrvih nekoliko kandidata u arhivi:")
    for s in arhiva[:10]:
        print(s, evidencija[s])

    print("\nRaspon ocena u arhivi:")
    ocene = [evidencija[s] for s in arhiva if evidencija[s] is not None]

    if ocene:
        for j, ime in enumerate(["zaraznost", "virulentnost", "imunoevazivnost"]):
            vals = [o[j] for o in ocene]
            print(f"{ime}: min={min(vals)}, max={max(vals)}")


if __name__ == "__main__":
    main()