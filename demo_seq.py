# demo_seq.py

import argparse

from oceni import oceni

from common import (
    G,
    optimizuj_sekvencijalno,
    POCETNA_SEKVENCA,
)


def main():
    parser = argparse.ArgumentParser(description="Sekvencijalna Pareto optimizacija sekvenci.")

    parser.add_argument("--B", type=int, default=80, help="ukupan budžet evaluacija")
    parser.add_argument("--N", type=int, default=20, help="maksimalna veličina populacije")
    parser.add_argument("--k", type=int, default=5, help="maksimalan broj promena u početnom mutantu")
    parser.add_argument("--M", type=int, default=10, help="broj novih kandidata po iteraciji")
    parser.add_argument("--lam", type=float, default=2.0, help="lambda za Poissonovu mutaciju")
    parser.add_argument("--seed", type=int, default=1, help="seed za RNG")

    args = parser.parse_args()

    arhiva, evidencija, stats = optimizuj_sekvencijalno(
        oceni_func=oceni,
        B=args.B,
        N=args.N,
        k=args.k,
        M=args.M,
        lam=args.lam,
        G=G,
        seed=args.seed,
        pocetna=POCETNA_SEKVENCA,
    )

    print("Status:", stats["status"])
    print("Broj evaluacija:", stats["evaluacije"])
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