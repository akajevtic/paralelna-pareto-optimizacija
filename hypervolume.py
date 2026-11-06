# hypervolume.py

def hv2d(tacke, ref):
    """
    Površina koju dominiraju tačke (x, y) u odnosu na ref.
    Maksimizacija.
    """
    povrsina = 0.0
    najbolji_y = ref[1]

    for x, y in sorted(tacke, reverse=True):
        if y > najbolji_y:
            povrsina += (x - ref[0]) * (y - najbolji_y)
            najbolji_y = y

    return povrsina


def hv3d(tacke, ref):
    """
    Zapremina koju dominiraju tačke (x, y, z) u odnosu na ref.
    Maksimizacija.
    """
    if not tacke:
        return 0.0

    tacke = sorted(tacke, key=lambda t: t[2], reverse=True)

    zapremina = 0.0

    for i, (_, _, z) in enumerate(tacke):
        sledeci_z = tacke[i + 1][2] if i + 1 < len(tacke) else ref[2]
        visina = z - sledeci_z

        if visina > 0:
            sloj = [(x, y) for x, y, _ in tacke[:i + 1]]
            zapremina += hv2d(sloj, ref[:2]) * visina

    return zapremina


def normalizuj(tacke, lo=(0.0, 0.0, 0.0), hi=(10.0, 10.0, 10.0)):
    """
    Normalizuje ocene na [0, 1].
    lo i hi su fiksne granice za sva pokretanja.
    """
    norm = []

    for t in tacke:
        nt = []

        for v, l, h in zip(t, lo, hi):
            if h == l:
                nt.append(0.0)
            else:
                x = (v - l) / (h - l)

                if x < 0.0:
                    x = 0.0
                elif x > 1.0:
                    x = 1.0

                nt.append(x)

        norm.append(tuple(nt))

    return norm


def hypervolume_arhive(
    ocene,
    lo=(0.0, 0.0, 0.0),
    hi=(10.0, 10.0, 10.0),
    ref=(-0.1, -0.1, -0.1),
):
    """
    Računa hipervolumen za listu trojki ocena.
    """
    if not ocene:
        return 0.0

    front = normalizuj(list(ocene), lo=lo, hi=hi)
    return hv3d(front, ref)