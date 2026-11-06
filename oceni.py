import math
import json
import threading
import time


BLOSUM62 = json.load(open('blosum62.json'))
PARETO_ANCHORS = json.load(open('anchors.json'))
AA = set('ACDEFGHIKLMNPQRSTVWY')

_tabele_niti = threading.local()


def _tabele():
    """Privatne kopije tabela za trenutnu nit (lenjivo, jednom po niti).

    Pod CPython 3.14 free-threading (PEP 703) niti zaista rade paralelno, ali
    gusto čitanje istih deljenih dict/list objekata iz više niti izaziva
    konkurenciju oko refbrojčanih polja (biased/deferred refcounting) — mereno
    i desetine puta SPORIJE nego sa jednom niti. Privatna kopija po niti daje
    identične rezultate i paralelne niti se skaliraju. Pod običnim (GIL)
    CPython-om ponašanje je funkcionalno isto kao i do sada.
    """
    tabele = getattr(_tabele_niti, "vrednost", None)
    if tabele is None:
        b62 = {a: dict(red) for a, red in BLOSUM62.items()}
        ankeri = {ime: {"seq": d["seq"],
                        "stats": tuple(float(x) for x in d["stats"])}
                  for ime, d in PARETO_ANCHORS.items()}
        tabele = (b62, ankeri)
        _tabele_niti.vrednost = tabele
    return tabele


def get_blosum_distance(seq1, seq2):
    b62 = _tabele()[0]
    score = 0
    max_score1 = 0
    max_score2 = 0
    for a, b in zip(seq1, seq2):
        score += b62[a][b]
        max_score1 += b62[a][a]
        max_score2 += b62[b][b]
    max_self = max(max_score1, max_score2)
    return 1.0 - (max(0, score) / max_self)

def oceni(seq):
    time.sleep(0.004)
    player_seq = seq.upper().strip()

    b62, pareto = _tabele()

    # Absolute exact match intercept
    for name, data in pareto.items():
        if seq == data["seq"]:
            return list(data["stats"])

    t_sum, s_sum, e_sum = 0.0, 0.0, 0.0
    total_weight = 0.0
    min_distance = 1.0

    for name, data in pareto.items():
        dist = get_blosum_distance(seq, data["seq"])
        if dist < min_distance:
            min_distance = dist

        if dist < 0.001: dist = 0.001

        # High power IDW cleanly blends neighboring nodes
        weight = 1.0 / (dist ** 4)
        total_weight += weight

        t_sum += data["stats"][0] * weight
        s_sum += data["stats"][1] * weight
        e_sum += data["stats"][2] * weight

    t = t_sum / total_weight
    s = s_sum / total_weight
    e = e_sum / total_weight

    # THE CORRECTION: Apply an exponential decay multiplier based on distance.
    # Moving away from exact anchor landmarks instantly scales numbers downward.
    # A single mutation away from Wuhan scales the output directly toward zero.
    decay_strictness = 8.0
    viability_decay = math.exp(-decay_strictness * min_distance)

    t *= viability_decay
    s *= viability_decay
    e *= viability_decay

    return min(10.0, round(t, 2)), min(10.0, round(s, 2)), min(10.0, round(e, 2))

# --- CLEAN TESTING EXAMPLES ---
# Exact match works cleanly
print("Pure Delta Profile:", oceni("TEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTN"))

# Chaotic broken sequence naturally drops to zero via the viability decay multiplier
print("Biologically Broken (RWRW...):", oceni("RWRWRWRWRWRWRWRWRWRWRWRWRWRWRWRW"))
