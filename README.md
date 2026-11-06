# Paralelna Pareto optimizacija

## Pokretanje
pip install numpy
python demo_seq.py --B 300 --seed 1
python demo_threads.py --B 300 --T 4
python demo_processes.py --B 800 --P 4 --topology ring --lam 0.5,1,2,4
python merenja.py                # P1-P6 + dodaci P2-pool i P8, semena 1,2,3
python merenja.py --quick --configs P1,P2
python test_core.py; python test_pareto_parallel.py; python test_threaded.py; python test_processes.py

## Organizacija
- common.py – dominacija, Pareto slojevi, crowding, generisanje, sekvencijalna verzija, konstanta G = 100
- threaded.py – T radnih niti nad zajedničkim stanjem (populacija, arhiva, evidencija, brojači, jedan RNG) pod Lock-om
- pareto_parallel.py – rangiranje preko ThreadPoolExecutor-a (dodatni eksperiment, meri se kao P2-pool)
- processes.py – P procesa, razmena po topologijama, spajanje arhiva, konstanta TOPOLOGIJA
- hypervolume.py – HV u 3D (normalizacija 0..10, ref -0.1)
- merenja.py – P1-P6 i dodaci, upis u rezultati.md/.json
- oceni.py, anchors.json, blosum62.json – dato od profesora

## Verzija sa nitima
- T niti se prave jednom i svaka radi u petlji dok se ne potroši budžet ili ne dostigne prag G. Nema barijere po iteraciji.
- Iteracija je samo brojač za statistiku: nova počinje na svakih M rezervisanih kandidata. Iteracija je gotova čim je M rezervisano, niko ni na koga ne čeka.
- Nit uzima snimak populacije pod Lock-om i generiše kandidata van glavnog Lock-a. Ceo program ima jedan numpy generator; pristup njemu ide pod posebnim `rng_lock`, pa je generisanje serijalizovano.
- Provera duplikata, upis `None` u evidenciju i rezervacija budžeta idu pod Lock-om; evaluacija van Lock-a.
- Posle evaluacije nit, pod Lock-om, upisuje ocene i odmah spaja tog jednog kandidata u populaciju i arhivu. Glavna nit ne spaja ništa.
- **Kandidati bez ocena u rangiranju:** pošto se svaki kandidat spaja tek posle svoje evaluacije, u Pareto rangiranje nikad ne ulazi kandidat sa `None` ocenama. Kandidat „čija evaluacija još traje“ postoji samo u evidenciji (sa `None`, radi provere duplikata), a u populaciju ulazi kada mu se evaluacija završi.
- Provera da nema evaluacija u toku (`in_flight == 0`) radi se jednom, na kraju programa, posle `join` svih niti.
- Rangiranje je sekvencijalno (propisani P2). P2-pool je isti algoritam sa rangiranjem kroz ThreadPoolExecutor iz pareto_parallel.py.

## Protokol razmene
- Poruka: {"tip": "CANDS", "from": rank, "iter": i, "kandidati": [(seq, (s1,s2,s3), origin), ...]}
- Slanje: posle svake E-te lokalne iteracije, najviše K nasumičnih ocenjenih kandidata iz evidencije, svim susedima po topologiji.
- Topologije: ring (rank → rank+1), star (0 ↔ svi), all2all, random (jedan nasumičan sused).
- Topologija se zadaje konstantom `TOPOLOGIJA` na vrhu processes.py, kako traže izmena 6 i odeljak 5; argument `--topology` je samo prepisuje, da bi merenja.py mogao da pokrene P3-P6 bez menjanja koda.
- Prijem: na početku iteracije red se prazni sa get_nowait; nov string se upisuje u evidenciju sa ocenama, ne troši budžet, odmah se spaja u populaciju i arhivu i resetuje brojač G.
- Ažuriranje u procesu: kao kod niti, populacija i arhiva se ažuriraju posle svake evaluacije.
- Svaki proces ima jedan slučajni tok (za generisanje i za izbor suseda/kandidata).
- Kraj: proces šalje arhivu u result_queue i prazni svoj red dok glavni proces ne postavi stop_event.
- Spajanje: glavni proces uniji arhive, uklanja duplikate i dominirane.

## Tumačenje rezultata
Merenja: rezultati.md, B=2000, semena 1,2,3. Propisani P2 je P2 bez pool-a.
- Vreme: P1 traje ~8.9 s. P2 (T=4) traje ~3.4 s, uz ubrzanje 2.65 i efikasnost 0.66. Procesi P3-P8 (P=4) traju ~3.1 s, uz ubrzanje ~2.87 i efikasnost ~0.72; vreme uključuje pokretanje procesa.
- P2 je sporiji nego pre izmene plana (ranije 2.9 s). Rangiranje se sada radi pod Lock-om posle svake evaluacije (~1900 puta umesto ~48), a jedini RNG serijalizuje generisanje.
- P2-pool (~9.1 s) je sporiji i od P1. Pool se poziva 1900 puta, a pod GIL-om provera dominacije ne ide paralelno, pa ostaje samo režija; rangiranje uz to drži Lock, pa niti čekaju.
- Kvalitet: razlike u HV između P1-P6 (0.17-0.21) manje su od rasipanja kroz semena (stdev 0.04-0.07). Sa 3 semena nijedna promena (λ, topologija, E) ne daje pouzdano bolji HV. P8 (all2all) ima najviši HV (0.223) i najmanje rasipanje (0.019).
- Razmena: nasumični kandidati iz evidencije uglavnom su lokalno nepoznati (80-90% poslatih je primljeno kao novo), ali retko ostaju u konačnoj arhivi (0.08-0.42 stranih tačaka po procesu).
- Tabela poređenja P2-P1, P3-P2, P4-P3, P5-P4, P6-P4 sa tumačenjem je u rezultati.md, kao i tabela po procesima.

## Odstupanja i napomene
- Budžet: B // P, ostatak ide prvim procesima.
- G u verziji sa nitima: zajednički brojač, uzastopnost je približna jer niti rade istovremeno.
- **Izostavljeni testovi:** postavka (odeljak 7) traži proveru da se populacija i arhiva ne menjaju tokom prikupljanja i da ažuriranje počinje tek kada su sve niti završile. Ti testovi su namerno izostavljeni: izmena 1 traži ažuriranje posle svake evaluacije, a izmena 2 uklanja barijeru po iteraciji, pa bi obe provere po dizajnu padale. Umesto njih testovi proveravaju da u evidenciji nema duplikata ni sa mnogo niti i da jedna iteracija ne započne više od M evaluacija.
- P2-pool i P8 (all2all) su dodatna merenja van P1-P6.
- Poslate i primljene poruke se razlikuju: poruke koje stignu posle kraja lokalne pretrage procesa (faza pražnjenja reda) se odbacuju i ne broje kao primljene.
- Hipervolumen: ocene se normalizuju fiksnim granicama 0..10 (sve dobijene ocene su u tom opsegu), referentna tačka je -0.1 po svakoj osi, da bi pokretanja bila uporediva.
