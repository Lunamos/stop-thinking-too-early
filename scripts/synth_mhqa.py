"""Synthetic natural-language multi-hop QA with fictional entities: chains of facts over typed relations, written as short
titled passages (one fact each, shuffled with facts from distractor chains), and a composed question over 2-5 hops.
The answer is the last entity of the chain. Used to train single-layer maps without any real benchmark data."""
import random

FIRST = ["Alden", "Brisa", "Corvin", "Dalia", "Emrys", "Faye", "Garrick", "Helka", "Ivo", "Jessamy", "Kasimir", "Liora",
         "Marek", "Nerys", "Orrin", "Petra", "Quill", "Rosalind", "Soren", "Talia", "Ulric", "Vesna", "Wendell", "Xanthe",
         "Yorick", "Zelda", "Anselm", "Beatrix", "Cassius", "Delphine", "Eamon", "Flora", "Gideon", "Honora", "Ignatius",
         "Juno", "Leander", "Mirela", "Niall", "Odile"]
LAST = ["Ashcombe", "Brennock", "Calloway", "Dunmere", "Everly", "Fairholt", "Greaves", "Halloran", "Iverson", "Jardine",
        "Kestrel", "Lockridge", "Marlowe", "Northcott", "Oakhurst", "Penhallow", "Quennell", "Ravensworth", "Stroud",
        "Thackeray", "Underhill", "Vance", "Whitlock", "Yardley", "Zeller", "Ambrose", "Blackwood", "Crane", "Delacroix",
        "Ellery"]
PLACE_A = ["Vel", "Mar", "Tor", "Ash", "Brin", "Cal", "Dun", "Esk", "Fen", "Glen", "Hol", "Kel", "Lor", "Mor", "Nar",
           "Os", "Pell", "Quar", "Rav", "Sel", "Tam", "Ul", "Var", "Wyn"]
PLACE_B = ["ford", "haven", "mere", "stead", "wick", "burg", "dale", "port", "ridge", "moor", "vale", "holm", "cliff",
           "field", "gate", "stone"]
WORK_A = ["The Silent", "A Distant", "The Last", "Crimson", "The Hollow", "Winter", "The Glass", "Paper", "The Iron",
          "Midnight", "The Salt", "Autumn", "The Broken", "Golden", "The Quiet"]
WORK_B = ["Harbor", "Orchard", "Lantern", "Kingdom", "Letters", "River", "Garden", "Mirror", "Voyage", "Tide", "Meadow",
          "Signal", "Archive", "Compass", "Frontier"]
ORG = ["Institute", "Company", "Society", "Foundation", "Works", "Guild", "Laboratory", "Press"]

# relation: (source type, target type, fact template, question phrase)
REL = {
    "director": ("work", "person", "{s} is a film directed by {t}.", "the director of {x}"),
    "author": ("work", "person", "{s} is a novel written by {t}.", "the author of {x}"),
    "spouse": ("person", "person", "{s} is married to {t}.", "the spouse of {x}"),
    "mentor": ("person", "person", "{s} studied under {t}.", "the teacher of {x}"),
    "birthplace": ("person", "place", "{s} was born in {t}.", "the birthplace of {x}"),
    "employer": ("person", "org", "{s} works for {t}.", "the employer of {x}"),
    "founder": ("org", "person", "{s} was founded by {t}.", "the founder of {x}"),
    "hq": ("org", "place", "{s} is headquartered in {t}.", "the headquarters city of {x}"),
    "region": ("place", "place", "{s} is a town in the province of {t}.", "the province containing {x}"),
    "mayor": ("place", "person", "The mayor of {s} is {t}.", "the mayor of {x}"),
}
BY_SRC = {}
for r, (st, tt, _, _) in REL.items():
    BY_SRC.setdefault(st, []).append(r)


class Names:
    def __init__(self, rng):
        self.rng = rng
        self.used = set()

    def make(self, typ):
        for _ in range(1000):
            if typ == "person":
                s = f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"
            elif typ == "place":
                s = self.rng.choice(PLACE_A) + self.rng.choice(PLACE_B)
            elif typ == "work":
                s = f"{self.rng.choice(WORK_A)} {self.rng.choice(WORK_B)}"
            else:
                s = f"{self.rng.choice(LAST)} {self.rng.choice(ORG)}"
            if s not in self.used:
                self.used.add(s)
                return s
        raise RuntimeError("name space exhausted")


def chain(rng, names, hops, start_type=None):
    typ = start_type or rng.choice(["work", "person", "org", "place"])
    ents = [names.make(typ)]
    rels = []
    for _ in range(hops):
        r = rng.choice(BY_SRC[typ])
        rels.append(r)
        typ = REL[r][1]
        ents.append(names.make(typ))
    return ents, rels


def make_item(rng, hops, n_distract_chains=2):
    names = Names(rng)
    ents, rels = chain(rng, names, hops)
    facts = [(ents[i], REL[r][2].format(s=ents[i], t=ents[i + 1])) for i, r in enumerate(rels)]
    for _ in range(n_distract_chains):
        # distractor chains share the relation sequence (so relation words do not reveal the answer)
        dents = [names.make(REL[rels[0]][0])]
        for r in rels:
            dents.append(names.make(REL[r][1]))
        facts += [(dents[i], REL[r][2].format(s=dents[i], t=dents[i + 1])) for i, r in enumerate(rels)]
    rng.shuffle(facts)
    x = ents[0]
    for r in rels:
        x = REL[r][3].format(x=x)
    q = x[0].upper() + x[1:]
    q = f"Who or what is {x}?"
    gold = [(t, s) for t, s in facts]
    return dict(id=f"syn{hops}_{rng.randrange(10 ** 9)}", question=q, answer=ents[-1], aliases=[ents[-1]],
                type=f"{hops}hop", level=str(hops), gold=gold, distractors=[])
