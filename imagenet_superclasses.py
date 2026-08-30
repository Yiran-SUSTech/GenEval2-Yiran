import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.imagenet_en_cn import IMAGENET_1K_CLASSES


DATA_PATH = Path(__file__).resolve().parent / "data" / "imagenet1k_superclasses.json"


SUPERCLASS_META = {
    "fish_sharks_rays": {"name": "fish, sharks, and rays", "macro_domain": "fish_marine", "description": "fish, sharks, or rays"},
    "birds_songbirds": {"name": "small birds and raptors", "macro_domain": "bird", "description": "a bird such as a songbird, raptor, or owl"},
    "amphibians": {"name": "amphibians", "macro_domain": "reptile_amphibian", "description": "an amphibian such as a frog or salamander"},
    "reptiles_turtles": {"name": "turtles and tortoises", "macro_domain": "reptile_amphibian", "description": "a turtle or tortoise"},
    "reptiles_lizards": {"name": "lizards and crocodilians", "macro_domain": "reptile_amphibian", "description": "a lizard, crocodile, or similar reptile"},
    "reptiles_snakes": {"name": "snakes", "macro_domain": "reptile_amphibian", "description": "a snake"},
    "arthropods_spiders": {"name": "spiders and arthropods", "macro_domain": "insect_arthropod", "description": "a spider, scorpion, tick, or similar arthropod"},
    "birds_ground_water": {"name": "ground birds and waterfowl", "macro_domain": "bird", "description": "a bird such as a grouse, parrot, duck, or goose"},
    "mammals_marsupials": {"name": "marsupials and monotremes", "macro_domain": "other_mammal", "description": "a marsupial or monotreme mammal"},
    "marine_invertebrates": {"name": "marine invertebrates", "macro_domain": "fish_marine", "description": "a marine invertebrate such as a jellyfish, shellfish, or crab"},
    "birds_wading_sea": {"name": "wading birds and seabirds", "macro_domain": "bird", "description": "a wading bird, seabird, or water bird"},
    "marine_mammals": {"name": "marine mammals", "macro_domain": "other_mammal", "description": "a marine mammal such as a whale or sea lion"},
    "dogs_toy_small": {"name": "toy and small companion dogs", "macro_domain": "canine", "description": "a small companion dog"},
    "dogs_hounds": {"name": "hounds", "macro_domain": "canine", "description": "a hound or sighthound dog"},
    "dogs_terriers": {"name": "terriers", "macro_domain": "canine", "description": "a terrier or schnauzer dog"},
    "dogs_sporting": {"name": "sporting dogs", "macro_domain": "canine", "description": "a retriever, pointer, setter, or spaniel dog"},
    "dogs_working": {"name": "working and shepherd dogs", "macro_domain": "canine", "description": "a shepherd, guard, mountain, or working dog"},
    "dogs_spitz_companion": {"name": "spitz and companion dogs", "macro_domain": "canine", "description": "a spitz-type or companion dog"},
    "wild_canines": {"name": "wild canines and foxes", "macro_domain": "canine", "description": "a wolf, fox, hyena, or other wild canine"},
    "cats_domestic": {"name": "domestic cats", "macro_domain": "feline", "description": "a domestic cat"},
    "cats_wild": {"name": "wild cats and big cats", "macro_domain": "feline", "description": "a wild cat or big cat"},
    "bears_mongooses": {"name": "bears and mongooses", "macro_domain": "other_mammal", "description": "a bear, mongoose, or meerkat"},
    "insects": {"name": "insects", "macro_domain": "insect_arthropod", "description": "an insect"},
    "echinoderms": {"name": "echinoderms", "macro_domain": "fish_marine", "description": "a sea star, sea urchin, or sea cucumber"},
    "small_mammals": {"name": "small mammals", "macro_domain": "other_mammal", "description": "a small mammal such as a rabbit, hamster, or squirrel"},
    "hoofed_mammals": {"name": "hoofed mammals", "macro_domain": "other_mammal", "description": "a hoofed mammal such as a horse, pig, cow, sheep, or antelope"},
    "other_mammals": {"name": "other mammals", "macro_domain": "other_mammal", "description": "a mammal such as a weasel, otter, sloth, or armadillo"},
    "primates": {"name": "primates", "macro_domain": "other_mammal", "description": "a primate such as a monkey or ape"},
    "elephants_pandas": {"name": "elephants and pandas", "macro_domain": "other_mammal", "description": "an elephant or panda"},
    "fish_large": {"name": "recognizable fish", "macro_domain": "fish_marine", "description": "a fish"},
    "wearables": {"name": "clothing and wearable items", "macro_domain": "household_object", "description": "a wearable item such as clothing, shoes, or eyewear"},
    "musical_instruments": {"name": "musical instruments", "macro_domain": "tool_instrument", "description": "a musical instrument"},
    "air_vehicles": {"name": "air vehicles", "macro_domain": "vehicle", "description": "an aircraft or other flying vehicle"},
    "watercraft": {"name": "boats and ships", "macro_domain": "vehicle", "description": "a boat, ship, or other watercraft"},
    "rail_vehicles": {"name": "rail vehicles", "macro_domain": "vehicle", "description": "a train, railcar, tram, or other rail vehicle"},
    "road_vehicles": {"name": "road vehicles", "macro_domain": "vehicle", "description": "a car, truck, bus, van, or other road vehicle"},
    "small_transport": {"name": "small personal transport", "macro_domain": "vehicle", "description": "a bicycle, scooter, tricycle, or similar small transport vehicle"},
    "buildings_places": {"name": "buildings and places", "macro_domain": "structure_scene", "description": "a building, shop, house, or other place"},
    "structures_monuments": {"name": "structures and monuments", "macro_domain": "structure_scene", "description": "a structure, monument, bridge, wall, or tower"},
    "furniture_home": {"name": "furniture and home fixtures", "macro_domain": "household_object", "description": "a piece of furniture or home fixture"},
    "kitchenware": {"name": "kitchenware and tableware", "macro_domain": "household_object", "description": "kitchenware, cookware, or tableware"},
    "tools_hardware": {"name": "tools and hardware", "macro_domain": "tool_instrument", "description": "a hand tool, hardware item, or workshop implement"},
    "electronics": {"name": "electronics and devices", "macro_domain": "tool_instrument", "description": "an electronic device, computer, screen, or communication device"},
    "sports_equipment": {"name": "sports equipment and game items", "macro_domain": "household_object", "description": "sports equipment, a ball, or a game-related item"},
    "containers": {"name": "containers and packages", "macro_domain": "household_object", "description": "a container, bottle, bag, package, or storage item"},
    "weapons_armor": {"name": "weapons and armor", "macro_domain": "tool_instrument", "description": "a weapon, armor piece, or military protective item"},
    "books_signs_media": {"name": "books, signs, and printed media", "macro_domain": "structure_scene", "description": "a book, sign, printed page, or similar media item"},
    "food_prepared": {"name": "prepared food and drinks", "macro_domain": "food_plant", "description": "prepared food or a drink"},
    "produce_plants": {"name": "produce and plant items", "macro_domain": "food_plant", "description": "a fruit, vegetable, flower, seed, or plant item"},
    "natural_landscapes": {"name": "natural landscapes", "macro_domain": "structure_scene", "description": "a natural landscape such as a mountain, coast, lake, or volcano"},
    "people_roles": {"name": "people", "macro_domain": "structure_scene", "description": "a person"},
    "fungi": {"name": "fungi", "macro_domain": "food_plant", "description": "a fungus or mushroom"},
    "household_misc": {"name": "everyday household objects", "macro_domain": "household_object", "description": "an everyday manmade household object"},
}


ANIMAL_RANGES = [
    (0, 6, "fish_sharks_rays"),
    (7, 24, "birds_songbirds"),
    (25, 32, "amphibians"),
    (33, 37, "reptiles_turtles"),
    (38, 51, "reptiles_lizards"),
    (52, 68, "reptiles_snakes"),
    (69, 79, "arthropods_spiders"),
    (80, 100, "birds_ground_water"),
    (101, 106, "mammals_marsupials"),
    (107, 126, "marine_invertebrates"),
    (127, 146, "birds_wading_sea"),
    (147, 150, "marine_mammals"),
    (151, 158, "dogs_toy_small"),
    (159, 178, "dogs_hounds"),
    (179, 204, "dogs_terriers"),
    (205, 221, "dogs_sporting"),
    (222, 252, "dogs_working"),
    (253, 268, "dogs_spitz_companion"),
    (269, 280, "wild_canines"),
    (281, 285, "cats_domestic"),
    (286, 293, "cats_wild"),
    (294, 299, "bears_mongooses"),
    (300, 326, "insects"),
    (327, 329, "echinoderms"),
    (330, 338, "small_mammals"),
    (339, 355, "hoofed_mammals"),
    (356, 364, "other_mammals"),
    (365, 384, "primates"),
    (385, 388, "elephants_pandas"),
    (389, 397, "fish_large"),
]


OBJECT_RULES = [
    ("food_prepared", ["pizza", "burrito", "hotdog", "cheeseburger", "carbonara", "consomme", "guacamole", "trifle", "ice cream", "ice lolly", "pretzel", "bagel", "mashed potato", "red wine", "espresso", "eggnog", "cup", "potpie", "meat loaf", "hot pot", "french loaf", "dough"]),
    ("fungi", ["mushroom", "agaric", "gyromitra", "stinkhorn", "earthstar", "hen-of-the-woods", "bolete", "coral fungus"]),
    ("produce_plants", ["cabbage", "broccoli", "cauliflower", "zucchini", "squash", "cucumber", "artichoke", "pepper", "cardoon", "apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "custard apple", "pomegranate", "rapeseed", "daisy", "ladys slipper", "corn", "acorn", "rose hip", "buckeye", "ear", "hay"]),
    ("natural_landscapes", ["alp", "bubble", "cliff", "coral reef", "geyser", "lakeside", "promontory", "sandbar", "seashore", "valley", "volcano"]),
    ("people_roles", ["ballplayer", "groom", "scuba diver"]),
    ("books_signs_media", ["book jacket", "comic book", "crossword", "menu", "street sign", "traffic light", "web site", "website"]),
    ("weapons_armor", ["rifle", "revolver", "assault rifle", "cannon", "bulletproof vest", "chain mail", "breastplate", "shield", "scabbard", "missile", "projectile", "holster", "muzzle", "pickelhaube"]),
    ("musical_instruments", ["accordion", "guitar", "banjo", "bassoon", "cello", "cornet", "drum", "flute", "french horn", "gong", "harmonica", "harp", "maraca", "marimba", "microphone", "oboe", "ocarina", "organ", "panpipe", "piano", "sax", "steel drum", "trombone", "violin"]),
    ("air_vehicles", ["airliner", "airship", "warplane", "wing", "space shuttle", "parachute", "balloon"]),
    ("watercraft", ["boat", "canoe", "catamaran", "container ship", "fireboat", "gondola", "lifeboat", "liner", "ocean liner", "pirate ship", "schooner", "speedboat", "submarine", "trimaran", "yawl", "ship", "aircraft carrier"]),
    ("rail_vehicles", ["bullet train", "locomotive", "freight car", "passenger car", "streetcar", "trolleybus"]),
    ("small_transport", ["bicycle", "mountain bike", "moped", "motor scooter", "rickshaw", "tricycle", "unicycle", "snowmobile"]),
    ("road_vehicles", ["ambulance", "amphibious vehicle", "wagon", "cab", "taxi", "convertible", "fire engine", "garbage truck", "go-kart", "golfcart", "jeep", "limousine", "minibus", "minivan", "model t", "pickup", "police van", "race car", "recreational vehicle", "school bus", "sports car", "tow truck", "tractor", "trailer truck", "truck", "car", "bus", "van", "tank", "forklift"]),
    ("buildings_places", ["altar", "apiary", "bakery", "barbershop", "barn", "boathouse", "bookshop", "castle", "church", "cinema", "cliff dwelling", "greenhouse", "library", "lumbermill", "monastery", "mosque", "palace", "planetarium", "prison", "restaurant", "shoe shop", "toyshop", "grocery store", "butcher shop"]),
    ("structures_monuments", ["bannister", "beacon", "bell cote", "breakwater", "dam", "dome", "fountain", "maze", "megalith", "obelisk", "pier", "pole", "totem pole", "viaduct", "vault", "steel arch bridge", "suspension bridge", "triumphal arch", "water tower", "stone wall", "worm fence", "picket fence", "chainlink fence", "screen", "stage", "patio", "terrace", "stupa", "theater curtain", "thatch", "tile roof", "gasmask", "solar dish"]),
    ("furniture_home", ["chair", "desk", "table", "sofa", "couch", "bed", "bookcase", "cabinet", "wardrobe", "crib", "cradle", "bassinet", "bathtub", "park bench", "pillow", "table lamp", "entertainment center"]),
    ("kitchenware", ["beaker", "beer bottle", "beer glass", "cocktail shaker", "coffee mug", "coffeepot", "crock pot", "dishwasher", "espresso maker", "frying pan", "goblet", "ladle", "measuring cup", "microwave", "mixing bowl", "plate", "pitcher", "pot", "saltshaker", "soup bowl", "teapot", "toaster", "waffle iron", "water bottle", "water jug", "wok", "wooden spoon", "cup", "can opener", "corkscrew", "strainer"]),
    ("tools_hardware", ["barometer", "binoculars", "broom", "bucket", "chain saw", "cleaver", "combination lock", "drill", "hammer", "hatchet", "hook", "iron", "lawn mower", "mailbox", "mailbag", "magnetic compass", "nail", "paintbrush", "plane", "plow", "screw", "screwdriver", "shovel", "swab", "syringe", "vacuum", "whistle", "rule", "ruler", "abacus", "balance beam", "barbell", "dumbbell", "croquet ball", "torch"]),
    ("electronics", ["cassette player", "cd player", "cellular telephone", "computer keyboard", "desktop computer", "digital clock", "digital watch", "dial telephone", "hard disc", "hand-held computer", "home theater", "ipod", "joystick", "laptop", "modem", "monitor", "mouse", "notebook computer", "oscilloscope", "pay-phone", "photocopier", "printer", "projector", "radio", "remote control", "screen", "television", "typewriter keyboard", "vending machine", "cassette", "tape player", "radio telescope"]),
    ("sports_equipment", ["baseball", "basketball", "football helmet", "golf ball", "ping-pong ball", "pool table", "puck", "punching bag", "racket", "rugby ball", "scoreboard", "ski", "soccer ball", "tennis ball", "volleyball", "croquet ball", "golfcart", "swab"]),
    ("containers", ["ashcan", "backpack", "barrel", "binder", "bottlecap", "carton", "chest", "crate", "envelope", "file cabinet", "packet", "plastic bag", "shopping basket", "shopping cart", "tray", "wallet", "purse", "pencil box", "pill bottle", "piggy bank", "box", "basket", "bag", "mailbox", "safe"]),
    ("wearables", ["abaya", "academic gown", "apron", "bikini", "bonnet", "bow tie", "bra", "cardigan", "cloak", "cowboy boot", "cowboy hat", "diaper", "fur coat", "gown", "jean", "jersey", "kimono", "lab coat", "loafer", "maillot", "mask", "miniskirt", "mitten", "neck brace", "necklace", "pajama", "poncho", "running shoe", "sandal", "sarong", "shoe", "sock", "sombrero", "stole", "suit", "sunglass", "sunglasses", "sweatshirt", "swimming trunks", "trench coat", "vestment", "wig", "tie", "helmet", "cap", "hat"]),
]


DEFAULT_OBJECT_CATEGORY = "household_misc"


def parse_imagenet_label(raw_label):
    english = raw_label.split("[")[0].strip()
    aliases = [part.strip() for part in english.split(",") if part.strip()]
    canonical = aliases[0]
    joined = " ".join(aliases).lower()
    return {
        "canonical_name": canonical,
        "aliases": aliases,
        "display_name": english,
        "search_text": joined,
    }


def assign_animal_superclass(class_id):
    for start, end, superclass_id in ANIMAL_RANGES:
        if start <= class_id <= end:
            return superclass_id
    return None


def matches_keywords(search_text, keywords):
    return any(keyword in search_text for keyword in keywords)


def assign_object_superclass(class_id, label_info):
    search_text = label_info["search_text"]
    for superclass_id, keywords in OBJECT_RULES:
        if matches_keywords(search_text, keywords):
            return superclass_id

    if 924 <= class_id <= 969:
        return "food_prepared"
    if 970 <= class_id <= 980:
        return "natural_landscapes"
    if 981 <= class_id <= 983:
        return "people_roles"
    if 984 <= class_id <= 990 or class_id == 998:
        return "produce_plants"
    if 991 <= class_id <= 997:
        return "fungi"
    return DEFAULT_OBJECT_CATEGORY


def build_default_superclass_map():
    buckets = {key: [] for key in SUPERCLASS_META}
    for class_id, raw_label in IMAGENET_1K_CLASSES.items():
        if class_id < 398:
            superclass_id = assign_animal_superclass(class_id)
        else:
            label_info = parse_imagenet_label(raw_label)
            superclass_id = assign_object_superclass(class_id, label_info)
        buckets[superclass_id].append(class_id)

    groups = []
    for superclass_id, class_ids in buckets.items():
        if not class_ids:
            continue
        meta = SUPERCLASS_META[superclass_id]
        groups.append(
            {
                "id": superclass_id,
                "name": meta["name"],
                "macro_domain": meta["macro_domain"],
                "description": meta["description"],
                "class_ids": sorted(class_ids),
            }
        )
    return groups


def save_default_superclasses(path=DATA_PATH):
    groups = build_default_superclass_map()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    return groups


def load_superclasses(path=DATA_PATH):
    if not path.exists():
        return save_default_superclasses(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_class_to_superclass(superclasses=None):
    superclasses = superclasses or load_superclasses()
    mapping = {}
    for superclass in superclasses:
        for class_id in superclass["class_ids"]:
            if class_id in mapping:
                raise ValueError(f"Duplicate superclass assignment for class_id={class_id}")
            mapping[class_id] = superclass
    expected = set(IMAGENET_1K_CLASSES.keys())
    actual = set(mapping.keys())
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Superclass mapping mismatch. missing={missing[:10]} extra={extra[:10]}")
    return mapping


def get_superclass_for_class_id(class_id, class_to_superclass=None):
    class_to_superclass = class_to_superclass or build_class_to_superclass()
    return class_to_superclass[int(class_id)]


def build_c2i_record(class_id, sample_id=None, include_negative=False):
    class_id = int(class_id)
    label_info = parse_imagenet_label(IMAGENET_1K_CLASSES[class_id])
    class_to_superclass = build_class_to_superclass()
    superclass = class_to_superclass[class_id]

    sample_id = sample_id or f"c2i_imagenet1k_{class_id:04d}"
    prompt = f"a photo of a {label_info['canonical_name']}"
    atoms = [
        {
            "question": f"Is the main subject in this image {superclass['description']}?",
            "answer": "Yes",
            "answer_type": "binary",
            "skill": "superclass",
            "weight": 1.0,
        }
    ]

    if include_negative:
        superclasses = load_superclasses()
        negative_candidates = [item for item in superclasses if item["macro_domain"] != superclass["macro_domain"]]
        if negative_candidates:
            negative = negative_candidates[class_id % len(negative_candidates)]
            atoms.append(
                {
                    "question": f"Is the main subject in this image {negative['description']}?",
                    "answer": "No",
                    "answer_type": "binary",
                    "skill": "negative_superclass",
                    "weight": 1.0,
                }
            )

    return {
        "sample_id": sample_id,
        "task_type": "c2i",
        "prompt": prompt,
        "condition": {
            "type": "class",
            "class_id": class_id,
            "class_name": label_info["canonical_name"],
            "class_display_name": label_info["display_name"],
            "superclass_id": superclass["id"],
            "superclass_name": superclass["name"],
            "macro_domain": superclass["macro_domain"],
        },
        "atoms": atoms,
        "metadata": {
            "class_id": class_id,
            "class_name": label_info["canonical_name"],
            "superclass_id": superclass["id"],
            "superclass_name": superclass["name"],
            "macro_domain": superclass["macro_domain"],
        },
    }
