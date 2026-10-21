"""Domain constants for the Cosmetics & Skincare data pipeline.

These tables encode the deterministic dermatological and retail domain
knowledge used to engineer features from raw ingredient lists and product
attributes. They are intentionally frozen so that regeneration of the artifact
splits is fully reproducible.
"""

from __future__ import annotations

from typing import Final

ACTIVE_INGREDIENT_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "Niacinamide": ("niacinamide", "nicotinamide", "vitamin b3"),
    "Retinol": ("retinol", "retinal", "retinoate", "retinoid"),
    "Salicylic_Acid": ("salicylic acid", "salicylate", "bha"),
    "Vitamin_C": (
        "vitamin c",
        "ascorbic acid",
        "ascorbyl",
        "l-ascorbic",
        "tetrahexyldecyl ascorbate",
    ),
    "Hyaluronic_Acid": ("hyaluronic acid", "hyaluronate", "hyaluronan"),
    "Glycolic_Acid": ("glycolic acid", "glycolate", "aha"),
    "Ceramides": ("ceramide", "ceramides"),
    "Azelaic_Acid": ("azelaic acid", "azeloyl"),
}

COMEDOGENIC_LOOKUP: Final[dict[str, int]] = {
    "coconut oil": 4,
    "cocos nucifera": 4,
    "isopropyl myristate": 5,
    "isopropyl palmitate": 4,
    "myristyl myristate": 5,
    "lanolin": 4,
    "acetylated lanolin": 4,
    "cocoa butter": 4,
    "theobroma cacao": 4,
    "wheat germ oil": 5,
    "sodium lauryl sulfate": 5,
    "laureth-4": 5,
    "ethylhexyl palmitate": 4,
    "isostearyl isostearate": 4,
    "glyceryl stearate se": 3,
    "butyl stearate": 3,
    "algae extract": 5,
    "carrageenan": 5,
    "oleyl alcohol": 4,
    "oleic acid": 4,
    "hexadecyl alcohol": 2,
    "stearic acid": 2,
    "peg-16 lanolin": 4,
    "sodium laurate": 4,
}

FRAGRANCE_KEYWORDS: Final[tuple[str, ...]] = ("fragrance", "parfum", "perfume")

ALCOHOL_DENAT_KEYWORDS: Final[tuple[str, ...]] = ("alcohol denat", "sd alcohol")

NON_CLEAN_KEYWORDS: Final[tuple[str, ...]] = (
    "paraben",
    "sulfate",
    "phthalate",
    "formaldehyde",
    "triclosan",
    "oxybenzone",
)

SKIN_TYPES: Final[tuple[str, ...]] = ("oily", "dry", "combination", "sensitive")

SKIN_TONES: Final[tuple[str, ...]] = ("fair", "light", "medium", "tan", "deep")

SKINCARE_CATEGORIES: Final[tuple[str, ...]] = (
    "Moisturizer",
    "Serum",
    "Cleanser",
    "Sunscreen",
    "Toner",
    "Exfoliant",
)

PRICING_CATEGORIES: Final[tuple[str, ...]] = (
    "Skincare",
    "Makeup",
    "Hair",
    "Fragrance",
    "Bath & Body",
)

COMPATIBILITY_TIERS: Final[tuple[str, ...]] = (
    "Holy_Grail_Safe",
    "Neutral",
    "Irritation_Risk",
)

REAL_BRANDS: Final[tuple[str, ...]] = (
    "The Ordinary",
    "CeraVe",
    "La Roche-Posay",
    "Drunk Elephant",
    "Glow Recipe",
    "Kiehl's",
    "Clinique",
    "Paula's Choice",
    "Tatcha",
    "Sunday Riley",
    "Ole Henriksen",
    "Farmacy",
    "Innisfree",
    "Laneige",
    "First Aid Beauty",
    "Youth to the People",
    "Biossance",
    "Summer Fridays",
    "Caudalie",
    "Murad",
    "SkinCeuticals",
    "Naturium",
    "Good Molecules",
    "Byoma",
    "Versed",
    "Cocokind",
    "Peach & Lily",
    "Cosrx",
    "Beauty of Joseon",
    "EltaMD",
    "Supergoop!",
    "Hero Cosmetics",
    "La Mer",
    "Augustinus Bader",
    "Shani Darden",
)

SYNTHETIC_INGREDIENT_POOL: Final[tuple[str, ...]] = (
    "Water",
    "Glycerin",
    "Propanediol",
    "Niacinamide",
    "Hyaluronic Acid",
    "Sodium Hyaluronate",
    "Retinol",
    "Salicylic Acid",
    "Glycolic Acid",
    "Azelaic Acid",
    "Vitamin C",
    "Ascorbic Acid",
    "Ceramide NP",
    "Squalane",
    "Dimethicone",
    "Caprylic/Capric Triglyceride",
    "Shea Butter",
    "Aloe Barbadensis Leaf Juice",
    "Panthenol",
    "Allantoin",
    "Tocopherol",
    "Caffeine",
    "Zinc PCA",
    "Centella Asiatica Extract",
    "Fragrance",
    "Alcohol Denat.",
    "Phenoxyethanol",
    "Ethylhexylglycerin",
)
