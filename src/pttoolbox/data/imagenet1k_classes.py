from .imagenet1k_lemmas import LEMMAS

TARGET2NAME = tuple(item[1] for item in LEMMAS)
TARGET2DESCRIPTION = tuple(item[2] for item in LEMMAS)
TARGET2SYNSET = tuple(item[0] for item in LEMMAS)
SYNSET2TARGET = {item[0]: num for num, item in enumerate(LEMMAS)}
SYNSET2NAME = {item[0]: item[1] for item in LEMMAS}


def synset2target(synset: str) -> int:
    """Convert synset to target."""
    return SYNSET2TARGET[synset]
