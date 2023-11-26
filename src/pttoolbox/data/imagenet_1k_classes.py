from .imagenet_1k_lemmas import LEMMAS


target2name = tuple(item[1] for item in LEMMAS)
target2descr = tuple(item[2] for item in LEMMAS)
class2synsetid = tuple(item[0] for item in LEMMAS)
synset2target = {item[0]: num for num, item in enumerate(LEMMAS)}
synset2name = {item[0]: item[1] for item in LEMMAS}
