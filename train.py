from tokeniser.tokeniser import Tokeniser

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

tokeniser = Tokeniser(merges=15)
tokeniser.train(corpus)

text = "is this document the third one?!"
tokeniser.encode(text)