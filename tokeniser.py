# need text to train the tokeniser on.
# tokeniser learns merge rules based on frequency of character pairs in data.

# example:

# i:1, s:2, is:3
# i and s are separate tokens, but you can merge them because they frequently
#appear together in many words, such as "is", "this", "dismiss" etc.
# this allows us to create new tokens and reduce computations.
# the new tokens can be further merged.

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]
