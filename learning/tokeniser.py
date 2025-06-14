# need text to train the tokeniser on.
# tokeniser learns merge rules based on frequency of character pairs in data.

# example:

# i:1, s:2, is:3
# i and s are separate tokens, but you can merge them because they frequently
#appear together in many words, such as "is", "this", "dismiss" etc.
# this allows us to create new tokens and reduce computations.
# the new tokens can be further merged iteratively.

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

#find unique chars in dataset and sort
unique_chars = set()
for doc in corpus:
    for char in doc:
        unique_chars.add(char)

#convert to list so it's mutable
vocab = list(unique_chars)
vocab.sort()

### pre tokenisation
# need to add special end of word token to mark word boundaries.

eod = "</w>"
vocab.append(eod)

#print(len(vocab))

#using a dictionary that acts as a frequency table for the words
word_splits = {}
for doc in corpus:
    words = doc.split(' ')
    for word in words:
        if word:
            char_list = list(word) + [eod]
            #using tuple for immutability, so it can be used as a dictionary key
            word_tuple = tuple(char_list)
            if word_tuple not in word_splits:
                word_splits[word_tuple] = 0
            word_splits[word_tuple] += 1

#print(word_splits) #{('T', 'h', 'i', 's', '</w>'): 2,...}

###iterative merging
# take each word split and count the frequency of each adjacent pair of symbols across corpus
import collections

def get_pair_stats(splits):
    pair_counts = collections.defaultdict(int)
    
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        #iterate over characters, create pairs, add frequency
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_counts[pair] += freq

    return pair_counts

#print(get_pair_stats(word_splits)) #{('T', 'h'): 2, ('h', 'i'): 5, ...}

#merge the common pairs, e.g. "is" is common, so merge "i" and "s"

def merge_pair(pair_to_merge, splits):
    new_splits = {}
    (first, second) = pair_to_merge
    merged_token = first + second
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        new_symbols = []
        i = 0
        while i < len(symbols):
            #if current 2 symbols match pair to merge
            if i < len(symbols) -1 \
            and symbols[i] == first \
            and symbols[i+1] == second:
                new_symbols.append(merged_token)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
            
        new_splits[tuple(new_symbols)] = freq
    
    return new_splits

###iterative byte pair encoding merge loop
#calculate pair frequencies using get_pair_stats(), then choose top k pairs to inspect and then apply merge_pairs
num_merges = 15
merges = {} #store merge rules in here
current_splits = word_splits.copy()

for i in range(num_merges):
    print(f"iteration: {i+1}/{num_merges}")

    #calculate pair frequencies
    pair_stats = get_pair_stats(current_splits)
    if not pair_stats:
        print("no more pairs to merge")
        break
    
    sorted_pairs = sorted(pair_stats.items(), key= lambda x: x[1], reverse=True)
    print(f"top 5: {sorted_pairs[:5]}")
    #find best pair
    best_pair = max(pair_stats, key=pair_stats.get)
    best_freq = pair_stats[best_pair]
    print(f"best pair: {best_pair}, best freq: {best_freq}")

    #merge best pair across all word representations
    current_splits = merge_pair(best_pair, current_splits)
    new_token = best_pair[0] + best_pair[1]
    print(f"new splits: {current_splits}")
    
    #add new token to vocab, and add new merge rule to list
    vocab.append(new_token)
    merges[best_pair] = new_token

print("final results")
print(f"Final vocab size: {len(vocab)}")
print(f"merges learnt: {merges}")
# Final vocab size: 35
# merges learnt: {('s', '</w>'): 's</w>', ('i', 's</w>'): 'is</w>', ('t', 'h'): 'th', ('th', 'e'): 'the', ('the', '</w>'): 'the</w>', ('d', 'o'): 'do', ('do', 'c'): 'doc', ('doc', 'u'): 'docu', ('docu', 'm'): 'docum', ('docum', 'e'): 'docume', ('docume', 'n'): 'documen', ('documen', 't'): 'document', ('i', 'r'): 'ir', ('.', '</w>'): '.</w>', ('d', '</w>'): 'd</w>'}      