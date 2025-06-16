#create tokeniser class, which will take in a corpus (list of documents) and output a vocab json file.
#maybe it should also take in a merge rule parameter, to modify how it merges.
#also need to add code to detokenise (take token id and turn it into text)
import collections

class Tokeniser:
    def __init__(self, merges=200):
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = merges
        self.merge_list = {}
        self.eod = "</w>"
        
    def train(self, corpus):
        self._initialise_vocab(corpus)
        current_splits = self._initialise_word_splits(corpus)
        self._update_vocab(current_splits)
        print("final results")
        print(f"Final vocab size: {len(self.vocab)}")
        print(f"merges learnt: {self.merge_list}")
        print(self.vocab)
        self._load_vocab_to_json()


    def encode(self, text):
        pass

    def decode(self, tokens):
        pass
    
    def _load_vocab_to_json(self):
        pass

    def _initialise_vocab(self, corpus):
        unique_chars = set()
        for doc in corpus:
            for char in doc:
                unique_chars.add(char)
        vocab = list(unique_chars)
        vocab.sort()
        vocab.append(self.eod)
        self.vocab = vocab

    def _initialise_word_splits(self, corpus):
        word_splits = {}
        for doc in corpus:
            words = doc.split(' ')
            for word in words:
                if word:
                    char_list = list(word) + [self.eod]
                    word_tuple = tuple(char_list)
                    if word_tuple not in word_splits:
                        word_splits[word_tuple] = 0
                    word_splits[word_tuple] += 1

        return word_splits

    def _update_vocab(self, current_splits):
        for i in range(self.merges):
            print(f"iteration: {i+1}/{self.merges}")

            #calculate pair frequencies
            pair_stats = self._get_pair_stats(current_splits)
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
            current_splits = self._merge_pair(best_pair, current_splits)
            new_token = best_pair[0] + best_pair[1]
            print(f"new splits: {current_splits}")
            
            #add new token to vocab, and add new merge rule to list
            self.vocab.append(new_token)
            self.merge_list[best_pair] = new_token

    def _get_pair_stats(self, splits):
        pair_counts = collections.defaultdict(int)
        
        for word_tuple, freq in splits.items():
            symbols = list(word_tuple)
            #iterate over characters, create pairs, add frequency
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pair_counts[pair] += freq

        return pair_counts

    def _merge_pair(self, pair_to_merge, splits):
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

