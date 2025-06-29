#create tokeniser class, which will allow  taking in a corpus (list of documents) and output a vocab json file.
import collections
import json

class Tokeniser:
    def __init__(self, merges=200):
        with open("./data/tokeniser_config.json", "r") as file:
            data = json.load(file)
            self.token_to_id = data.get("token_to_id", {})
            self.id_to_token = {int(key): value for key, value in data["id_to_token"].items()}
            self.merge_list = {tuple(pair): value for pair, value in data.get("merge_list", {})}
            self.eod = data.get("special_tokens", {}).get("eod", "</w>")
            self.unk = data.get("special_tokens", {}).get("unk", "</unk>")

        self.vocab = {}
        self.merges = merges
        
    def train(self, corpus):
        self.vocab = self._initialise_vocab(corpus)
        current_splits = self._initialise_word_splits(corpus)
        self.vocab, self.merge_list = self._update_vocab(self.vocab, current_splits)
        print("final results")
        print(f"Final vocab size: {len(self.vocab)}")
        #print(f"merges learnt: {self.merge_list}")
        self._save_to_json()


    def encode(self, text):
        #use token_to_id map to convert text into series of tokens
        words = text.split()
        word_splits = [list(word) + ['</w>'] for word in words]
        token_ids = []
        for word in word_splits:
            tokens = self._apply_merges(word, self.merge_list)
            for token in tokens:
                token = self.token_to_id.get(token, self.token_to_id["</unk>"])
                token_ids.append(token)

        #print(token_ids)
        return token_ids

    def decode(self, tokens_ids):
        token_list = []
        for id in tokens_ids:
            token = self.id_to_token.get(id, self.unk)
            token_list.append(token)
        concatenated_text = "".join(token_list)
        cleaned_text = concatenated_text.replace("</w>", " ")
        #print(cleaned_text)
        return cleaned_text
    
    def _save_to_json(self):
        if "</unk>" not in self.vocab:
            self.vocab.append("</unk>")
        token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        merges_list = [[list(pair), value] for pair, value in self.merge_list.items()]
        tokeniser_data = {
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "merge_list": merges_list,
            "special_tokens": {
                "eod": "</w>",
                "unk": "</unk>"
            }           
        }
        with open("./data/tokeniser_config.json", "w") as file:
            json.dump(tokeniser_data, file)

    def _initialise_vocab(self, corpus):
        unique_chars = set()
        for doc in corpus:
            for char in doc:
                unique_chars.add(char)
        vocab = list(unique_chars)
        vocab.sort()
        vocab.append(self.eod)
        return vocab

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

    def _update_vocab(self, vocab, current_splits):
        merge_list = {}
        for i in range(self.merges):
            if i % 100 == 0:
                print(f"iteration: {i+1}/{self.merges}")

            #calculate pair frequencies
            pair_stats = self._get_pair_stats(current_splits)
            if not pair_stats:
                #print("no more pairs to merge")
                break
            
            sorted_pairs = sorted(pair_stats.items(), key= lambda x: x[1], reverse=True)
            #find best pair
            best_pair = max(pair_stats, key=pair_stats.get)
            best_freq = pair_stats[best_pair]

            #merge best pair across all word representations
            current_splits = self._merge_pair(best_pair, current_splits)
            new_token = best_pair[0] + best_pair[1]
            
            #add new token to vocab, and add new merge rule to list
            vocab.append(new_token)
            merge_list[best_pair] = new_token

        return vocab, merge_list
        
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

    def _apply_merges(self, word, merges_set):
        word = word[:]
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            merge_candidate = None
            for pair in pairs:
                if pair in merges_set:
                    merge_candidate = pair
                    break
            if not merge_candidate:
                break
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == merge_candidate:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word