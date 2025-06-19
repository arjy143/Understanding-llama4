# Understanding-llama4
Creating llama4 from scratch. This was inspired by https://www.youtube.com/watch?v=biveB0gOlak&pp=ygUUbGxhbWEgNCBmcm9tIHNjcmF0Y2g%3D


I created a simple version of llama4 only using pytorch and basic python libraries. 

The create_tokeniser_config.py script loads a wikipedia dataset and creates a vocabulary for the model.

The training_loop.py script uses a subset of the wikipedia dataset to learn relationships between tokens, using the transformer block.

The test.py script uses another subset of the wikipedia dataset to see how well the model generates text based on the input.

Currently it's only trained on a small subset (data[:10]) of the wikipedia data, but it is able to output semi-coherent sentences. Maybe training it on a larger dataset on better hardware would have better results.
