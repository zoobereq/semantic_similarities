## Semantic Word Similarity

### Motivation
Computing word similarity is a fundamental problem in NLP and used in many applications such as plagiarism detection, question answering, and surverying diachronic language change.

### Method
The program implements and evaluates several methods of computing semantic word similarity:
- WordNet shortest-path similarity
- Wu-Palmer WordNet semantic depth similarity
- Word embeddings cosine similarity

### Code
The program first computes semantic similarity between the following six word pairs:
- *jaguar : cat*
- *jaguar : car*
- *king : queen*
- *king : rook*
- *tiger : zoo*
- *tiger : cat*

WordNet-based similarity scores are computed by selecting a pair of senses that yields the highest similarity score for both shortest-path and Wu-Palmer algorithms.  The cosine similarity is computed for dense high-dimensional vector representations derived from [GloVe Wiki Gigaword 50](https://nlp.stanford.edu/projects/glove/).  Users are free to implement different word embedding models.

The resulting similarity scores are then compared against human ratings, extracted from the [WordSimilarity-353 Test Collection](https://aclweb.org/aclwiki/WordSimilarity-353_Test_Collection_(State_of_the_art)).  Here again, users are free to implement their own baseline.

### Evaluation
The correlation between machine and human scores is expressed with the Spearman Correlation metric, first for the above-referenced six word pairs, and subsequently for 203 word pairs extracted from the WordSimilarity-353 Test Collection.



