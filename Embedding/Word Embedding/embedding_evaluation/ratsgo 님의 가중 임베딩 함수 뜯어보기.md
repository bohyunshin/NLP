# ratsgo 님의 가중 임베딩 함수 뜯어보기

### Reference

* ratsgo님의 cbow 함수([여기](https://github.com/ratsgo/embedding/blob/master/models/word_utils.py))
* a simple but tough to beat baseline for sentence embeddings, Sanjeev Arora et al.

한국어 임베딩 책에서 가중 임베딩이라는 것을 접했다. 단어 임베딩 벡터를 이용해서 문장의 임베딩 벡터를 생성하는데, 단순 합이 아니라 가중 합으로 구한다. 여기서는 ratsgo님이 구현하신 가중 임베딩 파이썬 코드를 하나 하나 뜯어본다.

먼저 CBoWModel 클래스를 선언한다.

```python
class CBoWModel(object):

    def __init__(self, train_fname, embedding_fname, model_fname, embedding_corpus_fname,
                 embedding_method="fasttext", is_weighted=True, average=False, dim=100, tokenizer_name="mecab"):
```

입력 인자 중 살펴봐야할 것은 is_weighted이다. True라면 논문에서와 같이 가중합으로 문장 벡터를 구하고 아니라면 일반 합으로 구한다. 가중 임베딩이 얼마나 효과가 있는지 알아보기 위함이다. 

```python
class CBoWModel(object):
    def __init__():
		if is_weighted:
            # ready for weighted embeddings
            self.embeddings = self.load_or_construct_weighted_embedding(embedding_fname, embedding_method, embedding_corpus_fname)
            print("loading weighted embeddings, complete!")
        else:
            # ready for original embeddings
            words, vectors = self.load_word_embeddings(embedding_fname, embedding_method)
            self.embeddings = defaultdict(list)
            for word, vector in zip(words, vectors):
                self.embeddings[word] = vector
            print("loading original embeddings, complete!")
```

만약 전자라면, `load_or_construct_weighted_embedding` 함수를 이용하여 임베딩을 로딩한다. 그렇지 않다면 별도 처리를 하지 않고 그대로 읽어 들인다.

```python
class CBoWModel(object):
	def compute_word_frequency(self, embedding_corpus_fname):
            total_count = 0
            words_count = defaultdict(int)
            with open(embedding_corpus_fname, "r") as f:
                for line in f:
                    tokens = line.strip().split()
                    for token in tokens:
                        words_count[token] += 1
                        total_count += 1
            return words_count, total_count
```

위 코드는 임베딩 학습 말뭉치에 쓰인 모든 문장, 모든 단어의 빈도를 하나 하나 센다. 입력 데이터로 형태소 분석이 완료된 데이터가 들어간다고 가정한다.
defaultdict은 키 값의 기본 값으로 0을 받는 함수이다. 즉,

```python
from collections import defaultdict
int_dict = defaultdict(int)
int_dict['이것은 defaultdict 예시입니다.']
```

위와 같이 value는 지정하지 않고 key 값만 지정한다면 해당하는 value 값은 자동으로 0이 된다. 여기서는 단어들의 기본 value 값으로 0을 지정하겠다는 뜻이다.

```python
class CBoWModel(object):
	def load_or_construct_weighted_embedding(self, embedding_fname, embedding_method, embedding_corpus_fname, a=0.0001):
        
            ##### 생략 ####
            
                with open(embedding_fname + "-weighted", "w") as f3:
                    for word, vec in zip(words, vecs):
                        if word in words_count.keys():
                            word_prob = words_count[word] / total_word_count
                        else:
                            word_prob = 0.0
                        weighted_vector = (a / (word_prob + a)) * np.asarray(vec)
                        dictionary[word] = weighted_vector
                        f3.writelines(word + "\u241E" + " ".join([str(el) for el in weighted_vector]) + "\n")
            return dictionary
```

다음으로는 각 단어 벡터의 등장 확률을 반영하여 가중치를 곱해준다. 등장 확률은 `word_prob = words_count[word] / total_word_count`로 구하고 가중치는 `(a / (word_prob + a))`로 구한다. 

```python
class CBoWModel(object):
	def train_model(self, train_data_fname, model_fname):
        model = {"vectors": [], "labels": [], "sentences": []}
        train_data = self.load_or_tokenize_corpus(train_data_fname)
        with open(model_fname, "w") as f:
            for sentence, tokens, label in train_data:
                tokens = self.tokenizer.morphs(sentence)
                sentence_vector = self.get_sentence_vector(tokens)
                model["sentences"].append(sentence)
                model["vectors"].append(sentence_vector)
                model["labels"].append(label)
                str_vector = " ".join([str(el) for el in sentence_vector])
                f.writelines(sentence + "\u241E" + " ".join(tokens) + "\u241E" + str_vector + "\u241E" + label + "\n")
        return model
```

위 코드는 학습 과정을 정의한 함수이다. `sentence_vector = self.get_sentence_vector(tokens)`을 통해 tokens들의 임베딩 벡터들을 문장 벡터로 합친다. for문을 통해 모든 훈련 데이터에 대해서, 문장은 `model["sentences"].append(sentence)`으로, 문장 벡터는 `model["vectors"].append(sentence_vector)`으로, 라벨은 `model["labels"].append(label)`으로 저장한다.

```python
class CBoWModel(object):
        def get_sentence_vector(self, tokens):
        vector = np.zeros(self.dim)
        for token in tokens:
            if token in self.embeddings.keys():
                vector += self.embeddings[token]
        if not self.average:
            vector /= len(tokens)
        vector_norm = np.linalg.norm(vector)
        if vector_norm != 0:
            unit_vector = vector / vector_norm
        else:
            unit_vector = np.zeros(self.dim)
        return unit_vector

```

`self.embeddings`에 있는 임베딩 벡터들을 모두 더하고 `self.average`가 True이면 평균을 낸다. 또한 예측 단계에서 코사인 유사도를 계산하기 편하도록 벡터의 norm으로 나눈다.

```python
class CBoWModel(object):
        def predict(self, sentence):
        tokens = self.tokenizer.morphs(sentence)
        sentence_vector = self.get_sentence_vector(tokens)
        scores = np.dot(self.model["vectors"], sentence_vector)
        pred = self.model["labels"][np.argmax(scores)]
        return pred
```

`predict` 함수에서는 테스트 문장의 라벨을 예측한다. 테스트 문장이 들어오면 형태소 분석을 하고, `get_sentence_vector`을 통해서 문장 임베딩 벡터를 뽑아낸다. 그리고 `model['vector']` (훈련 데이터의 문장 임베딩 벡터)와 내적을 하는데, 이 벡터들은 정규화가 되었으므로 내적이 곧 코사인 유사도 값이다. 따라서 내적을 통해 구한 `scores`가 가장 큰 label (argmax)을 구하면, 입력으로 들어온 테스트 문장과 코사인 유사도가 가장 큰 훈련 문장의 라벨을 구하는 것이다.

```python
class CBoWModel(object):
        def predict_by_batch(self, tokenized_sentences, labels):
        sentence_vectors, eval_score = [], 0
        for tokens in tokenized_sentences:
            sentence_vectors.append(self.get_sentence_vector(tokens))
        scores = np.dot(self.model["vectors"], np.array(sentence_vectors).T)
        preds = np.argmax(scores, axis=0)
        for pred, label in zip(preds, labels):
            if self.model["labels"][pred] == label:
                eval_score += 1
        return preds, eval_score

```

`predict_by_batch` 함수는 테스트 데이터 여러개에 대한 라벨을 계산한다.