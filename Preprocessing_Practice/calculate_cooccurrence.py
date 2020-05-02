import numpy as np
from tqdm import tqdm
import pandas as pd

corpus = [
    'this is the first document',
    'ts is the second second document',
    'and the third moment document moment document',
    #     'Is this the first document?',
    #     'The last document?',
]

corpus = [i.split(' ') for i in corpus]


def cal_occ(corpus, window, words_selected):
    vocab = sorted(list(set(words_selected)))

    whole = []
    for lst in corpus:
        whole += lst
    whole = list(set(whole))
    vocab = [j for i, j in enumerate(vocab) if j in whole]
    vocab_not = [j for i, j in enumerate(vocab) if j not in whole]

    n = len(vocab)

    if n != len(words_selected):
        print('입력한 단어는 총 {0}개 였으나 이 중 {1}개는 corpus에 나타나지 않아서 제외되었음.'
              .format(len(words_selected), len(words_selected) - len(vocab)))
        print('제외된 단어들: '.format(vocab_not))

    co_occurr = np.zeros([n, n])
    for sent in tqdm(corpus):
        for i, word in enumerate(sent):
            n_sent = len(sent)
            for j in range(max(i - window, 0), min(i + window + 1, n_sent)):
                try:
                    row_idx = vocab.index(word)
                except:
                    row_idx = -1

                try:
                    col_idx = vocab.index(sent[j])
                except:
                    col_idx = -1

                if row_idx != -1 and col_idx != -1:
                    co_occurr[row_idx, col_idx] += 1
                else:
                    pass

    np.fill_diagonal(co_occurr, 0)

    co_occurr = pd.DataFrame(co_occurr, columns=vocab, index=vocab)

    return co_occurr


# toy example
print(cal_occ(corpus, 2, ['document', 'first']))