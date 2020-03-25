## Network Visualization on Text Data



### Why Gephi?

Gephi를 이용하여 텍스트 데이터에 네트워크 시각화를 하는 방법에 대해서 다룹니다. 네트워크 시각화 툴은 파이썬 library로도 있고 node.js를 이용할 수도 있는 것 같은데, 여기서는 Gephi를 이용해서 시각화를 해볼 것입니다. 그 이유는, 파이썬 library 보다는 자유도가 높고, node.js보다는 쉽기 때문입니다. node.js는 C나 Java 기반이기 때문에, 시각화를 하기 위해 또 다른 언어를 배워야한다는 불편함이 있습니다. 반면에, Gephi는 클릭만으로 원하는 것과 가까운 결과물을 얻을 수 있다는 점이 매력 포인트입니다.



### How to Process?

$d$개의 문서가 있는 corpus를 상상해봅시다. 여기서 단어를 뽑아내고, 단어들간의 관계를 추출하여 이를 네트워크 시각화의 입력 행렬로 집어 넣습니다. 그리고 Gephi에서 원하는 설정을 통해 네트워크 시각화를 완성합니다. 과정은 아래와 같습니다.

* corpus로부터 단어 간의 관계를 나타내는 행렬을 뽑습니다. 가장 기본적으로는 co-occurrence 행렬이 있습니다. co-occurrence 행렬은 그 정의상, 대칭 행렬입니다.
  * co-occurrence 행렬도 종류가 여러개 있습니다. 하나의 단어 (unigram)만 볼 것인지, 또는 두 단어 (bi-gram)을 볼 것인지를 먼저 정해야 합니다.
  * unigram으로 단위를 정했다고 합시다. 그러면 이 단어 주위로 몇 개의 단어 이내에 들어온 단어들을 중심 단어와 함께 발생(co-occurr) 했다고 여기는지, 그 기준을 정해야 합니다. 이를 'window'라고 부릅니다. 만약 window=2라고 하면, 중심 단어 기준으로 앞, 뒤 두 단어씩 보고, 이 window에 있는 단어들이 중심 단어와 함께 일어났다고 보는 것이죠.
  * window를 너무 넓게 설정하면 중심 단어와 같이 일어났다고 보기 힘든 단어들도 포함하고, 너무 좁게 설정하면 중심 단어의 semantic meaning을 잘 파악하지 못할 수도 있으므로 적절한 window 설정이 필요합니다.

* 이렇게 행렬을 만들었으면, Gephi에서 시각화를 하면 끝입니다.



### Contents

순서는 아래와 같습니다.

- [ ] co-occurrence 행렬 구축하기
- [ ] Gephi 소개
- [ ] pagerank, eigenvalue centrality 소개
- [ ] Gephi에서 네트워크 custom 설정