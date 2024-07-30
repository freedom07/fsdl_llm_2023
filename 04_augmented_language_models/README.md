## 목차
목표는 
- language model의 limited context length를 어떻게 개선할 것인가? 
- language model의 up-to-date 정보를 어떻게 반영할 것인가?

Augmented 된 language model을 만들어보자. 근데 그럼 Augmented language models이 뭘까? LLM에 Retrieval, Chain, Tools 기능이 추가된 것이다. 아래 목차로 자세하게 살펴보자.
1. **Retrieval** (Augment with a bigger corpus)
    - 1. why retrieval augmentation
    - 2. Traditional information retrieval
    - 3. Information retrieval via embeddings
      - 1. all about embeddings
      - 2. embedding relevance and indexes
      - 3. embedding databases
      - 4. beyond naive nearest neighbor
    - 4. Patterns and case studies
2. **Chain** (augment context with more LM calls)
3. **Tools** (augmented with outside sources)

## Retrieval
### Why retrieval augmentation
- context length에 한계가 있다.
- context building은 information retrieval의 한 종류다.
### Traditional information retrieval
- information retrieval은 query -> object -> relevance -> ranking 순서를 거친다.
- 우리가 알고 싶은 정보에 대한 query를 작성하고 document와 같은 content collection에서 우리가 얼마나 알고 싶은 정보인지에 대한 relevance를 계산하고 ranking을 매겨 결과를 가져온다. 어찌보면 너무 당연하다.
- 기존에는 inverted indexes를 사용해서 검색을 했다. 예를들어 document가 있으면 term으로 나누고 term과 frequency를 가지고 dictionary를 만들어 사용했다. elastic search에서 사용하는 방식이다.
- relevance는 boolean search를 통해 진행되는데 해당 term이 docs에 있는지 없는지 본다. 그리고 BM25와 같은 ranking을 매긴다.
- term frequency(TF), inverse docuemnt frequency(IDF), Field length와 같은 방법들이 있었다.
- search index는 문서를 검색할 수 있게 도와주는 data structure의 일종이고, search engines 그 보다 더 복잡한 기능들, 예를들어 document ingest, document processing, transaction handling(adding , deleting document, merging index files), scaling via shards, ranking & relevance algorithms 등이 모두 포함된다. 
- traditional search의 limitation: query, docuement, word 사이의 통계적인 연관성만 가지고 검색한다. 문서의 semantic information, 단어간의 cross correlations 등을 알지 못한다. 한 단어가 다양한 맥락에 의해 표현되는 것을 캐치하지 못한다.

### AI-powered information retrieval via embeddings
search engine을 통해 AI에게 더 나은 information in context을 제공해줄 수 있고 반대로 AI는 search engine에게 더 나은 data representation을 제공해줄 수 있다.

**All about embeddings**

임베딩이 뭘까? 임베딩은 abstract, dense, compact, fixed-size로 구성된 data의 representation이다. 문서를 보면 단어가 굉장히 sparse하게 있는 데 이것을 dense representation으로 변경해줄 수 있다. 임베딩은 벡터고, 벡터는 meaning을 가진다.

misconception of embedding
1. 임베딩은 항상 네트워크의 마지막 레이어일 필요가 없다.
2. 그리고 single input이 대부분 single embedding을 의미하나, 다 그런건 아니다.
3. embedding을 구하기 위한 다양한 technique이 존재하고 꼭 NN에서 구해지는건 아니다. 
4. embedding은 바로 vector space에 그려지는건 아니다.

왜 임베딩을 이용할까? data type에 상관없이 벡터로 표현할 수 있다. initial data가 얼마나 큰지, 어떤 타입인지에 상관없이 벡터로 표현할 수 있다는 것은 굉장히 강력한 아이디어다.

어떻게 좋은 임베딩을 만들까?
1. downstream task를 이용할 것
    - task가 중요하다. general purpose embedding은 LLM에서는 general purpose로 사용될 수 없다. benchmark embedding이 굉장히 중요하다. mteb 리더보드, best embedding 모델을 찾을 수 있다.
    - 그냥 openai embedding 쓰면 되는거 아니야?
2. 비슷한 것은 가까이 있어야함: 비슷한건 가까이 다른건 멀리

Embedding to know
1. the og: word2vec
2. the baseline: sentence transformers
    - cheap, fast to run
    - widely abailable
    - works decently well
3. a multimodal option: clip
    - text와 image의 embedding을 모두 구할 수 있음
4. the one to use: openai
    - text-embedding-ada-002
    - near-sota
    - easy to use, good results in pracice
    - pretty solid baseline으로 동작가능하다.
5. where things are going: instructor
    - mteb에서 현재 1등

off-the-shelf embeddings: good start but limited
  - off-the-embedding 모델이 off-the-shelf 에 대해 reliable하지 않는다.
  - instructor와 같은 접근 방법이 도움이 될 것
  - retrieval quality가 중요해지기 때문에, training하는 것을 피하지말자.

---

**Embedding relevance and indexes**

Embedding indexes는 data structure인데, approximate nearest neighbor search를 수행하도록 도와줌

Different index types are available. speed, scalability, accuracy 사이에 다양한 tradeoff 관계가 존재함

ANN index tool이 여러 있음.
- 어떤 index를 선택해야할까?
- 프로토타입 만들 땐 뭐로 하든지 상관없겠지만 production에서는 어떤 index를 선택하는지 보다 IR system을 선택하는게 더 중요하다
- faiss + hnsw

ANN limiation
-  ANN index는 단순히 data structure다.
  - hosting이 안되고
  - storing data / metadata를 vector와 함께 저장 할 수 없고
  - sparse와 dense 벡터를 combining하기 힘들고
  - embedding function을 스스로 메니징 하기 힘들고
  - vertical, horizontal scaling이 힘들고
- 그래서 이것들을 최대한 지원하는 IR system database를 선택하는게 중요하다.

---

**Embedding databases**

그럼 어떻게 search 할건가? embedding database vs just database
- elasticsearch, postgres(pgvector), redis run NN/ANN
- 아직까지 most complicated queries 또는 highest scale은 지원되지 않는 것 같다. PoC로 해볼만한듯.

dump in a bunch of data
- database stuff
  - scale, reliability
    - 데이터 중심 어플리케이션 설계 책 추천해줌
  - document splitting
    - split한 document가 너무 길면?
        1. seperator(ex. \n) 선택
        2. max size of chunk까지 text split
        3. (advanced) semantically consistent하게 chunk를 모아보자.
  - embedding mgmt(management)
    - 서로 다른 데이터 타입에 대해 어떤 임베딩을 써야하는가?
    - embedding function을 바꿀 때 어떻게 해야하는가?
- run a query : query language
  - "most similar documents" 는 쉽다.
  - recentcy 와 같은 다른 메타데이터를 함께 필터링하고 싶을땐?
  - query에 알맞는 document가 없는 경우 
    - a search string
    - request for a summary
    - etc...
- 가장 relevant 한 data 가져오기: search algorithm
  - 인덱스의 계층구조를 어떻게 다를 것인가?
    - 만약 모든 k-NN들이 같은 document으로 부터 온 chunk일 경우. (예를들어 유사한 chunk가 3개 추천되었는데 전부 같은 document일 경우)
    - what if all of the docs are form the same corpus?
    - what if all of the docs are old?

managed embedding databases
full text search를 제공하는 것은 지금까지 잘 사용해왔던 search engine이고 또 여전히 유용하고, reliable하다.

---

**Beyond naive nearest neighbor**

Problem: 내가 입력한 query와 관련된 document가 아니다.
- your queries are short question
- your docs are long form
- embeddings are not that comparable
  - embedding을 pretrain 할 때 사용된 데이터 셋이 내가 가지고 있는 데이터 셋과 전혀 무관하다.

Some approaches to addreess this
- neatest neighbor search로 좋은 결과를 얻지 못했다면 한번 볼만하다.

Problem: you might have some structure to your data
- whole index를 동시에 검색하는 것은 효율적이지 않다
- 대신에 데이터의 구조를 이용해서 검색하는 방법이 있다
  - 라마인덱스가 노션, 트위터 등에서 수집되는 데이터 셋의 임베딩을 만드는데 좀 더 hiearchical way로 검색하기 쉽게 해준다는데 한번 보자. 비슷했던 것 같은데..ㅎㅎ

### Patterns and case studies
[Copilot-explorer](https://github.com/thakkarparth007/copilot-explorer)

그래서 LLM은 external data와 함께 사용했을 때 more powerful 하고 많은 규칙과 heuristics들을 실제로 해봐야할 것. 

## Chain
chain은 complex reasoning을 도와주지만 token limit를 조심할 것. 이런 툴들로 LM 지식을 더 확장해보자. langchain의 chains 예시를 참고해보자.

## Tools
Toolformer: language model can teach themselves to use tools

chain based vs plugin based
