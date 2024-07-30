## 목차
1. Choosing your base model
2. Iteration and prompt management
3. Testing
4. Deployment
5. Monitoring
6. Continual improvement and fine-tuning

## Iteration and prompt management
prompts + chain으로 작업한 뒤에 이 작업을 어떻게 관리해야할까?

(2015쯤) 과거의 전통적인 딥러닝은
- 매일 모델을 학습시키고, 스프레드시트에다가 하이퍼파라미터를 작성했다.
- 모델 파일을 laptop에 그냥 저장하고
- experiment를 다시 해보거나, 팀과 작업을 공유하고 싶을 때 방법이 없었다.

현재의 전통적인 딥러닝은
- 매일 model.train() 을 실행시키고
- comparable하고 shareable 하고 fully reproducible 한 experiment 로그를 받을 수 있음

현재의 프롬프트 엔지니어링은
- 매일 prompt를 변경하고, playground에서 일단 막 실험해보고
- 과거의 프롬프트는 시간이 지나면 사라지고
- experiment를 reproduce하거나, 작업을 팀원들과 공유할 수 있는 방법이 많이없다.

미래의 프롬프트 엔지니어링은
- ??

prompt engineering tool이 필요하긴한걸까?
왜 딥러닝에서 experiment management tool이 impactful 했을까?
- experiments를 한번 실행하는데 오랜 시간이 걸림. 그래서 go back하거나 state를 refresh하는 것은 중요함
- parallel하게 많은 실험을 실행해야할 수 있음
prompt engineering에서는 딥러닝과 동일한 dynamic가 필요없다.
- experiment가 빠르다.
- experiment가 주로 sequential하다. 프롬프트를 변경하거나 작업할 때 딥러닝 때처럼 다양한 작업을 한번에 병렬로 돌리지 않는다.

이게 최적의 방법은 아닐 것. 이렇게 하고 있는 이유가 prompt를 제대로 평가할 방법이 없어서 일듯함. 만약 서로 다른 두 프롬프트 중 뭐가 더 좋은지 비교할 수 있다면 다양한 prompt를 병렬로 돌릴 수 있을 것.

prompt/chain tracking
1. do nothing
2. track prompts in git
3. track prompts in specialized tool

experiment management tools are moving into prompt
- w&b, comet, mlflow, ..

recommendations
1. prompts 와 chains를 git에서 관리
2. 만약
- non-technical stakeholder와 협업해야하고
- prompt eval을 자동화하고 싶을 때
experiment mgt tool을 도입해보는 것을 추천


## Testing
새로운 모델 또는 prompt가 이전보다 좋아졌다는 것을 어떻게 measure할 수 있을까?
- 기존 딥러닝에서는 어떻게 했을까?
  - training에서는 train set과 eval set이 다를 때 overfitting을 찾아내고
  - production에서는 test set과 prod data의 acc가 다를 때 drift를 찾아냄
- 그럼 LLM 작업에서는?
  - training distribution에는 접근할 수 없다
  - production distribution은 거의 대부분 prompt training 때와는 다르다.

그럼 LLM test는 어떻게 하나? 두가지를 생각해보자
1. what data?
2. what metric?

building an evaluation dataset for your task
1. start incrementally
    - write a short story about {subject}
      - subject = dogs or linkedin or hats
      - ad-hoc으로 시작하기
      - interesting한 evaluation examples을 찾았다면 small dataset으로 organize할 것
        - evaluate하기 위해 모든 example에 대해 model을 실행시켜볼 것.
        - 이 때 interesting exmample이란 hard하고 diffirent해야한다. 
2. use your llm to help
    - LLM은 test case를 생성하는데 도움을 준다.
    - [auto-evaluator](https://autoevaluator.langchain.com/) 권장
3. 출시하고 나서도 더 많은 데이터를 추가. 아래 두가지 카테고리의 데이터를 모으면 좋을듯함.
    - hard data: 모델이 잘 처리하기 어려운 데이터
      - 유저가 싫어하는 것
      - annotator가 싫어하는 것
      - another 모델에게 물어봤을 때 싫어하는 것
    - different data: 기존 평가 데이터셋과는 다른 유형의 데이터 셋
      - 현재 eval set과 비교했을 때 outliers에 있는 것들
      - 많이 알려지지 않은 topic, intents, documents 등
4. toward test coverage for AI
    - 유저가 자주 쓰는 케이스로 test set을 만들고 test coverage를 높여야함
    - test coverage와 distribution shift와 비슷한 의미를 지님
      - distribution shift는 reference distribution으로 부터 test distribution이 얼마나 멀리 떨어져 있는지 보면서 dataset이 변한게 있는지 체크하는 과정이라면
      - test coverage는 eval set이 얼마나 production data를 잘 cover하고 있는지 보여주고, test coverage를 올리는 작업이 더 유용한 helpful한 eval 데이터를 찾는 과정으로 볼 수도 있겠다.

## Deployment & Monitoring
- 그냥 API 호출하면된다.
- 물론 좀 더 복잡하게
  - prompt construction이나 complicated chain 같이 좀 더 복잡하게 API call을 할 수도 있다.
  - service와 LLM 로직을 isolate하는 것을 권장
- 물론 open source LLM을 deploy 하는 것은 전혀 다른 이야기다.

Production에서 LLM output을 향상시키는 방법은?
- self-critique
  - 다른 LLM에게 right answer인지 물어봄
- sample many times 그리고 best option을 선택함. 또는 samples들을 emsemble.
  - [guardrails.ai](https://github.com/guardrails-ai/guardrails)
 
Monitoring
- outcomes and end-user feedback
  - 유저가 좋다면 장땡
  - 근데 어떻게 받을 것인가?
    - good feedback 이란 low friction + high signal
    - 가장 좋은건 user의 workflow안에 포함하는 것
      - accept changes 패턴
      - thumbs up / down 패턴
      - the role of longer-form feedback (수는 적지만 강력한 데이터 ex. 왜 응답이 부정확하다고 느꼈나요?)
- model performance metrics (if applicable)
- proxy metrics
  - ex) length of response (유저가 짧은 응답을 좋아하는지, 긴 응답을 좋아하는지)
- measuring what actually goes wrong

## Continual improvement and fine-tuning
유저 피드백을 반영하기 위한 actions
1. make the prompt better
2. fine-tuned model
    - supervised fine-tuning
      - 특정 task에 모델을 adapt시키고 싶거나, in-context learning이 제대로 동작하지 않을 때
      - 또는 데이터가 많을 때
      - 또는 모델을 smaller, chapter하게 만들고 싶을 때
    - fine-tuning from human feedback
      - techically complex하고 expensive하여 아직까지 많은 회사들이 도입하고 있진 않음.
