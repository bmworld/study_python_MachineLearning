
# 🧩 Anaconda 가상환경 & OS 개념 요약

## 💡 Anaconda 기본 명령
```
:: Anaconda Prompt(권장)에서 실행
 conda --version
 conda update -n base -c defaults conda

 :: (한번만) 채널 ToS 동의 필요하면
 conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
 conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

 :: 가상환경 생성
 conda create -n ai-env python=3.9 -y

 :: 활성화
 conda activate ai-env
 -> Terminal 왼쪽에 (ai-env) 표시됨

 :: 확인
 python --version
 where python
 conda info --envs
```


---

## 💡 가상환경 이해
· **가상환경(Virtual Env)**  
→ 프로젝트마다 독립된 Python 버전·패키지 환경을 분리 관리  
→ 충돌 없이 여러 버전 병행 가능 (AI, Web, Data 각각 분리)

· **설치 위치**  
`/opt/homebrew/anaconda3/envs/ai-env`  
(프로젝트 폴더 내부가 아니라, Anaconda의 envs 폴더에 생성)

---

## 💡 OS 기초 개념
· **UNIX** – 서버용 OS의 뿌리 (macOS·Linux의 기반)  
· **Windows** – 개인 PC 중심 OS (명령체계 다름)  
· **Linux** – 리누스 토르발스가 만든 오픈소스 UNIX 계열  
→ 자유로운 배포·수정 가능 (Ubuntu, RedHat 등 파생)

---

**요약 한 줄**
> Anaconda는 Python 환경을 관리하는 플랫폼이고,  
> 각 가상환경은 OS 위에서 독립 실행되는 ‘작은 개발 우주’다.
