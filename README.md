# 🎵 AI Music Genre Classifier

음악 파일을 업로드하면 AI 모델이 음악의 특징을 분석하여  
**10개의 음악 장르 중 하나를 예측하는 프로젝트**입니다.

Random Forest 기반 머신러닝 모델부터 ResNet18 딥러닝 모델까지 구현하며  
**데이터 전처리 → 모델 학습 → 성능 개선 → 예측 → Streamlit 배포**의 전체 과정을 진행했습니다.

---

## 🌐 Live Demo

Streamlit에서 직접 음악 파일을 업로드하여 테스트할 수 있습니다.

👉 [🎵 AI Music 실행하기](https://ai-music-final-mj2gmrph3khrrncd5dbsgb.streamlit.app/)

---

## 🤖 사용 모델

| 단계 | 모델 | 주요 기능 |
|---|---|---|
| 6강 | Random Forest | 음악 특징 데이터를 이용한 기본 장르 분류 |
| 7강 | Random Forest | 신뢰도 분석, 멀티파일 비교, 예측 이력 |
| 8강 | ResNet18 | Mel-Spectrogram 이미지를 이용한 딥러닝 장르 분류 |

---

## 🧠 ResNet18 모델 개선

ImageNet 사전학습 ResNet18을 기반으로 GTZAN 음악 데이터를 학습하고  
데이터 증강 및 하이퍼파라미터 조정을 통해 최종 모델을 개선했습니다.

| 구분 | AS-IS | TO-BE |
|---|---:|---:|
| 모델 | ResNet18 | Fine-tuned ResNet18 |
| 검증 정확도 | 76.47% | **82.32%** |
| 성능 향상 | - | **+5.85%p** |
| Epoch | 10 | 조정 |
| Batch Size | 64 | 학습 조건 개선 |
| 데이터 증강 | 기본 | 적용 |

---

## 📈 모델 학습 과정

Epoch별 학습 결과를 그래프로 시각화하여 학습 과정의 변화를 확인할 수 있도록 구성했습니다.

- Validation Accuracy 변화
- Train Loss 변화
- Learning Rate 변화

기존 모델은 Epoch 9에서 최고 검증 정확도 **77.54%**를 기록했으며,  
최종 개선 모델은 검증 정확도 **82.32%**를 기록했습니다.

---

## 🎧 예측 방식

ResNet18 모델은 업로드된 음악을 **3초 단위로 분할**한 뒤  
각 구간을 Mel-Spectrogram 이미지로 변환하여 분석합니다.

각 구간의 ResNet18 예측 결과를 평균하여  
최종 음악 장르와 예측 확률을 결정합니다.

지원 장르:

`blues` · `classical` · `country` · `disco` · `hiphop` ·  
`jazz` · `metal` · `pop` · `reggae` · `rock`

---

## 🛠 주요 기술

- Python 3.11
- PyTorch
- Torchvision
- ResNet18
- Scikit-learn
- Librosa
- Pandas
- NumPy
- Matplotlib
- Streamlit
- GitHub

---

## 📦 필요한 패키지

프로젝트 실행에 필요한 Python 패키지는 `requirements.txt`에 정리되어 있습니다.

```bash
pip install -r requirements.txt

▶️ 실행 방법
streamlit run src/app.py

📁 프로젝트 구조

ai-music-final/
│
├── assets/
│   └── training_result.png
│
├── models/
│   ├── model_rf.joblib
│   ├── label_encoder.joblib
│   ├── scaler.joblib
│   └── resnet18_gtzan_final.pth
│
├── src/
│   └── app.py
│
├── README.md
├── requirements.txt
└── runtime.txt

🔄 프로젝트 진행 과정
1. Random Forest 기반 기본 음악 장르 예측

음악 파일에서 MFCC, Chroma, RMS, Spectral Centroid 등의 특징을 추출하고
Random Forest 모델을 이용해 음악 장르를 분류했습니다.

2. 고급 분석 기능 추가

기본 예측 기능에서 확장하여 다음 기능을 추가했습니다.

예측 신뢰도 확인
여러 음악 파일 비교
예측 이력 저장
Mel-Spectrogram 시각화
3. ResNet18 기반 딥러닝 모델 적용

음악 데이터를 3초 단위로 나누고 Mel-Spectrogram 이미지로 변환한 뒤
ResNet18을 이용해 이미지 기반 음악 장르 분류를 진행했습니다.

4. 모델 성능 개선

기존 ResNet18 모델을 기준으로 데이터 증강과 하이퍼파라미터 조정을 적용했습니다.

구분	AS-IS	TO-BE
모델	ResNet18	Fine-tuned ResNet18
검증 정확도	76.47%	82.32%
성능 향상	-	+5.85%p
Epoch	10	조정
Batch Size	64	학습 조건 개선
데이터 증강	기본	적용
5. 학습 과정 시각화

Epoch별 학습 결과를 그래프로 시각화하여 학습 과정의 변화를 확인할 수 있도록 구성했습니다.

Validation Accuracy 변화
Train Loss 변화
Learning Rate 변화
6. Streamlit 웹 서비스 배포

학습된 모델을 Streamlit 애플리케이션에 연결하여
사용자가 직접 WAV 파일을 업로드하고 음악 장르를 예측할 수 있도록 구현했습니다.

👉 🎵 AI Music 실행하기

🎯 프로젝트 결과

Random Forest 기반 머신러닝 음악 장르 분류에서 시작하여
ResNet18을 활용한 딥러닝 음악 장르 분류 모델까지 확장했습니다.

단순히 모델을 학습하고 예측하는 것에서 끝내지 않고
기존 모델과 개선 모델의 성능을 비교하고, 학습 과정을 그래프로 시각화하여 모델 개선 과정을 확인할 수 있도록 구성했습니다.

최종적으로 ResNet18 모델의 검증 정확도를
76.47% → 82.32%로 향상시켰으며,
Streamlit을 통해 실제 음악 파일을 업로드하여 예측할 수 있는 웹 서비스를 구현했습니다.
