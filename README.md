# Multimodal Binary Classifier

멀티모달 (Vision + Text) 기반 광고 이미지 이진 분류 시스템. 다양한 Vision Backbone과 Text Encoder를 지원하며, Hydra 기반 설정 관리로 유연한 실험이 가능합니다.

## Architecture

```
Multimodal:
  Image -> Vision Encoder (frozen/finetune/LoRA) -> Pooling -> ┐
                                                                 ├-> Fusion (GMU) -> MLP Head -> Binary Output
  Text  -> Text Encoder (frozen)                 -> ────────── ┘

Vision-only:
  Image -> Vision Encoder -> Pooling -> [Projection] -> L2 Norm -> MLP Head -> Binary Output

Text-only:
  Text -> Text Encoder -> [Projection] -> L2 Norm -> MLP Head -> Binary Output
```

## Project Structure

```
multimodal-classifier/
├── Dockerfile                  # Docker 이미지 빌드 (pytorch/pytorch:2.10.0-cuda13.0)
├── docker-compose.yml          # 컨테이너 실행 설정
├── .env.example                # 환경 변수 템플릿 (UID/GID, DATA_DIR)
├── requirements.txt            # Python 의존성
│
├── train.py                    # Hydra 기반 학습 entry point
├── test.py                     # 추론/평가 스크립트
├── lightning_module.py         # PyTorch Lightning 모듈
├── datamodule.py               # Lightning DataModule
├── dataset.py                  # CSV 데이터 로딩, transform
├── losses.py                   # Loss 함수 및 리포트 생성
│
├── model/                      # 모듈형 모델 컴포넌트
│   ├── classifier.py           # 메인 분류기 조립 (build_classifier)
│   ├── base.py                 # 추상 base 클래스 (VisionBackbone, TextEncoder)
│   ├── fusion.py               # 멀티모달 fusion (GMU, concat)
│   ├── pooling.py              # Pooling (mean, cls, attention)
│   ├── head.py                 # Classification MLP head
│   ├── lora.py                 # LoRA 어댑터
│   ├── vision/                 # Vision backbone
│   │   ├── siglip2.py          # SigLIP2 SO400M / Base
│   │   ├── siglip2_naflex.py   # SigLIP2 NaFlex
│   │   ├── resnet.py           # ResNet50/101
│   │   ├── vgg.py              # VGG16
│   │   ├── vit.py              # ViT Base/Large
│   │   └── efficientnet.py     # EfficientNet
│   └── text/                   # Text encoder
│       ├── bge_m3.py           # BGE-M3 (1024-dim)
│       └── sentence_transformer.py  # LaBSE (768-dim), E5-Large (1024-dim)
│
├── conf/                       # Hydra 설정 디렉토리
│   ├── config.yaml             # 메인 설정 (defaults 정의)
│   ├── data/default.yaml       # 데이터 경로 (DATA_DIR 환경 변수 사용)
│   ├── vision_encoder/         # Vision backbone 설정
│   ├── text_encoder/           # Text encoder 설정
│   ├── pooling/                # mean, cls, attention
│   ├── fusion/                 # gated (GMU), concat
│   ├── augmentation/           # none, weak, strong
│   ├── train/default.yaml      # 학습 하이퍼파라미터
│   └── experiment/             # 실험 프리셋
│
└── scripts/                    # 실험 실행/분석 스크립트
    ├── run_directFT_lightning.sh
    └── compare_results.py
```

## Setup

### Docker (권장)

```bash
# 1. 환경 변수 설정
cp .env.example .env
# .env 편집: UID/GID (id -u, id -g로 확인), DATA_DIR 경로 설정

# 2. 이미지 빌드 및 컨테이너 실행
docker compose up -d --build

# 3. 컨테이너 접속
docker compose exec torch-dev bash

# 4. 학습 실행 (/workspace에 프로젝트가 마운트됨)
python train.py +experiment=directFT_lightning_base
```

`.env` 예시:
```bash
USERNAME=ycahn
USER_UID=1013
USER_GID=1014
DATA_DIR=/home/ycahn/data
```

Docker 환경 구성:
- Base image: `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel`
- 프로젝트: `.` -> `/workspace` (bind mount)
- 데이터: `DATA_DIR` -> `/data` (bind mount)
- Python 의존성은 이미지 빌드 시 설치됨

### 호스트에서 직접 실행

```bash
pip install -r requirements.txt
export DATA_DIR=/path/to/your/data
python train.py
```

## Usage

```bash
# 기본 학습 (SigLIP2 SO400M + LaBSE + Attention Pooling + GMU Fusion)
python train.py

# Vision backbone 변경
python train.py vision_encoder=resnet50
python train.py vision_encoder=vit
python train.py vision_encoder=siglip2_base
python train.py vision_encoder=efficientnet

# Text encoder 변경/비활성화
python train.py text_encoder=bge_m3
python train.py text_encoder=e5_large
python train.py text_encoder=none

# Pooling / Fusion 변경
python train.py pooling=mean fusion=concat
python train.py pooling=attention fusion=gated

# 실험 프리셋 사용
python train.py +experiment=directFT_lightning_base
python train.py +experiment=directFT_lightning_1_allLayers_mean

# 학습 파라미터 override
python train.py train.batch_size=32 train.epochs=100 train.learning_rate=1e-4

# LoRA 활성화
python train.py train.lora.enabled=true train.lora.rank=32 train.lora.alpha=32

# Backbone fine-tuning (상위 N개 레이어 unfreeze)
python train.py train.freeze_backbone=false train.finetune_layers=6 train.backbone_lr=1e-5

# Hydra multirun (하이퍼파라미터 sweep)
python train.py -m vision_encoder=siglip2_so400m,resnet50 pooling=mean,attention

# 설정 확인 (학습 실행 없이)
python train.py --cfg job
```

## Data Format

CSV 파일에 `local_image_path`, `label`, `text_info` 컬럼 필요:

```csv
local_image_path,label,text_info
/data/images/img001.jpg,True,감지된 OCR 텍스트
/data/images/img002.jpg,False,
```

- `label`: `True` / `False`
- `text_info`: 이미지에서 추출된 OCR 텍스트 (없으면 빈 문자열)

데이터 경로는 `DATA_DIR` 환경 변수 또는 CLI override로 설정:
```bash
# 환경 변수
export DATA_DIR=/path/to/data

# 또는 CLI override
python train.py data.base_dir=/path/to/data
```

## Model Components

### Vision Backbone
| 모델 | 출력 차원 | Config |
|------|----------|--------|
| SigLIP2 SO400M | 1152 | `vision_encoder=siglip2_so400m` |
| SigLIP2 Base | 768 | `vision_encoder=siglip2_base` |
| SigLIP2 NaFlex SO400M | 1152 | `vision_encoder=siglip2_naflex_so400m` |
| SigLIP2 NaFlex Base | 768 | `vision_encoder=siglip2_naflex_base` |
| ResNet50 | 2048 | `vision_encoder=resnet50` |
| ResNet101 | 2048 | `vision_encoder=resnet101` |
| VGG16 | 512 | `vision_encoder=vgg16` |
| ViT Large | 768 | `vision_encoder=vit` |
| EfficientNetV2-L | 1280 | `vision_encoder=efficientnet` |

### Text Encoder
| 모델 | 출력 차원 | Config |
|------|----------|--------|
| LaBSE | 768 | `text_encoder=labse` |
| BGE-M3 | 1024 | `text_encoder=bge_m3` |
| E5-Large | 1024 | `text_encoder=e5_large` |
| 비활성화 | - | `text_encoder=none` |

### Multimodal Fusion
- **Gated (GMU)**: Gated Multimodal Unit - 샘플별 모달리티 가중치를 sigmoid gate로 학습
- **Concat**: L2 normalize + concatenation

### Training Features
- **AUROC 기반 모델 선택**: val_auroc 기준 best checkpoint 저장
- **LoRA fine-tuning**: SigLIP2 backbone (q_proj, k_proj, v_proj)에 적용
- **Layer-wise fine-tuning**: 상위 N개 레이어만 unfreeze (`finetune_layers`)
- **Parameter group LR**: backbone, LoRA, head에 별도 learning rate 설정 (`backbone_lr`, `lora_lr`)
- **Scheduler**: Cosine annealing with warmup (`cosine_warmup`), Cosine (`cosine`)
- **Loss**: BCE 또는 Focal loss (label smoothing 지원)
- **Logging**: Weights & Biases (W&B), TensorBoard
- **Reproducibility**: deterministic mode, seeded DataLoader

## Configuration

주요 학습 설정 (`conf/train/default.yaml`):

```yaml
modality: "multimodal"        # multimodal / vision_only / text_only
freeze_backbone: true
finetune_layers: 0            # 0=all frozen, N>0=상위 N개 unfreeze, -1=전체 unfreeze
lora:
  enabled: false
  rank: 16
  alpha: 8
  target_layers: 6            # SigLIP2 SO400M 기준 총 27 layers
scheduler_type: "cosine_warmup"
loss_type: "focal"
early_stopping:
  patience: 3
  metric: "val_auroc"
```

## Output

학습 결과는 `whole_finetune/<experiment_name>/<timestamp>/`에 저장:

```
whole_finetune/<experiment_name>/<timestamp>/
├── wandb/                       # W&B 로그
├── results.json                 # 테스트 메트릭
├── test_predictions.json        # 샘플별 예측
├── test_report.md               # 오답 우선 정렬 리포트
├── checkpoint_best.ckpt         # Best model checkpoint
└── .hydra/config.yaml           # Resolved configuration
```
