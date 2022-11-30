My playground.

당분간, 나는 호환되지 않는 변경을 할 것입니다.

## 종속성

- Python 3 (아마도 Python 3.6 또는 그 이상에서 작동됩니다.)
- [PyTorch](https://pytorch.org/get-started/locally/)
- requirements.txt 참고

일반적으로 최신 버전을 지원합니다. 버그나 호환성 문제가 있는 경우 버전을 지정합니다.

- [설치-우분투](INSTALL-ubuntu.md)
- [설치-윈도우](INSTALL-windows.md)

## waifu2x

리포지토리에는 waifu2x PyTorch 구현 및 사전 훈련된 모델이 포함되어 있습니다.
CLI 및 웹 API가 지원됩니다.
현재 학습이 시행되지 않습니다.

참고 [waifu2x/README.md](waifu2x/README.md)

## 마이그레이션

이전 리포지토리가 있는 경우 다음 명령을 사용하여 lfs 후크를 제거합니다.
(현재 이 저장소는 lfs를 사용하지 않습니다.)

```
git lfs uninstall
git lfs prune
```

`git lfs uninstall`은 전역적으로 영향을 미치는 것 같지만 `lfs`를 제대로 비활성화하는 방법을 모르겠습니다.
다시 복제하는 것이 좋습니다.
