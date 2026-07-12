# SubtitleTool

映画・海外ドラマ向けの日本語字幕作成ツールです。

Blu-rayなどに収録された英語PGS字幕をOCRでテキスト化し、
ローカルLLM（Ollama）を使用して自然な日本語字幕へ翻訳します。

翻訳結果は自動検証とリトライにより品質を向上させた後、
日本語字幕付きMKVとして出力します。

字幕の抽出・OCR・翻訳・MUXまで、すべてローカル環境で完結します。

## Features

- MKVから英語PGS字幕を自動検出
- PGS字幕の抽出
- PgsToSrtによるOCR
- OCR後の字幕テキスト自動クリーンアップ
- Ollama（Qwen3）による日本語翻訳
- 前後15字幕の文脈を考慮した翻訳
- 作品ごとの翻訳プロファイル対応
- 作品ごとの翻訳スタイル対応
- 作品ごとの用語集対応
- OCRノイズ辞書対応
- 翻訳途中からの再開（Resume）
- 翻訳結果の自動検証
- エラー内容に応じた自動リトライ
- 日本語字幕付きMKV生成
- Mux結果の自動検証
- ETA付き進捗表示
- エラー発生時のレスポンス保存
- すべてローカル環境で完結

## Requirements

SubtitleToolを実行するために、以下のソフトウェアが必要です。

- Python 3.13以降
- FFmpeg
    - ffmpeg
    - ffprobe
- MKVToolNix
    - mkvextract
    - mkvmerge
- Tesseract OCR
- PgsToSrt
- .NET 8 Runtime
- Ollama
- qwen3:14b

各コマンドはPATHから実行できる状態にしてください。

## Environment

以下の環境で動作確認を行っています。

### OS

- Ubuntu 24.04（WSL2）

### Python

- Python 3.13

### .NET

- .NET 8 Runtime

### FFmpeg

- FFmpeg 7.x

### MKVToolNix

- mkvextract
- mkvmerge

### OCR

- PgsToSrt
- Tesseract OCR 4.x

### Ollama

- Ollama 0.9.x

### LLM

- qwen3:14b

### GPU

- NVIDIA GeForce RTX 5070 Ti 16GB

### CPU

- Intel Core i9-12900KS

### Memory

- 64GB

## Install

リポジトリを取得します。

```bash
git clone <repository>
cd subtitletool
```

作品ごとの翻訳設定を作成します。

`config/default/` をコピーし、作品名など任意のプロファイル名へリネームしてください。

例

```bash
cp -r config/default config/stargate
```

必要に応じて以下の設定ファイルを編集してください。

### style.txt

翻訳スタイルを定義します。

例

- 口調
- 敬語
- 軍隊用語
- 世界観

### glossary.txt

作品固有の用語を定義します。

例

- 人名
- 地名
- 軍事用語
- 固有名詞

### noise.local.json

OCRで頻繁に発生するノイズを管理します。

翻訳中に自動で更新され、OCR品質の改善に利用されます。

翻訳モデルをインストールします。

```bash
ollama pull qwen3:14b
```

インストール後、各コマンドが利用できることを確認してください。

```bash
python3 --version
ffmpeg -version
ffprobe -version
mkvextract --version
mkvmerge --version
tesseract --version
dotnet --version
ollama --version
```

利用可能なモデルを確認します。

```bash
ollama list
```

## Directories

```
subtitletool
│
├── bin/
│   └── subtitletool
│
├── config/
│   ├── default/
│   │   ├── glossary.txt
│   │   ├── noise.json
│   │   └── style.txt
│   │
│   └── {PROFILE}/
│       ├── glossary.txt
│       ├── noise.local.json
│       └── style.txt
│
├── src/
│   ├── commands/
│   └── lib/
│
├── tests/
│
└── tmp/
    └── （実行時に生成）
```

### bin/

実行コマンドを配置します。

### config/default/

新しい翻訳プロファイルを作成するためのテンプレートです。

### config/{PROFILE}/

作品ごとの翻訳設定を管理します。

### src/commands/

CLIコマンドを実装します。

### src/lib/

各機能をライブラリとして実装します。

### tests/

回帰テストを配置します。

### tmp/

デバッグ用ファイルや、一時的な作業ファイルを出力します。

## Commands

SubtitleToolは各処理を個別に実行することも、一括で実行することもできます。

### 全処理を実行

英語PGS字幕の抽出から、日本語字幕付きMKVの生成までを一括で実行します。

```bash
subtitletool make movie.mkv
```

翻訳プロファイルを指定する場合は、`--profile`を指定します。

```bash
subtitletool make \
    movie.mkv \
    --profile stargate
```

### 個別実行

各処理を個別に実行することもできます。

```bash
subtitletool scan movie.mkv
subtitletool extract movie.mkv
subtitletool ocr movie.eng.sup
subtitletool translate movie.eng.srt
subtitletool mux movie.mkv
```

主に開発時やトラブルシューティング時の確認に利用します。

## Configuration

翻訳設定はプロファイル単位で管理します。

各作品ごとに専用ディレクトリを作成することで、作品ごとに翻訳品質を最適化できます。

```
config/
├── default/
├── stargate/
├── friends/
└── movie/
```

各プロファイルには次の設定ファイルがあります。

### style.txt

翻訳スタイルを定義します。

例

- 口調
- 敬語
- 話し方
- 軍隊用語
- 世界観

---

### glossary.txt

作品固有の用語を定義します。

例

- 人名
- 地名
- 組織名
- 軍事用語
- 固有名詞

---

### noise.local.json

OCRで頻繁に発生するノイズを管理します。

翻訳中に自動更新され、次回以降の翻訳品質向上に利用されます。

作品ごとに独立して管理されるため、他作品へ影響しません。

---

# Developer Guide

ここからはSubtitleToolの内部設計について説明します。

ライブラリ構成や翻訳パイプライン、Validation・Retry・Muxなど、開発者向けの内容を記載しています。

## Architecture

SubtitleToolは、各処理をライブラリ単位で責務を分離して実装しています。

```
MKV
 │
 ├─ Scan
 │
 ├─ Extract
 │
 ├─ OCR
 │
 ├─ Cleanup
 │
 ├─ Translate
 │     ├─ Prompt Builder
 │     ├─ Validation
 │     ├─ Retry
 │     └─ Resume
 │
 └─ Mux
       ├─ MuxPlan
       ├─ FFmpeg
       └─ Mux Validation
```

各処理は独立して実装されているため、単体テストやリファクタリングを容易に行える構成になっています。

## Translation Pipeline

翻訳はチャンク単位で実行されます。

```
OCR
 │
 ▼
Cleanup
 │
 ▼
Prompt Builder
 │
 ▼
Ollama
 │
 ▼
Translation Validation
 │
 ├─ OK
 │      │
 │      ▼
 │   Resume保存
 │      │
 │      ▼
 │   次チャンク
 │
 └─ NG
        │
        ▼
     Retry
        │
        ▼
 Translation Validation
```

### 翻訳設定

- 30字幕単位で翻訳
- 前後15字幕をコンテキストとして付与
- Resumeにより途中から再開可能
- Validationに失敗した場合は自動Retry
- Retryは最大3回まで実行

## Translation Validation

LLMの翻訳結果はそのまま採用せず、複数の観点から自動検証を行います。

### Validation順序

Validationは次の順番で実行されます。

1. JSON形式
2. 字幕ID
3. 字幕件数
4. 翻訳漏れ
5. 用語集違反
6. 中国語文字の混入
7. ラテン文字OCRノイズ
8. 未翻訳英文

この順序には意味があります。

上位のValidationほど翻訳結果全体の整合性を確認するものであり、
下位のValidationほど翻訳品質を改善するための検証になります。

新しいValidationを追加する場合も、この順序を基準として組み込み位置を決定します。

### Validation内容

現在実装しているValidationは次のとおりです。

| Validation           | 内容                     |
|----------------------|------------------------|
| JSON Format          | JSON形式の妥当性を確認します。      |
| Subtitle ID          | 字幕IDの欠落・重複・順序を確認します。   |
| Subtitle Count       | 字幕件数が一致していることを確認します。   |
| Missing Translation  | 翻訳漏れがないことを確認します。       |
| Glossary             | 用語集違反がないことを確認します。      |
| Chinese Characters   | 中国語文字が含まれていないことを確認します。 |
| Latin OCR            | OCRで破損したラテン文字列を検出します。  |
| Untranslated English | 未翻訳の英文が残っていないことを確認します。 |

Validationに失敗した場合は、エラー内容に応じたRetryプロンプトを生成し、自動で再翻訳を実行します。

正常な字幕は保持したまま問題のある字幕のみを修正することで、翻訳品質と処理時間の両立を実現しています。
