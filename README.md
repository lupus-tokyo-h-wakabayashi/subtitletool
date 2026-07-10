# SubtitleTool

映画・海外ドラマ向けの日本語字幕作成ツールです。

Blu-rayなどに含まれる英語PGS字幕からOCRを行い、
ローカルLLM(Ollama)で自然な日本語字幕へ翻訳し、
日本語字幕付きMKVを生成します。

すべてローカル環境で完結します。

## Features

- MKVから英語PGS字幕を自動検出
- PGS字幕抽出
- OCR(PgsToSrt)
- OCR後の自動クリーンアップ
- Ollama(Qwen3)による日本語翻訳
- 前後文脈付き翻訳
- 作品別スタイル対応
- 用語集対応
- 日本語字幕付きMKV生成
- 途中保存対応
- ETA付き進捗表示

## Environment

Tested Environment

OS
- Ubuntu 24.04 (WSL2)

Python
- Python 3.13

.NET
- .NET 8 Runtime

FFmpeg
- 7.x

MKVToolNix
- mkvextract
- mkvmerge

Tesseract OCR
- Version 4.x

Ollama
- 0.9.x

LLM
- qwen3:14b

GPU
- NVIDIA GeForce RTX 5070 Ti 16GB

CPU
- Intel Core i9-12900KS

Memory
- 64GB


## Requirements

- ffmpeg
- ffprobe
- mkvtoolnix
- tesseract-ocr
- dotnet-runtime-8.0
- Ollama
- qwen3:14b


## Install
```
git clone ...
cd subtitletool
```
```
cp config/prompts/translate.example.txt \
    config/prompts/translate.txt
```
```
cp config/glossary/MOVIE_TITLE.example.txt \
    config/glossary/{MOVIE_TITLE}.txt
```
```
cp config/styles/common.example.txt \
   config/styles/common.txt
```
```
cp config/styles/MOVIE_TITLE.example.txt \
   config/styles/{MOVIE_TITLE}.txt
```


## Directories
subtitletool

    config/
        prompts/
        glossary/
        styles/

    src/
        commands/
        lib/
    
    bin/


## Flow
MKV

↓

scan

↓

extract

↓

OCR

↓

cleanup

↓

translate

↓

mux

↓

Japanese MKV



## Commands
```angular2html
subtitletool scan movie.mkv
subtitletool extract movie.mkv
subtitletool ocr movie.sup
subtitletool translate movie.eng.srt
subtitletool mux movie.mkv
subtitletool make movie.mkv
```

## Settings

### 翻訳プロンプト
- config/
- prompts/

### 作品ごとの用語集
- glossary/

### 作品ごとの翻訳スタイル
- styles/

## Quality
翻訳時は

・前後15字幕

・30字幕単位

で翻訳します。


## Architecture
MKV

↓

PGS

↓

OCR

↓

Cleanup

↓

Prompt Builder

↓

Ollama

↓

SRT

↓

MKV


