# Image Captioning Cli

This is a mod of [Wi-zz/joy-caption-pre-alpha](https://huggingface.co/Wi-zz/joy-caption-pre-alpha) and [fancyfeast/joy-caption-alpha-two](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two).

# Notice: I will contribute to Wi-zz after shaping the code.

## Overview

This application generates descriptive captions for images using advanced ML models. It processes single images or entire directories, leveraging CLIP and LLM models for accurate and contextual captions. It has NSFW captioning support with natural language. This is just an extension of the original author's efforts to improve performance. Their repo is located here: [https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two).

## Features

- Single image and batch processing
- Multiple directory support
- Custom output directory
- Adjustable caption generation options
- Progress tracking


## Install Required
Joy Caption Alpha Two: https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/tree/main

## Usage

```
python app_cli.py [--input_image INPUT_IMAGE | --input_dir INPUT_DIR] --output OUTPUT [options]
```

### Arguments

| Argument          | Description                                                                                                  |
|-------------------|--------------------------------------------------------------------------------------------------------------|
| `--input_image`   | Path to a single input image.                                                                                |
| `--input_dir`     | Path to a directory containing input images. Mutually exclusive with `--input_image`.                        |
| `--output`        | **(Required)** Path to the output directory where captions will be saved.                                   |
| `--caption_type`  | Type of caption to generate. Options: `descriptive`, `descriptive_informal`, `training_prompt`, `midjourney`, `booru_tag_list`, `booru_like_tag_list`, `art_critic`, `product_listing`, `social_media_post`. Default: `descriptive`. |
| `--caption_length`| Length of the caption (e.g., `any`, `very_short`, `short`, `medium_length`, `long`, `very_long`, or a specific number of words). Default: `long`. |
| `--extra_options` | Extra options to include in the caption generation by specifying their numeric indices. Available options are listed below. |
| `--name_input`    | Name of a person or character if applicable.                                                              |
| `--custom_prompt` | Custom prompt to override default settings.                                                                |

### Extra Options Mapping

| Option Number | Description                                                                                                      |
|---------------|------------------------------------------------------------------------------------------------------------------|
| `1`           | If there is a person/character in the image you must refer to them as `{name}`.                                   |
| `2`           | Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style). |
| `3`           | Include information about lighting.                                                                            |
| `4`           | Include information about camera angle.                                                                         |
| `5`           | Include information about whether there is a watermark or not.                                                  |
| `6`           | Include information about whether there are JPEG artifacts or not.                                             |
| `7`           | If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc. |
| `8`           | Do NOT include anything sexual; keep it PG.                                                                    |
| `9`           | Do NOT mention the image's resolution.                                                                          |
| `10`          | You MUST include information about the subjective aesthetic quality of the image from low to very high.          |
| `11`          | Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.        |
| `12`          | Do NOT mention any text that is in the image.                                                                    |
| `13`          | Specify the depth of field and whether the background is in focus or blurred.                                   |
| `14`          | If applicable, mention the likely use of artificial or natural lighting sources.                                |
| `15`          | Do NOT use any ambiguous language.                                                                               |
| `16`          | Include whether the image is sfw, suggestive, or nsfw.                                                            |
| `17`          | ONLY describe the most important elements of the image.                                                          |

### Examples

- **Process a single image:**

  ```bash
  python app_cli.py --input_image path/to/image.jpg --output path/to/output
  ```

- **Process all images in a directory:**

  ```bash
  python app_cli.py --input_dir path/to/directory --output path/to/output
  ```

- **Process multiple directories:**

  ```bash
  python app_cli.py --input_dir path/to/dir1 --input_dir path/to/dir2 --output path/to/output
  ```

- **Specify caption type and length:**

  ```bash
  python app_cli.py --input_dir path/to/directory --output path/to/output --caption_type art_critic --caption_length medium_length
  ```

- **Include extra options and custom prompt:**

  ```bash
  python app_cli.py --input_image image.jpg --output captions/ --extra_options 1 3 5 --custom_prompt "Provide a detailed art critique for this image."
  ```

- **Specify the name of a character in the image:**

  ```bash
  python app_cli.py --input_image image.jpg --output captions/ --name_input "Alice"
  ```

## Technical Details

- **Models**: CLIP (vision), LLM (language), custom ImageAdapter
- **Optimization**: CUDA-enabled GPU support
- **Error Handling**: Skips problematic images in batch processing

## Requirements

- Python 3.x
- PyTorch
- Transformers library
- PIL (Pillow)
- CUDA-capable GPU (recommended)

## Installation

### Windows

```bash
git clone https://huggingface.co/John6666/joy-caption-alpha-two-cli-mod
cd joy-caption-alpha-two-cli-mod
python -m venv venv
.\venv\Scripts\activate
# Change as per https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Linux

```bash
git clone https://huggingface.co/John6666/joy-caption-alpha-two-cli-mod
cd joy-caption-alpha-two-cli-mod
python3 -m venv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).
