import argparse
import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM
)
from huggingface_hub import InferenceClient

# Path to the CLIP model
CLIP_PATH = "google/siglip-so400m-patch14-384"

# Path to the checkpoint directory
CHECKPOINT_PATH = Path("cgrkzexw-599808")

# Supported image formats
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}  # Supported image formats

# Mapping of caption types to their respective prompt templates
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive Informal": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

# Hugging Face token from environment variables
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Define Extra Options Mapping
EXTRA_OPTIONS_MAP = {
    1: "If there is a person/character in the image you must refer to them as {name}.",
    2: "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    3: "Include information about lighting.",
    4: "Include information about camera angle.",
    5: "Include information about whether there is a watermark or not.",
    6: "Include information about whether there are JPEG artifacts or not.",
    7: "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    8: "Do NOT include anything sexual; keep it PG.",
    9: "Do NOT mention the image's resolution.",
    10: "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    11: "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    12: "Do NOT mention any text that is in the image.",
    13: "Specify the depth of field and whether the background is in focus or blurred.",
    14: "If applicable, mention the likely use of artificial or natural lighting sources.",
    15: "Do NOT use any ambiguous language.",
    16: "Include whether the image is sfw, suggestive, or nsfw.",
    17: "ONLY describe the most important elements of the image."
}

class ImageAdapter(nn.Module):
    """
    A neural network module to adapt image features for the language model.
    """

    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract

        # Adjust input features if deep extraction is enabled
        if self.deep_extract:
            input_features = input_features * 5

        # Define linear layers for transformation
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)

        # Layer normalization if enabled
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)

        # Positional embeddings for image tokens if enabled
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

        # Embedding layer for special tokens
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)   # Initialize embeddings

    def forward(self, vision_outputs: torch.Tensor):
        """
        Forward pass to adapt vision outputs.
        """
        if self.deep_extract:
            # Concatenate multiple layers of vision outputs for deep extraction
            x = torch.concat((
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ), dim=-1)
            # Ensure the tensor has the correct shape
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            # Use the second last layer of vision outputs
            x = vision_outputs[-2]

        # Apply layer normalization if enabled
        x = self.ln1(x)

        # Add positional embeddings if available
        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        # Pass through linear layers and activation
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        # Embed special tokens: <|image_start|> and <|image_end|>
        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1)
        )
        assert other_tokens.shape == (x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        # Concatenate special tokens with image embeddings
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        """
        Get the embedding for the end-of-text token.
        """
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

def load_models(device: torch.device):
    print("Loading CLIP")
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    clip_model = AutoModel.from_pretrained(CLIP_PATH)
    clip_model = clip_model.vision_model

    # Verify and load custom vision model checkpoint
    assert (CHECKPOINT_PATH / "clip_model.pt").exists()
    print("Loading VLM's custom vision model")
    checkpoint = torch.load(CHECKPOINT_PATH / "clip_model.pt", map_location=device)
    # Adjust keys in the state dictionary
    checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
    clip_model.load_state_dict(checkpoint)
    del checkpoint

    # Set CLIP model to evaluation mode and move to device
    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_model.to(device)

    # Load tokenizer for the text model
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH / "text_model", use_fast=True)
    assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

    # Load language model (LLM)
    print("Loading LLM")
    print("Loading VLM's custom text model")

    if device.type == 'cuda':
        device_map = {"": 0}
        torch_dtype = torch.bfloat16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32

    text_model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH / "text_model",
        device_map=device_map,  # Adjust device_map based on availability
        torch_dtype=torch_dtype
    )
    text_model.eval()
    text_model.to(device)

    # Initialize Image Adapter and load its state
    print("Loading image adapter")
    image_adapter = ImageAdapter(
        clip_model.config.hidden_size,
        text_model.config.hidden_size,
        False,
        False,
        38,
        False
    )
    image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location=device))
    image_adapter.eval()
    image_adapter.to(device)

    return clip_model, image_adapter, tokenizer, text_model

def stream_chat(
    clip_model: nn.Module,
    image_adapter: nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    text_model: AutoModelForCausalLM,
    input_image_path: str,
    caption_type: str,
    caption_length: str,
    extra_options: list[int],
    name_input: str,
    custom_prompt: str,
    device: torch.device
) -> tuple[str, str]:
    """
    Generate a caption for the given image based on the selected options.
    """
    # Clear CUDA cache to free up memory
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    # Determine the length of the caption
    length = None if caption_length == "any" else caption_length

    if isinstance(length, str):
        try:
            length = int(length)
        except ValueError:
            pass

    # Select the appropriate prompt template based on caption length
    if length is None:
        map_idx = 0
    elif isinstance(length, int):
        map_idx = 1
    elif isinstance(length, str):
        map_idx = 2
    else:
        raise ValueError(f"Invalid caption length: {length}")

    prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

    # Map numeric extra options to their corresponding strings
    selected_extra_options = [EXTRA_OPTIONS_MAP.get(opt, "") for opt in extra_options]
    # Remove any empty strings in case of invalid indices
    selected_extra_options = list(filter(None, selected_extra_options))

    # Append extra options to the prompt if any
    if len(selected_extra_options) > 0:
        prompt_str += " " + " ".join(selected_extra_options)

    # Format the prompt with name, length, and word count if applicable
    prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    # Override prompt if a custom prompt is provided
    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()

    # Debugging: Print the final prompt
    print(f"Prompt: {prompt_str}")

    # Preprocess the input image
    image = Image.open(input_image_path).convert("RGB")
    image = image.resize((384, 384), Image.LANCZOS)
    pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
    pixel_values = pixel_values.to(device)

    # Embed the image using CLIP and the Image Adapter
    with torch.autocast(device.type, enabled=device.type == 'cuda'):
        vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
        embedded_images = image_adapter(vision_outputs.hidden_states)
        embedded_images = embedded_images.to(device)

    # Build the conversation history for the language model
    convo = [
        {
            "role": "system",
            "content": "You are a helpful image captioner.",
        },
        {
            "role": "user",
            "content": prompt_str,
        },
    ]

    # Format the conversation without tokenization and with a generation prompt
    convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    assert isinstance(convo_string, str)

    # Tokenize the conversation and the prompt separately
    convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
    prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False)
    assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
    convo_tokens = convo_tokens.squeeze(0).to(device)   # Remove batch dimension and move to device
    prompt_tokens = prompt_tokens.squeeze(0).to(device)

    # Identify the positions of the end-of-text tokens
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eot_id_indices = (convo_tokens == eot_id).nonzero(as_tuple=True)[0].tolist()
    assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

    # Calculate the length of the preamble before the prompt
    preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]

    # Embed the conversation tokens using the text model
    convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0))

    # Construct the input embeddings by inserting the image embeddings
    input_embeds = torch.cat([
        convo_embeds[:, :preamble_len],                      # Tokens before the prompt
        embedded_images.to(dtype=convo_embeds.dtype),        # Image embeddings
        convo_embeds[:, preamble_len:],                      # Prompt and subsequent tokens
    ], dim=1).to(device)

    # Construct input IDs with dummy tokens for image embeddings
    input_ids = torch.cat([
        convo_tokens[:preamble_len].unsqueeze(0),
        torch.zeros((1, embedded_images.shape[1]), dtype=torch.long, device=device),   # Dummy tokens
        convo_tokens[preamble_len:].unsqueeze(0),
    ], dim=1).to(device)

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    # Debugging: Print the input to the model
    decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Input to model: {decoded_input}")

    # Generate the caption using the language model
    generate_ids = text_model.generate(
        input_ids,
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        max_new_tokens=300,
        do_sample=True,
        suppress_tokens=None
    )   # Uses default settings: temp=0.6, top_p=0.9

    # Remove the prompt tokens from the generated output
    generate_ids = generate_ids[:, input_ids.shape[1]:]
    if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids("<|eot_id|>"):
        generate_ids = generate_ids[:, :-1]

    # Decode the generated tokens to obtain the caption
    caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    return prompt_str, caption.strip()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate captions for images using JoyCaption Alpha Two.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_image", type=str, help="Path to the input image.")
    group.add_argument("--input_dir", type=str, help="Path to the directory containing input images.")

    # Define caption type choices without spaces
    CAPTION_TYPE_CHOICES = {
        "descriptive": "Descriptive",
        "descriptive_informal": "Descriptive Informal",
        "training_prompt": "Training Prompt",
        "midjourney": "MidJourney",
        "booru_tag_list": "Booru tag list",
        "booru_like_tag_list": "Booru-like tag list",
        "art_critic": "Art Critic",
        "product_listing": "Product Listing",
        "social_media_post": "Social Media Post",
    }

    parser.add_argument(
        "--caption_type",
        type=str,
        choices=list(CAPTION_TYPE_CHOICES.keys()),
        default="descriptive",
        help="Type of caption to generate. Options: " + ", ".join(CAPTION_TYPE_CHOICES.keys())
    )
    parser.add_argument(
        "--caption_length",
        type=str,
        default="long",
        help="Length of the caption (e.g., any, very_short, short, medium_length, long, very_long, or a specific number of words)."
    )
    parser.add_argument(
        "--extra_options",
        type=int,
        nargs='*',
        choices=list(EXTRA_OPTIONS_MAP.keys()),
        default=[],
        help="Extra options to include in the caption generation by specifying their numeric indices. Available options:\n" +
             "\n".join([f"{k}: {v}" for k, v in EXTRA_OPTIONS_MAP.items()])
    )
    parser.add_argument(
        "--name_input",
        type=str,
        default="",
        help="Name of a person or character if applicable."
    )
    parser.add_argument(
        "--custom_prompt",
        type=str,
        default="",
        help="Custom prompt to override default settings."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory where captions will be saved."
    )

    args = parser.parse_args()

    # Map the caption type to the original CAPTION_TYPE_MAP keys
    args.caption_type = CAPTION_TYPE_CHOICES.get(args.caption_type, "Descriptive")

    return args

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    clip_model, image_adapter, tokenizer, text_model = load_models(device)

    # Define the output directory
    output_dir = Path(args.output)
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory at: {output_dir.resolve()}")
        except Exception as e:
            print(f"Error creating output directory '{args.output}': {e}")
            return
    else:
        if not output_dir.is_dir():
            print(f"Error: The provided output path '{args.output}' is not a directory.")
            return

    # Determine if input is a single image or a directory
    if args.input_dir:
        input_path = Path(args.input_dir)
        if not input_path.is_dir():
            print(f"Error: The provided input_dir '{args.input_dir}' is not a directory.")
            return

        # Gather all supported images in the directory
        images = [img for img in input_path.iterdir() if img.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            print(f"No supported image formats found in directory '{args.input_dir}'.")
            return

        for image_path in images:
            print(f"\nProcessing image: {image_path.name}")
            prompt, caption = stream_chat(
                clip_model=clip_model,
                image_adapter=image_adapter,
                tokenizer=tokenizer,
                text_model=text_model,
                input_image_path=str(image_path),
                caption_type=args.caption_type,
                caption_length=args.caption_length,
                extra_options=args.extra_options,
                name_input=args.name_input,
                custom_prompt=args.custom_prompt,
                device=device
            )
            print("\n=== Caption Generation Result ===")
            print(f"Prompt Used:\n{prompt}\n")
            print(f"Generated Caption:\n{caption}\n")

            # Save only the caption to the output directory
            output_file = output_dir / f"{image_path.stem}.txt"
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(caption)
                print(f"Saved caption to: {output_file}")
            except Exception as e:
                print(f"Error saving caption for '{image_path.name}': {e}")
    else:
        if not args.input_image:
            print("Error: No input image provided.")
            return

        image_path = Path(args.input_image)
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            print(f"Error: The provided input_image '{args.input_image}' is not a supported image file.")
            return

        prompt, caption = stream_chat(
            clip_model=clip_model,
            image_adapter=image_adapter,
            tokenizer=tokenizer,
            text_model=text_model,
            input_image_path=args.input_image,
            caption_type=args.caption_type,
            caption_length=args.caption_length,
            extra_options=args.extra_options,
            name_input=args.name_input,
            custom_prompt=args.custom_prompt,
            device=device
        )
        print("\n=== Caption Generation Result ===")
        print(f"Prompt Used:\n{prompt}\n")
        print(f"Generated Caption:\n{caption}")

        # Save only the caption to the output directory
        output_file = output_dir / f"{image_path.stem}.txt"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(caption)
            print(f"Saved caption to: {output_file}")
        except Exception as e:
            print(f"Error saving caption for '{image_path.name}': {e}")

if __name__ == "__main__":
    main()