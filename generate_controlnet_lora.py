import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Images from Lora Weights")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a immersive scene of sks cyberpunk, race cars",
        help="prompt to generate the image",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/dreambooth-lora/cyperbunk",
        help=("the path to the trained model file"),
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="./outputs/lora",
        help=("the path to folder to hold generated images"),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help=("inference steps"),
    )

    parser.add_argument(
        "--image_prompt",
        type=str,
        default="./data/old_game/8090_game.png",
        help=("the img that goes with prompt"),
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()  # get arguments
    file_name = re.sub(
        r"\W+", "-", args.prompt
    )  # change all non-alphanumeric characters to dash

    if torch.cuda.is_available():
        device = "cuda"
        # if limited by GPU memory, chunking the attention computation in addition to using fp16
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
    else:
        device = "cpu"
        # if on CPU or want to have maximum precision on GPU, use default full-precision setting
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        )
    print(f"device is {device}")

    pipe.unet.load_attn_procs(args.model_path)
    pipe.to(device)

    init_image = Image.open(args.image_prompt)
    init_image = init_image.resize((768, 512))

    image = pipe(
        args.prompt, num_inference_steps=args.steps, strength=0.7, guidance_scale=7.5
    ).images[0]
    image.save(args.output_folder + "/" + file_name + ".png")


if __name__ == "__main__":
    main()
