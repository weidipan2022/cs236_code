python generate_lora.py --prompt "A scene of a sks futuristic cityscape" --model_path "./models/dreambooth-lora/miles" --output_folder "./outputs" --steps 50

python generate_controlnet_lora.py --prompt "A scene of a sks futuristic cityscape" --image_prompt "./data/old_game/8090_game.png" --model_path "./models/dreambooth-lora/miles" --output_folder "./outputs/8090" --steps 50