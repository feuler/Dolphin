import argparse
from PIL import Image
import torch
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import AutoProcessor

def run_inference(onnx_model_path: str, image_path: str, prompt: str):
    """
    Runs inference using the exported Dolphin ONNX model.

    Args:
        onnx_model_path (str): Path to the directory containing the ONNX model files.
        image_path (str): Path to the input image.
        prompt (str): The prompt for the model.
    """
    print(f"Loading ONNX model from: {onnx_model_path}")
    # Use the appropriate device (CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ORTModelForVision2Seq loads all necessary ONNX files and sets up the generation pipeline.
    model = ORTModelForVision2Seq.from_pretrained(onnx_model_path, provider="CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider")
    
    # The processor contains the tokenizer and image processor
    processor = AutoProcessor.from_pretrained(onnx_model_path)
    tokenizer = processor.tokenizer

    print(f"Loading image from: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    print("Preparing inputs for the model...")
    # 1. Process the image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 2. Process the prompt
    task_prompt = f"<s>{prompt} <Answer/>"
    decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    print("Running generation with ONNX model...")
    # The generate method works just like in transformers
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=4096,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    print("Decoding generated sequence...")
    sequence = tokenizer.batch_decode(outputs.sequences)[0]
    
    # Clean up the output string
    result = sequence.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").replace(task_prompt, "").strip()

    print("\n" + "="*20 + " INFERENCE RESULT " + "="*20)
    print(f"Prompt: {prompt}")
    print("-" * 58)
    print(f"Result:\n{result}")
    print("=" * 58)


def main():
    parser = argparse.ArgumentParser(description="Run inference with Dolphin ONNX model.")
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        default="./dolphin_onnx_fp16", # Assumes you ran the fp16 export
        help="Path to the directory with the exported ONNX model.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./bytedance-dolphin/demo/element_imgs/table_1.jpeg",
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Parse the table in the image.",
        help="Prompt to guide the model's generation.",
    )
    args = parser.parse_args()

    run_inference(args.onnx_model_path, args.image_path, args.prompt)

if __name__ == "__main__":
    main()