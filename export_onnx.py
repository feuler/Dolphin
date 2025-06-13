import argparse
import os
import shutil
from pathlib import Path
import traceback # For more detailed error logging

import onnx # For loading/saving ONNX models
from onnxconverter_common import convert_float_to_float16 # For FP16 conversion

from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import AutoProcessor

def export_dolphin_to_onnx(model_id: str, output_dir: str, fp16: bool = False):
    """
    Exports the ByteDance/Dolphin model to ONNX format.
    Uses onnxconverter_common.convert_float_to_float16 for FP16 conversion.

    Args:
        model_id (str): The Hugging Face model identifier.
        output_dir (str): The final directory path where the ONNX model 
                          (either FP32 or FP16) will be saved.
                          Example: "./dolphin_onnx_fp16" or "./dolphin_onnx_fp32".
        fp16 (bool): Whether to convert the exported model to float16 precision.
    """
    fp32_temp_dir = Path(f"{output_dir}_fp32_temp")
    final_model_path = Path(output_dir)

    # --- Initial Cleanup ---
    if fp32_temp_dir.exists():
        print(f"Removing existing temporary directory: {fp32_temp_dir}")
        shutil.rmtree(fp32_temp_dir)
    if final_model_path.exists():
        print(f"Removing existing final output directory: {final_model_path}")
        shutil.rmtree(final_model_path)

    fp32_temp_dir.mkdir(parents=True, exist_ok=False)
    print(f"Created temporary directory for FP32 export: {fp32_temp_dir}")

    # --- Step 1: Export the model to ONNX in FP32 format ---
    print("\n[Step 1/2] Starting ONNX export to FP32...")
    try:
        model_fp32_exporter = ORTModelForVision2Seq.from_pretrained(model_id, export=True)
        processor = AutoProcessor.from_pretrained(model_id)

        print(f"Saving the exported FP32 ONNX model and configuration files to {fp32_temp_dir}...")
        model_fp32_exporter.save_pretrained(fp32_temp_dir) # Saves encoder, decoder, etc. ONNX files
        processor.save_pretrained(fp32_temp_dir) # Saves tokenizer and processor configs
        print("✅ FP32 ONNX export completed successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred during the initial ONNX export process.")
        print(f"Error details: {e}")
        traceback.print_exc()
        if fp32_temp_dir.exists():
            print(f"Cleaning up temporary directory due to error: {fp32_temp_dir}")
            shutil.rmtree(fp32_temp_dir)
        return

    # --- Step 2: (Optional) Convert the model to FP16 ---
    if fp16:
        print("\n[Step 2/2] Converting the model to FP16...")
        try:
            final_model_path.mkdir(parents=True, exist_ok=True) # Ensure final FP16 dir exists

            # Convert each ONNX file to FP16
            for onnx_file_path_fp32 in fp32_temp_dir.glob("*.onnx"):
                print(f"Converting {onnx_file_path_fp32.name} to FP16...")
                model_fp32_loaded = onnx.load(str(onnx_file_path_fp32))
                model_fp16 = convert_float_to_float16(model_fp32_loaded)
                
                output_onnx_file_path_fp16 = final_model_path / onnx_file_path_fp32.name
                onnx.save(model_fp16, str(output_onnx_file_path_fp16))
            print("✅ All ONNX model files converted to FP16.")

            # Copy non-ONNX files (configs, tokenizer files) from the temp FP32 dir to the final FP16 dir
            print(f"Copying non-ONNX configuration files from {fp32_temp_dir} to {final_model_path}...")
            for item_path in fp32_temp_dir.iterdir():
                if item_path.name.endswith(".onnx"):
                    continue # Skip ONNX files as they were converted and saved

                target_item_path = final_model_path / item_path.name
                if item_path.is_dir():
                    shutil.copytree(item_path, target_item_path, dirs_exist_ok=True)
                else: # It's a file
                    shutil.copy2(item_path, target_item_path) # copy2 preserves metadata
            print("✅ Non-ONNX files copied to FP16 directory.")

        except Exception as e:
            print(f"\n❌ An error occurred during the FP16 conversion or file copying.")
            print(f"Error details: {e}")
            traceback.print_exc()
            return # Exit after printing error
        finally:
            if fp32_temp_dir.exists():
                print(f"Cleaning up temporary FP32 directory: {fp32_temp_dir}")
                shutil.rmtree(fp32_temp_dir)
    else:
        print("\n[Step 2/2] Skipping FP16 conversion. Finalizing FP32 model.")
        print(f"Moving FP32 model from {fp32_temp_dir} to {final_model_path}...")
        shutil.move(str(fp32_temp_dir), str(final_model_path))
        print(f"✅ FP32 model finalized at {final_model_path}.")

    # --- Final Summary ---
    print("\n" + "="*50)
    print("✅ Full ONNX export process completed!")
    if final_model_path.exists():
        print(f"   Model saved to: {final_model_path.resolve()}")
        print(f"   Precision: {'FP16' if fp16 else 'FP32'}")
        print("The output directory contains:")
        for file_item in sorted(os.listdir(final_model_path)):
            print(f"   - {file_item}")
    else:
        print(f"⚠️ Warning: Final output directory {final_model_path.resolve()} does not exist or was not created.")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Export ByteDance/Dolphin model to ONNX.")
    parser.add_argument("--model_id", type=str, default="ByteDance/Dolphin", help="Hugging Face model ID.")
    parser.add_argument("--output_dir", type=str, default="./dolphin_onnx", 
                        help="Base name for the output directory. Suffix '_fp16' or '_fp32' will be added.")
    parser.add_argument("--fp16", action="store_true", help="Convert the model to float16 precision.")
    args = parser.parse_args()

    actual_output_dir = f"{args.output_dir}_{'fp16' if args.fp16 else 'fp32'}"

    export_dolphin_to_onnx(args.model_id, actual_output_dir, args.fp16)

if __name__ == "__main__":
    main()