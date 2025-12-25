import argparse
import os
import sys
from typing import List

import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from trl import AutoModelForCausalLMWithValueHead

# Add the src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from llamafactory.extras import logging
from llamafactory.hparams import ModelArguments
from llamafactory.model import patch_valuehead_model
from llamafactory.model.model_utils.valuehead import load_valuehead_params

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
logger = logging.get_logger(__name__)


class RewardModelInference:
    """
    Reward model inference class for models trained with AutoModelForCausalLMWithValueHead
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = True,
    ):
        """
        Initialize the reward model for inference
        
        Args:
            model_name_or_path: Path to the base model
            trust_remote_code: Whether to trust remote code
        """
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.add_regression_token = True
        max_pixels = 300000

        # Create model arguments
        model_args = ModelArguments(
            model_name_or_path=self.model_name_or_path,
            adapter_name_or_path=None,
            trust_remote_code=self.trust_remote_code,
        )
 
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)
        
        vhead_path = model_args.model_name_or_path
        vhead_params = load_valuehead_params(vhead_path, model_args)
        assert vhead_params is not None
        model.load_state_dict(vhead_params, strict=False)
        print(f"Loaded valuehead from checkpoint: {vhead_path}")
        
        # default processer
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            max_pixels=max_pixels
        )
        
        self.model = model.half()
        self.model.eval()

    def get_reward_scores(self, messages_list: List[dict]) -> List[float]:
        """
        Get reward scores for a list of message conversations
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if self.add_regression_token:
            regression_token_id = self.processor.tokenizer.convert_tokens_to_ids('<|regression|>')
            assert regression_token_id is not None, "Special token ID not found"
            
            batch_size = inputs['input_ids'].size(0)
            seq_len = inputs['input_ids'].size(1)
            
            # Find the actual end position for each sequence (last 1 in attention_mask)
            seq_lengths = inputs['attention_mask'].sum(dim=1)  # [batch_size]
            
            # Create new tensors with one extra position
            new_input_ids = torch.zeros(
                (batch_size, seq_len + 1),
                dtype=inputs['input_ids'].dtype,
                device=inputs['input_ids'].device
            )
            new_attention_mask = torch.zeros(
                (batch_size, seq_len + 1),
                dtype=inputs['attention_mask'].dtype,
                device=inputs['attention_mask'].device
            )
            
            # Fill the new tensors
            for i in range(batch_size):
                actual_len = seq_lengths[i].item()
                # Copy original sequence
                new_input_ids[i, :actual_len] = inputs['input_ids'][i, :actual_len]
                new_attention_mask[i, :actual_len] = inputs['attention_mask'][i, :actual_len]
                
                # Insert regression token at the actual end position
                new_input_ids[i, actual_len] = regression_token_id
                new_attention_mask[i, actual_len] = 1
                
                # Keep padding tokens as 0 for the rest
                if actual_len < seq_len:
                    new_input_ids[i, actual_len+1:] = inputs['input_ids'][i, actual_len:]
            
            inputs['input_ids'] = new_input_ids
            inputs['attention_mask'] = new_attention_mask
        inputs = inputs.to("cuda")
        
        # Get model predictions
        if inputs['attention_mask'].sum(dim=1).max().item() > 16000:
            first_image_path = [msg['image'] for msg in messages_list[0][1]['content'] if msg['type'] == 'image'][0]
            warning_str = f"Sequence length exceeds 16000, which may cause memory issues. len: {inputs['attention_mask'].sum(dim=1).max().item()}, first_image_path: {first_image_path}"
            logger.warning(warning_str)
            print(warning_str)
            return [0.0] * len(messages_list)
        
        with torch.no_grad():
            # Disable dropout and batch norm for deterministic behavior
            _, _, values = self.model(
                **inputs, 
                output_hidden_states=True, 
                return_dict=True, 
                use_cache=False
            )
            scores = values.gather(
                dim=-1, 
                index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1)
            ).squeeze(-1)
            
        return scores.cpu().tolist()
    

def main():
    parser = argparse.ArgumentParser(description="Reward Model Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory containing model checkpoints')
    args = parser.parse_args()

    # Initialize reward model
    total_model_path = os.path.join(args.ckpt_dir, args.model_path)
    reward_model = RewardModelInference(
        model_name_or_path=total_model_path,
        trust_remote_code=True,
    )
    
    # four documents to be evaluated
    input_data_lst = [
        ('4d149dcf-377d-4bd4-843c-281b8c48bd64_1', 3),
        ('4d149dcf-377d-4bd4-843c-281b8c48bd64_5', 1),
        ('4d149dcf-377d-4bd4-843c-281b8c48bd64_7', 3),
        ('062ad0f7-6d44-46f7-9fa3-3ce4361c5565_0', 2)
    ]
    messages_lst = []
    for item in input_data_lst:
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You need to create a professional document page(s). "
                }
            ]
        }]
        for page_idx in range(1, item[1]+1):
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join('./temp_debug', item[0], f'page_{page_idx:03d}.png')
                    }
                ]
            })
        messages_lst.append(messages)
    
    scores = reward_model.get_reward_scores(messages_lst)
    print(f"scores: {scores}")
    # DocReward-3B scores: [-0.406494140625, -2.5390625, -1.9267578125, -1.2568359375]


if __name__ == "__main__":
    main()
