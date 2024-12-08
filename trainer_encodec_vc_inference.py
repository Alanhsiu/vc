import random
import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset, load_from_disk
from encodec import EncodecModel
from argparse import ArgumentParser, Namespace
# from encodec_model.nar_bart_model import NARBartForConditionalGeneration
from .encodec_model.nar_bart_model import NARBartForConditionalGeneration
from transformers import (AutoTokenizer, BartForConditionalGeneration,
                          BatchEncoding)
import json
import time
''' Inference code. '''


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

''' Non-auto-regressive model decoding function. '''
def nar_decode(model, tokenizer, inputs, batch_code, layer=0):
    base_input = inputs
    base_input["decoder_input_ids"] = batch_code
    # this is the line where error occurs
    decode_nar = model.forward(**base_input).logits
    id_range_start, id_range_end = tokenizer.convert_tokens_to_ids(
        f"v_tok_{0 + 1024 * layer}"), tokenizer.convert_tokens_to_ids(f"v_tok_{1024 + 1024 * layer}")
    
    # Create a tensor where values are equal to their own indices
    indices = torch.arange(decode_nar.size(-1)).to(decode_nar.device)
    
    # Create a mask for the range
    mask = (indices >= id_range_start) & (indices < id_range_end)
    
    # Set values out of range to very low value
    decode_nar_masked = torch.where(mask, decode_nar, torch.tensor(float("-inf")).to(decode_nar.device))
    
    # Get the argmax within the range
    return torch.argmax(decode_nar_masked, dim=-1)

''' Create attention masks. '''''
def get_attention_mask(seq_length, max_length):
    return [1] * seq_length + [0] * (max_length - seq_length)

''' Pack inputs for model. '''
def pack_inputs(tokenizer, instruction_ids, encodec_ids):
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id
    # pad_token_id = tokenizer.pad_token_id

    input_ids = []
    # attention_mask = []
    
    encoder_input_ids = [bos_token_id] + instruction_ids + [sep_token_id] + encodec_ids + [eos_token_id]
    # print("size: ", len(encoder_input_ids), "-- encoder_input_ids: ", encoder_input_ids)
    
    input_ids.append(encoder_input_ids)
    
    inputs = BatchEncoding(tensor_type="pt")
    inputs["input_ids"] = torch.tensor(input_ids)
    
    return inputs

def pack_inputs_v2(tokenizer, single_src_encodec, single_instruction):
    instruction_ids = tokenizer(single_instruction)["input_ids"][1 : -1]
    src_encodec_ids = tokenizer.convert_tokens_to_ids(
        [f"v_tok_{u}" for u in single_src_encodec[0]])

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id
    
    encoder_input_ids = [bos_token_id] + instruction_ids + [sep_token_id] + src_encodec_ids + [eos_token_id]
    # print("size: ", len(encoder_input_ids), "++ encoder_input_ids: ", encoder_input_ids)
    
    return encoder_input_ids

def pack_inputs_v3(packed_input):
    input_ids = []
    input_ids.append(packed_input)
    inputs = BatchEncoding(tensor_type="pt")
    inputs["input_ids"] = torch.tensor(input_ids)
    return inputs

''' Inference functions for ground truth. '''
def ground_truth_only(tokenizer, dataset, device):
    layer_list = []
    print("Instruction: ", dataset["instruction"][0])
    for layer_i in range(8):
        encode_input = tokenizer(
            "".join([f"v_tok_{u + layer_i * 1024}" for u in dataset[f"tgt_encodec_{layer_i}"][0]]),
            return_tensors="pt", add_special_tokens=False)
        encode_input = encode_input["input_ids"].to(device)
        layer_list.append(encode_input)

    return layer_list

''' Cascade AR and NAR model decoding function. '''
def cascade_ar_nar(ar_model, nar_model, ar_tokenizer, nar_tokenizer, dataset, device):
    layer_list = []

    instruction_ids = ar_tokenizer(dataset["instruction"][0])["input_ids"][1 : -1]
    
    # Get AR prediction../previous_ckpt/vc_nar/checkpoint-70000/
    src_encodec_ids = ar_tokenizer.convert_tokens_to_ids(
        [f"v_tok_{u}" for u in dataset[f"src_encodec_0"][0]])
    inputs = pack_inputs(ar_tokenizer, instruction_ids, src_encodec_ids)
    inputs = inputs.to(device)
    bad_words_ids = [[ar_tokenizer.convert_tokens_to_ids(f"v_tok_{i}")] for i in range(1024, 1024*8)]
    decode_ar = ar_model.generate(**inputs, max_length=1024, num_beams=1,
                                  do_sample=True, use_cache=True, bad_words_ids=bad_words_ids)
    layer_list.append(decode_ar[:, 2:-1])
    
    # Iterative predict NAR code
    # Encoder input: instruction + transcription + curr_src_encodec_inputs
    for layer in range(1, 8):
        curr_src_encodec_ids = nar_tokenizer.convert_tokens_to_ids(
        [f"v_tok_{u + layer * 1024}" for u in dataset[f"src_encodec_{layer}"][0]])
        inputs = pack_inputs(nar_tokenizer, instruction_ids, curr_src_encodec_ids)
        inputs = inputs.to(device)
        layer_list.append(nar_decode(nar_model, nar_tokenizer, inputs, layer_list[-1], layer))

    return layer_list
    
def cascade_ar_nar_v2(ar_model, nar_model, ar_tokenizer, nar_tokenizer, device, single_src_encodec, single_instruction, temperature = 1.0):
    layer_list = []
    
    instruction_ids = ar_tokenizer(single_instruction)["input_ids"][1 : -1]
    src_encodec_ids = ar_tokenizer.convert_tokens_to_ids([f"v_tok_{u}" for u in single_src_encodec[0]])

    inputs = pack_inputs(ar_tokenizer, instruction_ids, src_encodec_ids)
    inputs = inputs.to(device)
    
    # Debugging
    input_ids_tensor = inputs['input_ids']
    tensor_list = input_ids_tensor.flatten().tolist()

    for number in tensor_list:
        if number >= 59480:
            print(f"Error: Found invalid number {number} in tensor list which is >= 59480")
            raise ValueError(f"Invalid number {number} found in tensor list; numbers must be < 59480")

    bad_words_ids = [[ar_tokenizer.convert_tokens_to_ids(f"v_tok_{i}")] for i in range(1024, 1024*8)]
    # append 0 to 50264 to bad words ids
    vocab_ids = [[i] for i in range(0, 50265)]
    bad_words_ids.extend(vocab_ids)
    # 50265 to 59480 is the range of good words
    decode_ar = ar_model.generate(**inputs, max_length=1024, num_beams=1,
                                    do_sample=True, use_cache=True, bad_words_ids=bad_words_ids, temperature = temperature)
    layer_list.append(decode_ar[:, 2:-1])
    predicted_ids = layer_list[-1].flatten().tolist()
    # Iterative predict NAR code
    # Encoder input: instruction + curr_src_encodec_inputs
    for layer in range(1, 8):
        curr_src_encodec_ids = nar_tokenizer.convert_tokens_to_ids(
        [f"v_tok_{u + layer * 1024}" for u in single_src_encodec[layer]])
        inputs = pack_inputs(nar_tokenizer, instruction_ids, curr_src_encodec_ids)
        inputs = inputs.to(device)
        layer_list.append(nar_decode(nar_model, nar_tokenizer, inputs, layer_list[-1], layer))
    return layer_list, decode_ar

def cascade_ar_nar_v3(predicted_ids, nar_model, ar_tokenizer, nar_tokenizer, device, single_src_encodec, single_instruction):
    layer_list = []

    # Tokenize the instruction, remove special tokens
    instruction_ids = ar_tokenizer(single_instruction)["input_ids"][1:-1]

    src_encodec_ids = ar_tokenizer.convert_tokens_to_ids(
        [f"v_tok_{u}" for u in single_src_encodec[0]])

    # Package inputs from AR tokenizer and move to the specified device
    # inputs = pack_inputs(ar_tokenizer, instruction_ids, src_encodec_ids)
    # inputs = inputs.to(device)

    # Create a tensor from the predicted IDs and ensure it's the correct dtype
    prediction_tensor = torch.tensor(predicted_ids, dtype=torch.int64)
    if prediction_tensor.numel() == 0:  # Ensure prediction_tensor is not empty
        print("Warning: prediction_tensor is empty after conversion. Using default or placeholder values.")
        prediction_tensor = torch.tensor([[0]], dtype=torch.int64)  # Use a default value or handle accordingly
    prediction_tensor = prediction_tensor.view(1, -1)  # Reshape tensor to 1 x N for model compatibility

    # Check if CUDA is available for device
    if torch.cuda.is_available():
        prediction_tensor = prediction_tensor.to(device)
    else:
        print("CUDA is not available. Tensor will remain on CPU.")

    # Append initial tensor to the layer list
    layer_list.append(prediction_tensor)
    # print("Layer 0: ", layer_list[0])
    # Process each layer in the NAR model
    for layer in range(1, 8):
        curr_src_encodec_ids = nar_tokenizer.convert_tokens_to_ids(
            [f"v_tok_{u + layer * 1024}" for u in single_src_encodec[layer]])
        inputs = pack_inputs(nar_tokenizer, instruction_ids, curr_src_encodec_ids)
        inputs = inputs.to(device)
        layer_list.append(nar_decode(nar_model, nar_tokenizer, inputs, layer_list[-1], layer))
    return layer_list

def cascade_ar_nar_v4(ar_model, ar_tokenizer, device, single_src_encodec, single_instruction):
    layer_list = []
    
    instruction_ids = ar_tokenizer(single_instruction)["input_ids"][1 : -1]
    src_encodec_ids = ar_tokenizer.convert_tokens_to_ids(
        [f"v_tok_{u}" for u in single_src_encodec[0]])
    
    inputs = pack_inputs(ar_tokenizer, instruction_ids, src_encodec_ids)
    inputs = inputs.to(device)
    
    # Debugging
    input_ids_tensor = inputs['input_ids']
    tensor_list = input_ids_tensor.flatten().tolist()
    

    for number in tensor_list:
        if number >= 59480:
            print(f"Error: Found invalid number {number} in tensor list which is >= 59480")
            raise ValueError(f"Invalid number {number} found in tensor list; numbers must be < 59480")

    bad_words_ids = [[ar_tokenizer.convert_tokens_to_ids(f"v_tok_{i}")] for i in range(1024, 1024*8)]
    # append 0 to 50264 to bad words ids
    vocab_ids = [[i] for i in range(0, 50265)]
    bad_words_ids.extend(vocab_ids)
    # 50265 to 59480 is the range of good words
    decode_ar = ar_model.generate(**inputs, max_length=1024, num_beams=1,
                                    do_sample=True, use_cache=True, bad_words_ids=bad_words_ids)
    return decode_ar

def pack_inputs_batch(tokenizer, instructions, encodecs):
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    input_ids = []
    
    for instruction_ids, encodec_ids in zip(instructions, encodecs):
        encoder_input_ids = [bos_token_id] + instruction_ids + [sep_token_id] + encodec_ids + [eos_token_id]
        input_ids.append(encoder_input_ids)
    max_len = max(len(seq) for seq in input_ids)
    padded_input_ids = [seq + [pad_token_id] * (max_len - len(seq)) for seq in input_ids]

    encoder_input_ids = torch.tensor(padded_input_ids)
    
    inputs = BatchEncoding(tensor_type="pt")

    try:
        inputs["input_ids"] = encoder_input_ids
    except ValueError as e:
        print(f"Error while creating tensor: {e}")
        print(f"Input IDs: {encoder_input_ids}")
        raise e
    return inputs

def cascade_ar_nar_batch(ar_model, nar_model, ar_tokenizer, nar_tokenizer, device, batch_src_encodec, batch_instruction, temperature = 1.0):
    instruction_ids_batch = []
    src_encodec_ids_batch = []
    batch_layer_list = []   

    for single_instruction, single_src_encodec in zip(batch_instruction, batch_src_encodec):
        instruction_ids = ar_tokenizer(single_instruction)["input_ids"][1 : -1]
        src_encodec_ids = ar_tokenizer.convert_tokens_to_ids([f"v_tok_{u}" for u in single_src_encodec[0]])
        instruction_ids_batch.append(instruction_ids)
        src_encodec_ids_batch.append(src_encodec_ids)

    inputs = pack_inputs_batch(ar_tokenizer, instruction_ids_batch, src_encodec_ids_batch)

    # inputs_batch = ar_tokenizer(inputs_batch, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    
    # Debugging
    input_ids_tensor = inputs['input_ids']
    tensor_list = input_ids_tensor.flatten().tolist()
    
    for number in tensor_list:
        if number >= 59480:
            print(f"Error: Found invalid number {number} in tensor list which is >= 59480")
            raise ValueError(f"Invalid number {number} found in tensor list; numbers must be < 59480")
            
    bad_words_ids = [[ar_tokenizer.convert_tokens_to_ids(f"v_tok_{i}")] for i in range(1024, 1024*8)]
    vocab_ids = [[i] for i in range(0, 50265)]
    bad_words_ids.extend(vocab_ids)

    decode_ar_batch = ar_model.generate(**inputs, max_length=1024, num_beams=5, do_sample=True, use_cache=True, bad_words_ids=bad_words_ids, temperature = temperature)
    # decode_ar_batch --> tensor([[],[],[],[],[],[],[],[]], device = 'cuda:0')
    filtered_sequences = []
    for seq in decode_ar_batch:
        # Filter out padding, EOS, and BOS tokens
        filtered_seq = seq[(seq != 1) & (seq != ar_tokenizer.eos_token_id) & (seq != ar_tokenizer.bos_token_id)]
        if len(filtered_seq.shape) == 1:
            filtered_seq = filtered_seq.unsqueeze(0)  # Add a batch dimension to keep the same shape
        filtered_sequences.append(filtered_seq)

    for idx, filtered_seq in enumerate(filtered_sequences):
        layer_list = []
        layer_list.append(filtered_seq)
        for layer in range(1, 8):
            curr_src_encodec_ids = nar_tokenizer.convert_tokens_to_ids(
            [f"v_tok_{u + layer * 1024}" for u in batch_src_encodec[idx][layer]])
            inputs = pack_inputs(nar_tokenizer, instruction_ids_batch[idx], curr_src_encodec_ids)
            inputs = inputs.to(device)
            
            # problem
            layer_list.append(nar_decode(nar_model, nar_tokenizer, inputs, layer_list[-1], layer))
        batch_layer_list.append(layer_list) 
        
    return batch_layer_list, filtered_sequences

def get_ar_prediction(args, ar_model, nar_model, ar_tokenizer, nar_tokenizer, single_src_encodec, single_instruction, episode_counter=0):
    device = args.device
    # output_path = args.output_path
    # Modify the output_path to indicate this is a milestone audio file
    # output_path_ckpt = output_path.replace(".wav", f"_save_{episode_counter}.wav")
    ar_model.to(device)
    nar_model.to(device)
    layer_list, decode_ar = cascade_ar_nar_v2(ar_model, nar_model, ar_tokenizer, nar_tokenizer, device, single_src_encodec, single_instruction)
    
    encodec_code = convert_to_encode_code(nar_tokenizer, layer_list) 
    # print("encodec_code[0](get_ar_prediction): ", encodec_code[0])   
    # audio = synthesize_audio(encodec_code, device)
    # sf.write(output_path_ckpt, np.ravel(audio), samplerate=24000)
    # sf.write(output_path, np.ravel(audio), samplerate=24000)
    # print("Episode", episode_counter, ": audio saved to ", output_path_ckpt)
    
    return encodec_code[0], decode_ar

def get_ar_prediction_for_data(args, ar_model, ar_tokenizer, single_src_encodec, single_instruction):
    # set_seed(args.seed)
    device = args.device
    ar_model.to(device)
    decode_ar = cascade_ar_nar_v4(ar_model, ar_tokenizer, device, single_src_encodec, single_instruction)
    return decode_ar

def get_ar_prediction_sampling_rate(args, ar_model, nar_model, ar_tokenizer, nar_tokenizer, single_src_encodec, single_instruction, episode_counter=0, temperature = 1.0):
    # set_seed(args.seed)
    device = args.device
    output_path = args.output_path
    # Modify the output_path to indicate this is a milestone audio file
    output_path_ckpt = output_path.replace(".wav", f"_save_{episode_counter}.wav")
    ar_model.to(device)
    nar_model.to(device)
    layer_list, decode_ar = cascade_ar_nar_v2(ar_model, nar_model, ar_tokenizer, nar_tokenizer, device, single_src_encodec, single_instruction, temperature = temperature)
    
    encodec_code = convert_to_encode_code(nar_tokenizer, layer_list) 
    # print("encodec_code[0](get_ar_prediction): ", encodec_code[0])   
    audio = synthesize_audio(encodec_code, device)
    sf.write(output_path_ckpt, np.ravel(audio), samplerate=16000)
    sf.write(output_path, np.ravel(audio), samplerate=16000)
    print("Episode", episode_counter, ": audio saved to ", output_path_ckpt)
    return encodec_code[0], decode_ar, output_path_ckpt

def get_ar_prediction_v3(args, ar_model, nar_model, ar_tokenizer, nar_tokenizer, single_src_encodec, single_instruction, episode_counter=0, temperature = 1.0):
    device = args.device
    output_path = args.output_path
    # Modify the output_path to indicate this is a milestone audio file
    output_path_ckpt = output_path.replace(".wav", f"_save_{episode_counter}.wav")
    ar_model.to(device)
    nar_model.to(device)
    layer_list, decode_ar = cascade_ar_nar_v2(ar_model, nar_model, ar_tokenizer, nar_tokenizer, device, single_src_encodec, single_instruction, temperature = temperature)
    
    encodec_code = convert_to_encode_code(nar_tokenizer, layer_list) 
    audio = synthesize_audio(encodec_code, device)
    sf.write(output_path_ckpt, np.ravel(audio), samplerate=24000)
    print("Episode", episode_counter, ": audio saved to ", output_path_ckpt)
    return encodec_code[0], decode_ar, output_path_ckpt

def get_ar_prediction_get_audio(args, ar_model, nar_model, ar_tokenizer, nar_tokenizer, single_src_encodec, single_instruction, episode_counter=0, temperature = 1.0):
    # set_seed(args.seed)
    device = args.device
    # output_path = args.output_path
    # output_path_ckpt = output_path.replace(".wav", f"_save_{episode_counter}.wav")
    ar_model.to(device)
    nar_model.to(device)
    layer_list, decode_ar = cascade_ar_nar_v2(ar_model, nar_model, ar_tokenizer, nar_tokenizer, device, single_src_encodec, single_instruction, temperature = temperature)
    encodec_code = convert_to_encode_code(nar_tokenizer, layer_list) 
    audio = synthesize_audio(encodec_code, device)
    # sf.write(output_path_ckpt, np.ravel(audio), samplerate=24000)
    # sf.write(output_path, np.ravel(audio), samplerate=24000)
    return decode_ar, audio


def convert_to_encode_code_batch(tokenizer, layer_list_batch):
    batch_encodec_code = []
    
    for i in range(len(layer_list_batch[0])):  # Iterate over the batch size
        encodec_code = []
        for layer, layer_ids in enumerate(layer_list_batch):
            # Decode the current layer's IDs for the i-th item in the batch
            decoded_ids = tokenizer.batch_decode(layer_ids[i].unsqueeze(0))
            decoded_ids = decoded_ids[0].replace("</s>", "").replace("<pad>", "")
            encodec_code.append([int(j) - layer * 1024 for j in decoded_ids.split("v_tok_") if len(j) > 0])
        batch_encodec_code.append(encodec_code)
    
    return batch_encodec_code

def get_ar_prediction_batch(args, ar_model, nar_model, ar_tokenizer, nar_tokenizer, batch_src_encodec, batch_instruction, episode_counter=0, temperature = 1.0):
    device = args.device
    output_path = args.output_path
    ar_model.to(device)
    nar_model.to(device)
    layer_list_batch, decode_ar_batch = cascade_ar_nar_batch(ar_model, nar_model, ar_tokenizer, nar_tokenizer, device, batch_src_encodec, batch_instruction, temperature = temperature)
    output_path_ckpt_list = []
    for idx in range(len(layer_list_batch)):
        encodec_code = convert_to_encode_code(nar_tokenizer, layer_list_batch[idx])
        audio = synthesize_audio(encodec_code, device)
        output_path_ckpt = output_path.replace(".wav", f"_save_{episode_counter+idx}_item_{idx}.wav")
        sf.write(output_path_ckpt, np.ravel(audio), samplerate=24000)
        output_path_ckpt_list.append(output_path_ckpt)
        print(f"Episode {episode_counter}, item {idx}: audio saved to {output_path_ckpt}")
    
    return batch_src_encodec, batch_instruction, decode_ar_batch, output_path_ckpt_list

def get_ar_prediction_audio_batch(args, ar_model, nar_model, ar_tokenizer, nar_tokenizer, batch_src_encodec, batch_instruction, episode_counter=0, temperature = 1.0):
    device = args.device

    ar_model.to(device)
    nar_model.to(device)
    
    start_time = time.time()
    layer_list_batch, decode_ar_batch = cascade_ar_nar_batch(ar_model, nar_model, ar_tokenizer, nar_tokenizer, device, batch_src_encodec, batch_instruction, temperature = temperature)
    # print(f"Time taken for batch inference: {time.time() - start_time} seconds")
    
    start_time = time.time()
    audio_list = []
    # print("Length of layer_list_batch: ", len(layer_list_batch))
    for idx in range(len(layer_list_batch)):
        encodec_code = convert_to_encode_code(nar_tokenizer, layer_list_batch[idx])
        audio = synthesize_audio(encodec_code, device)
        audio_list.append(audio)
    # print(f"Time taken for batch audio synthesis: {time.time() - start_time} seconds")
    return audio_list, decode_ar_batch


def get_ar_prediction_v2(args, predicted_ids, nar_model, ar_tokenizer, nar_tokenizer, single_src_encodec, single_instruction, episode_counter):
    # set_seed(args.seed)
    device = args.device
    output_path = args.output_path
    # Modify the output_path to indicate this is a milestone audio file
    output_path_ckpt = output_path.replace(".wav", f"_save_{episode_counter}.wav")
    nar_model.to(device)
    layer_list = cascade_ar_nar_v3(predicted_ids, nar_model, ar_tokenizer, nar_tokenizer, device, single_src_encodec, single_instruction)
    encodec_code = convert_to_encode_code(nar_tokenizer, layer_list) 
    # append predicted_ids at the beginning of the encodec code
    # encodec_code.insert(0, predicted_ids)  
    # print dimensions of encodec code
    # print("encodec_code dimensions: ", len(encodec_code))
    try:
        audio = synthesize_audio(encodec_code, device)
        sf.write(output_path_ckpt, np.ravel(audio), samplerate=24000)
        sf.write(output_path, np.ravel(audio), samplerate=24000)
        print("Episode", episode_counter, ": audio saved to ", output_path_ckpt)
    except Exception as e:
        print(f"Error get_ar_prediction_v2: {e}")
        # print("encodec_code[0](get_ar_prediction_v2): ", encodec_code[0])
        return None
        
    # print("Episode", episode_counter, ": audio saved to", output_path_ckpt)
    # print("encodec_code[0](get_ar_prediction_v2): ", encodec_code[0])
    
    return encodec_code[0], output_path_ckpt

''' Inference functions for NAR model only, without AR model. '''
def nar_model_only(model, tokenizer, dataset, device):
    layer_list = []

    instruction_ids = tokenizer(dataset["instruction"][0])["input_ids"][1 : -1]
    transcription_ids = tokenizer(dataset["transcription"][0])["input_ids"][1 : -1]

    # Use ground truth AR prediction
    tgt_encodec_input = tokenizer(
        "".join([f"v_tok_{u}" for u in dataset[f"tgt_encodec_0"][0]]),
        return_tensors="pt", add_special_tokens=False)
    tgt_encodec_input_ids = tgt_encodec_input["input_ids"].to(device)
    layer_list.append(tgt_encodec_input_ids)

    # Iterative predict NAR code
    # Encoder input: instruction + transcription + curr_src_encodec_inputs
    for layer in range(1, 8):
        curr_src_encodec_ids = tokenizer.convert_tokens_to_ids(
        [f"v_tok_{u + layer * 1024}" for u in dataset[f"src_encodec_{layer}"][0]])
        inputs = pack_inputs(tokenizer, instruction_ids, transcription_ids, curr_src_encodec_ids, 1023)
        inputs = inputs.to(device)
        layer_list.append(nar_decode(model, tokenizer, inputs, layer_list[-1], layer))

    return layer_list

''' Convert prediction results to encodec code. '''
def convert_to_encode_code(tokenizer, layer_list):
    encodec_code = []
    for layer, layer_ids in enumerate(tokenizer.batch_decode(torch.cat(layer_list))):
        layer_ids = layer_ids.replace("</s>", "")
        encodec_code.append([int(i) - layer * 1024 for i in layer_ids.split("v_tok_") if len(i) > 0])
    return encodec_code

''' Synthesize audio from encodec code. '''        
# def synthesize_audio(encodec_code, device):
#     try:
#         model = EncodecModel.encodec_model_24khz()
#         model.set_target_bandwidth(6.0)
#         model.to(device)
        
#         encodec_input = torch.tensor(encodec_code).unsqueeze(0)
#         encodec_input = encodec_input.to(device)
#         audio = model.decode([(encodec_input, None)]).cpu().detach().numpy()[0]
        
#         # change the type of audio to float32
#         audio = audio.astype(np.float32)
#         return audio
#     except Exception as e:
#         print(f"Error synthesize_audio: {e}")
#         print("encodec_code: ", encodec_code)
#         print("model: ", model)
#         # print("encodec_input: ", encodec_input)
#         # print("encodec_input.shape: ", encodec_input.shape)
#         return None

def synthesize_audio(encodec_code, device):
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        model.to(device)
        
        encodec_array = np.array(encodec_code)
        if np.isnan(encodec_array).any() or np.isinf(encodec_array).any():
            print("Error: encodec_code contains NaN or Inf values")
            return None
        
        encodec_input = torch.tensor(encodec_code).unsqueeze(0).to(device)
        audio = model.decode([(encodec_input, None)]).cpu().detach().numpy()[0]
        audio = audio.astype(np.float32)
        return audio
    
    except Exception as e:
        print(f"Error synthesize_audio: {e}")
        print("encodec_code: ", encodec_code)
        return None

    
def run(args, input_dir, output_path):
    set_seed(args.seed)
    device = args.device
    
    dataset = load_from_disk(input_dir)
    print("Dataset loaded, size is ", len(dataset))
    
    ''' Synthesize speech from ground truth encodec code. '''
    if args.ground_truth_only:
        tokenizer = AutoTokenizer.from_pretrained(args.ground_truth_model_name)
        
        layer_list = ground_truth_only(tokenizer, dataset, device)        
        encodec_code = convert_to_encode_code(tokenizer, layer_list)    
        audio = synthesize_audio(encodec_code, device)
        sf.write(args.ground_truth_output_path, np.ravel(audio), samplerate=24000)
        print("Mode: ground truth only")

    ''' Synthesize speech from cascade AR and NAR encodec code. '''
    if args.cascade_ar_nar:
        ar_tokenizer = AutoTokenizer.from_pretrained(args.ar_checkpoint)
        ar_model = BartForConditionalGeneration.from_pretrained(args.ar_checkpoint)
        ar_model.to(device)

        nar_tokenizer = AutoTokenizer.from_pretrained(args.nar_checkpoint)
        nar_model = NARBartForConditionalGeneration.from_pretrained(args.nar_checkpoint)
        nar_model.to(device)
        
        layer_list = cascade_ar_nar(ar_model, nar_model, ar_tokenizer, nar_tokenizer, dataset, device)
        encodec_code = convert_to_encode_code(nar_tokenizer, layer_list)    
        audio = synthesize_audio(encodec_code, device)
        sf.write(output_path, np.ravel(audio), samplerate=24000)
        print("Mode: cascade AR and NAR")
        return encodec_code
    
    ''' Synthesize speech from NAR encodec code (+ ground truth encodec code). '''
    if args.nar_model_only:
        nar_tokenizer = AutoTokenizer.from_pretrained(args.nar_checkpoint)
        nar_model = NARBartForConditionalGeneration.from_pretrained(args.nar_checkpoint)
        nar_model.to(device)

        layer_list = nar_model_only(nar_model, nar_tokenizer, dataset, device)
        encodec_code = convert_to_encode_code(nar_tokenizer, layer_list)    
        audio = synthesize_audio(encodec_code, device)
        sf.write(args.nar_output_path, np.ravel(audio), samplerate=24000)
        print("Mode: NAR only")
             
def parse_args() -> Namespace:
    parser = ArgumentParser()
    # parser.add_argument("-d", "--dataset", type=str, default="/home/b0990106x/TextRL/data-encodec/")
    parser.add_argument("-s", "--splits", type=str, nargs="+", default=["train"])
    
    parser.add_argument("--ground_truth_only", action="store_true")
    parser.add_argument("--cascade_ar_nar", action="store_true")
    parser.add_argument("--nar_model_only", action="store_true")
    
    parser.add_argument("--ground_truth_model_name", type=str, default="voidful/bart-base-unit")
    parser.add_argument("--ar_checkpoint", type=str, default="lca0503/speech-chatgpt-base-ar-v2-epoch10-wotrans")
    parser.add_argument("--nar_checkpoint", type=str, default="lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans")

    # parser.add_argument("--ground_truth_output_path", type=str, default="output_wav/vc/ground_truth/train_1.wav")
    # parser.add_argument("--cascade_output_path", type=str, default="output_wav/vc/ar_nar_cascade/train_1.wav")
    # parser.add_argument("--nar_output_path", type=str, default="output_wav/vc/nar/train_1.wav")
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=torch.device, default="cuda")
    
    args = parser.parse_args()    
    return args


# if __name__ == "__main__":
#     args = parse_args()
#     run(args)
