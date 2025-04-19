import os
import wandb
import traceback
import copy
import torch

class Logger:
    def __init__(self, config, wandb_log=True):
        self.config = config
        self.wandb_log = wandb_log
        self.epoch_logs = {}
        self.epoch = 0
        
    def reset(self):
        self.epoch_logs = {}
        self.epoch = 0

    def log(self, log_dict, epoch_end_log=True):
        if 'epoch' in log_dict:
            self.epoch = log_dict['epoch']
            
        if self.wandb_log:
            try:
                log_dict_ = copy.deepcopy(log_dict)
                if 'epoch' not in log_dict:
                    log_dict_['epoch'] = self.epoch
                wandb.log(log_dict_)
            except Exception as e:
                print(f"Error logging to wandb: {e}")
                print("Logging to wandb failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Traceback:", traceback.format_exc())

        log_path = os.path.join(self.config.output_dir, wandb.run.id, "log.txt")
        with open(log_path, "a") as f:
            for key, value in log_dict.items():
                if key != "epoch":
                    f.write(f"{key}: {value}\n")
            f.write("\n")

        # Accumulate logs for averaging
        if epoch_end_log:
            for key, value in log_dict.items():
                if key not in ["epoch" , "lang"]:
                    if key not in self.epoch_logs:
                        self.epoch_logs[key] = []
                    self.epoch_logs[key].append(value)

    def log_epoch_average(self):
        avg_logs = {}
        for key, values in self.epoch_logs.items():
            avg_logs[f"epoch_avg_{key}"] = sum(values) / len(values)
        self.log(avg_logs)
        self.epoch_logs.clear()


def rm_output_keys(output):
    for o_idx in range(len(output)):
        for k in list(output[o_idx].keys()):
            if k not in ["multistep_pred_multimasks_high_res", "multistep_pred_ious", "multistep_object_score_logits"]:
                del output[o_idx][k]


import numpy as np



def insert_perf(perf_dict, new_perf):
    for key in new_perf.keys():
        perf_dict[key].append(new_perf[key])


                
import argparse
from omegaconf import OmegaConf
import sys

def override_config_with_args(cfg):
    parser = argparse.ArgumentParser()

    def register_args(prefix, node):
        for key, value in node.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (int, float, bool, str)):
                arg_type = type(value)
                if isinstance(value, bool):
                    # Special handling for booleans
                    parser.add_argument(f"--{full_key}", type=str,
                        choices=["true", "false"], help=f"(bool) override for {full_key}")
                else:
                    parser.add_argument(f"--{full_key}", type=arg_type, help=f"override for {full_key}")
            elif isinstance(value, dict) or OmegaConf.is_dict(value):
                register_args(full_key, value)

    register_args("", cfg)

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    # Apply overrides
    for full_key, val in args_dict.items():
        print(f"Key: {full_key}, Value: {val}")
        if val is None:
            continue
        parts = full_key.split(".")
        sub_cfg = cfg
        for p in parts[:-1]:
            sub_cfg = sub_cfg[p]
        last_key = parts[-1]
        original_type = type(sub_cfg[last_key])
        if original_type == bool:
            val = val.lower() == "true"
        else:
            val = original_type(val)
        sub_cfg[last_key] = val

    return cfg


import editdistance
def compute_wer(model, audio, batch_size, gt_texts, decoder="rnnt", language_id="en", verbose=True):
    assert decoder in ["rnnt", "ctc"], "Decoder must be 'rnnt' or 'ctc'"
    model.cur_decoder = decoder

    # Transcribe using model
    with torch.no_grad():
        predictions = model.transcribe(audio, batch_size=batch_size, logprobs=(decoder == "rnnt"), language_id=language_id)[0]

    total_words = 0
    total_errors = 0

    for pred, gt in zip(predictions, gt_texts):
        hyp_words = pred.strip().split()
        ref_words = gt.strip().split()

        errors = editdistance.eval(hyp_words, ref_words)
        total_errors += errors
        total_words += len(ref_words)

        if verbose:
            wer = errors / len(ref_words) if ref_words else 0.0
            print(f"WER: {wer:.3f} | Pred: {pred} | Ref: {gt}")

    final_wer = total_errors / total_words if total_words else 0.0
    print(f"\nTotal WER ({decoder}): {final_wer:.4f}")
    return final_wer

LANGUAGES = ['hindi','bengali','marathi','telugu','tamil','urdu','gujarati','kannada','odia','malayalam','punjabi','sanskrit']
short_form = ['hi','bn','mr','te','ta','ur','gu','kn','or','ml','pa','sa']
    

def run_eval(logger, type_, model, val_set, noisy_val_set, config, epoch, curr_lang_idx, lang, short_form_lang):
    perf_dict = {}
    log_dict = {}
    
    for mode in ["rnnt", "ctc"]:
        print(f"Evaluating {lang} with {mode} decoder")
        audio = val_set[lang]["audio"]
        transcript = [val_set[lang]['transcript'][os.path.basename(path)] for path in audio]
        val_perf = compute_wer(model, audio, config.batch_size, transcript, decoder=mode, language_id=short_form_lang, verbose=False)
        audio = noisy_val_set[lang]["audio"]
        transcript = [noisy_val_set[lang]['transcript'][os.path.basename(path)] for path in audio]
        noisy_val_perf = compute_wer(model, audio, config.batch_size, transcript, decoder=mode, language_id=short_form_lang, verbose=False)
        perf_dict[f"{mode}_wer"] = val_perf
        perf_dict[f"{mode}_noisy_wer"] = noisy_val_perf
        perf_dict[f"{mode}_avg_wer"] = (val_perf + noisy_val_perf) / 2
        
        log_dict[f"{type_}/perf_{lang}_{mode}_wer"] = perf_dict[f"{mode}_wer"]
        log_dict[f"{type_}/perf_{lang}_{mode}_noisy_wer"] = perf_dict[f"{mode}_noisy_wer"]
        log_dict[f"{type_}/perf_{lang}_{mode}_avg_wer"] = perf_dict[f"{mode}_avg_wer"]
    
    log_dict["epoch"] = epoch
    log_dict["lang"] = curr_lang_idx
    logger.log(log_dict)
    return perf_dict


import numpy as np

def compute_perf_matrix(val_performance, metric="rnnt_wer"):
    langs = list(val_performance.keys())
    max_len = max(len(v) for v in val_performance.values())  # current step count
    T = len(langs)

    perf_matrix = np.full((max_len, T), np.nan)  # fill with NaN to handle missing

    for j, lang in enumerate(langs):
        for i, record in enumerate(val_performance[lang]):
            perf_matrix[i, j] = record[metric]

    return perf_matrix, langs

def compute_bwt_new(val_perf, metric="rnnt_wer"):
    langs = list(val_perf.keys())
    bwt_curves = {lang: [] for lang in langs}

    for i, lang in enumerate(langs):
        # Get WER after training on its own task
        if i >= len(val_perf[lang]):
            continue  # Not trained yet
        wer_ii = val_perf[lang][i][metric]

        # For all future tasks t > i
        for t in range(i + 1, len(langs)):
            # Check if lang was evaluated after training t
            if t < len(val_perf[lang]):
                wer_ti = val_perf[lang][t][metric]
                bwt = wer_ii - wer_ti
                bwt_curves[lang].append((t + 1, bwt))  # task indices are 1-based
    return bwt_curves

import wandb

def log_bwt_curves_wandb(bwt_curves):
    for lang, points in bwt_curves.items():
        if not points:
            continue

        # Create scatter table
        scatter_table = wandb.Table(columns=["Task Index", "BWT"])
        for x, y in points:
            scatter_table.add_data(x, y)

        # Scatter plot with hover
        wandb.log({
            f"BWT/{lang}/scatter": wandb.plot.scatter(
                scatter_table,
                "Task Index",
                "BWT",
                title=f"BWT Scatter - {lang}"
            )
        })

        # Line plot for visual continuity
        x_vals, y_vals = zip(*points)
        wandb.log({
            f"BWT/{lang}/line": wandb.plot.line_series(
                xs=list(x_vals),
                ys=[list(y_vals)],
                keys=[lang],
                title=f"BWT Line - {lang}",
                xname="Task Index"
            )
        })


def freeze_layer(model, num_layers):
    for param in model.parameters():
        param.requires_grad = False
    
    for i, layer in enumerate(model.encoder.layers):
        if i > num_layers:
            for param in layer.parameters():
                param.requires_grad = True
    
    for param in model.decoder.parameters():
        param.requires_grad = True
        
    
    for param in model.ctc_decoder.parameters():
        param.requires_grad = True
        
    for param in  model.joint.parameters():
        param.requires_grad = True
        
def save_model(model, path):
    unfrozen_state_dict = {
        name: param.data
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(unfrozen_state_dict, path)
    
def get_params(model):
    unfrozen_state_dict = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            unfrozen_state_dict[name] = param.data
        
    # for name, param in unfrozen_state_dict.items():
    #     print(name, param)
    return unfrozen_state_dict  
  
def get_params_clone(model):
    unfrozen_state_dict = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            unfrozen_state_dict[name] = param.data.clone()
        
    # for name, param in unfrozen_state_dict.items():
    #     print(name, param)
    return unfrozen_state_dict
  
def get_zero_params(model, device):
    zero_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            zero_state_dict[name] = torch.zeros_like(param.data).to(device)

            
    return zero_state_dict
    # model.load_state_dict(torch.load(path), strict=False)
    
def get_grads(model):
    unfrozen_state_dict = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            unfrozen_state_dict[name] = param.grad
        
    # for name, param in unfrozen_state_dict.items():
    #     print(name, param)
    return unfrozen_state_dict

def set_grads(model, grad_dict):
    for name, param in model.named_parameters():
        if name in grad_dict:
            param.grad = grad_dict[name]
        else:
            param.grad = None

import gc
def check_garbage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


import datetime
import linecache
import os

os.environ['CUDA_LAUNCH_BLOCKING']='1'

#import pynvml3
from py3nvml import py3nvml
import torch
import socket

# different settings
print_tensor_sizes = False
use_incremental = False


if 'GPU_DEBUG' in os.environ:
    gpu_profile_fn = f"Host_{socket.gethostname()}_gpu{os.environ['GPU_DEBUG']}_mem_prof-{datetime.datetime.now():%d-%b-%y-%H-%M-%S}.prof.txt"
    print('profiling gpu usage to ', gpu_profile_fn)


## Global variables
last_tensor_sizes = set()
last_meminfo_used = 0
lineno = None
func_name = None
filename = None
module_name = None


def gpu_profile(frame, event, arg):
    # it is _about to_ execute (!)
    global last_tensor_sizes
    global last_meminfo_used
    global lineno, func_name, filename, module_name

    if event == 'line':
        try:
            # about _previous_ line (!)
            if lineno is not None:
                py3nvml.nvmlInit()
                handle = py3nvml.nvmlDeviceGetHandleByIndex(int(os.environ['GPU_DEBUG']))
                meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                line = linecache.getline(filename, lineno)
                where_str = module_name+' '+func_name+':'+str(lineno)

                new_meminfo_used = meminfo.used
                mem_display = new_meminfo_used-last_meminfo_used if use_incremental else new_meminfo_used
                with open(gpu_profile_fn, 'a+') as f:
                    f.write(f"{where_str:<50}"
                            f":{(mem_display)/1024**2:<7.1f}Mb "
                            f"{line.rstrip()}\n")

                    last_meminfo_used = new_meminfo_used
                    if print_tensor_sizes is True:
                        for tensor in get_tensors():
                            if not hasattr(tensor, 'dbg_alloc_where'):
                                tensor.dbg_alloc_where = where_str
                        new_tensor_sizes = {(type(x), tuple(x.size()), x.dbg_alloc_where)
                                            for x in get_tensors()}
                        for t, s, loc in new_tensor_sizes - last_tensor_sizes:
                            f.write(f'+ {loc:<50} {str(s):<20} {str(t):<10}\n')
                        for t, s, loc in last_tensor_sizes - new_tensor_sizes:
                            f.write(f'- {loc:<50} {str(s):<20} {str(t):<10}\n')
                        last_tensor_sizes = new_tensor_sizes
                py3nvml.nvmlShutdown()

            # save details about line _to be_ executed
            lineno = None

            func_name = frame.f_code.co_name
            filename = frame.f_globals["__file__"]
            if (filename.endswith(".pyc") or
                    filename.endswith(".pyo")):
                filename = filename[:-1]
            module_name = frame.f_globals["__name__"]
            lineno = frame.f_lineno
            
            #only profile codes within the parenet folder, otherwise there are too many function calls into other pytorch scripts
            #need to modify the key words below to suit your case.
            if 'gpu_memory_profiling' not in os.path.dirname(os.path.abspath(filename)):   
                lineno = None  # skip current line evaluation

            if ('car_datasets' in filename
                    or '_exec_config' in func_name
                    or 'gpu_profile' in module_name
                    or 'tee_stdout' in module_name):
                lineno = None  # skip othe unnecessary lines
            
            return gpu_profile

        except (KeyError, AttributeError):
            pass

    return gpu_profile


def get_tensors(gpu_only=True):
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    wandb.init(project="transfer_learning_curves", entity="frozenwolf")
    val_performance = {
    "lang1": [
        {"rnnt_wer": 0.30},  # After training on lang1
        {"rnnt_wer": 0.35},  # After training on lang2
        {"rnnt_wer": 0.40},  # After training on lang3
    ],
    "lang2": [
        {"rnnt_wer": 0.50},  # Untrained (before training on lang2)
        {"rnnt_wer": 0.32},  # After training on lang2
        {"rnnt_wer": 0.33},  # After training on lang3
    ],
    "lang3": [
        {"rnnt_wer": 0.55},  # Untrained (before training on lang3)
        {"rnnt_wer": 0.52},  # Still untrained (before lang3)
        {"rnnt_wer": 0.31},  # After training on lang3
    ]
}
    # bwt_curves, fwt_curves = compute_per_lang_bwt_fwt(val_performance)

    # # 'task_id' is the current number of tasks trained (i.e. step)
    # log_transfer_curves_to_wandb(bwt_curves, fwt_curves)
    
    val_performance = { ### new random 4th value
    "lang1": [
        {"rnnt_wer": 0.30},  # After training on lang1
        {"rnnt_wer": 0.35},  # After traijning on lang2
        {"rnnt_wer": 0.40},  # After training on lang3
    ],
    "lang2": [
        {"rnnt_wer": 0.50},  # Untrained (before training on lang2)
        {"rnnt_wer": 0.32},  # After training on lang2
    ],
    "lang3": [
        {"rnnt_wer": 0.55},  # Untrained (before training on lang3)
    ],
}
    bwt_curves = compute_bwt_new(val_performance)
    print(bwt_curves)
    log_bwt_curves_wandb(bwt_curves)

        