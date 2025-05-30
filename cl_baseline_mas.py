import os
os.environ['GPU_DEBUG']='1'
import wandb
import pickle
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
import tempfile
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
import NeMo.nemo.collections.asr as nemo_asr
from NeMo.nemo.collections.asr.models.hybrid_rnnt_ctc_models import TranscribeConfig, InternalTranscribeConfig
import gc
import sys
from utils import gpu_profile, check_garbage
from utils import Logger, override_config_with_args, insert_perf, compute_bwt_new, save_model, log_bwt_curves_wandb, compute_wer, run_eval, freeze_layer  # Make sure utils is adapted
from torch.distributed.elastic.multiprocessing.errors import record
from torch.amp import GradScaler, autocast
from utils import get_params, get_zero_params, get_grads, get_params_clone, set_grads

def seed_everything(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=5))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    print(f"Node Rank: {node_rank}, Global Rank: {global_rank}, Local Rank (GPU): {local_rank}, World Size: {world_size}")


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return [move_to_device(x, device) for x in batch]
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    else:
        raise TypeError(f"Unsupported type: {type(batch)}")


LANGUAGES = ['hindi','bengali','marathi','telugu','tamil','urdu','gujarati','kannada','odia','malayalam','punjabi','sanskrit']
short_form = ['hi','bn','mr','te','ta','ur','gu','kn','or','ml','pa','sa'] 
    
    
val_performance = {i:[] for i in LANGUAGES}
test_performance = {i:[] for i in LANGUAGES}


def penalty(model, main_importance, prev_params):
    loss = 0
    for n, p in model.named_parameters():
       if p.requires_grad:
           loss += torch.sum(main_importance[n] * (p - prev_params[n]) ** 2)
    return loss


@record
def train():
    dataloader =   None
    setup_distributed()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    seed_everything()
    config = OmegaConf.load("config.yaml")
    config = override_config_with_args(config)


    dataset = pickle.load(open(config.dataset.annotation_path, "rb"))
    
    for dict_name, dict in dataset.items():
        for lang in LANGUAGES:
            dataset[dict_name][lang]['audio'] = [os.path.join(config.dataset.path, dict_name.replace('noisy_',''), lang, file) for file in dataset[dict_name][lang]['audio']]
            assert os.path.exists(dataset[dict_name][lang]['audio'][0]), f"File not found: {dict[lang]['audio'][0]}"
            assert os.path.basename(dataset[dict_name][lang]['audio'][0]) in dataset[dict_name][lang]['transcript'], f"Transcript not found for {dataset[dict_name][lang]['audio'][0]}"
            
    
    train_set, val_set, test_set = dataset["train"], dataset["val"], dataset["test"]
    noisy_val_set, noisy_test_set = dataset["noisy_val"], dataset["noisy_test"]
    



    if is_main_process():
        wandb.init(
        project="CL-ASR",
        config=OmegaConf.to_container(config),
        notes=config.notes,
        entity=config.entity
    )
        track_file_Type = [".py", ".sh", ".yaml", "ipynb", ".json", ".txt"]
        wandb.run.log_code(".", include_fn=lambda path: (
        any([path.endswith(file_type) for file_type in track_file_Type])
        and ("wandb" not in path)
        and (config.output_dir not in path)
        and ("NeMo" not in path)
    ))
        logger = Logger(config, wandb_log=True)

        run_id = wandb.run.id

        if not os.path.exists(config.output_dir):
            os.mkdir(config.output_dir)

        os.mkdir(os.path.join(config.output_dir, run_id))
    
        print("Languages:", LANGUAGES)

        run_id = wandb.run.id
    #     pickle.dump(run_id, open(os.path.join(config.output_dir, "run_id.pkl"), "wb"))
    # else:
    #     while True:
    #         try:
    #             run_id = pickle.load(open(os.path.join(config.output_dir, "run_id.pkl"), "rb"))
    #             break
    #         except:
    #             import time
    #             print("Waiting for main process to create run_id.pkl")
    #             time.sleep(2)
    

    # torch.distributed.barrier()
    # if is_main_process():
    #     os.remove(os.path.join(config.output_dir, "run_id.pkl"))
    
    model =  nemo_asr.models.ASRModel.from_pretrained(f"ai4bharat/indicconformer_stt_{short_form[0]}_hybrid_rnnt_large").to(device)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    freeze_layer(model, config.model.freeze_encoder_till)

    model.encoder.encoder_frozen_till = config.model.freeze_encoder_till

    
    print("Trainable parameters after freezing:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    model.ctc_wer.log_prediction = False
    model.wer.log_prediction = False
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    checkpoint = None
    
    main_importance = None
    for lang_idx, (lang, short_form_lang) in enumerate(zip(LANGUAGES, short_form)):

        torch.distributed.barrier()
        if is_main_process():
            print(f"\n============= Training on language: {lang} =============")

        audio_files = train_set[lang]['audio'][:config.dataset.train_size]
        

        transcripts_dict = train_set[lang]['transcript']
        transcripts = [transcripts_dict[os.path.basename(path)] for path in audio_files]
        durations = train_set[lang]['duration']

        # prepare dataloader
        transcribe_cfg = TranscribeConfig(
            batch_size=config.batch_size,
            return_hypotheses=False,
            num_workers=0,
            verbose=False,
            logprobs=True,
            language_id=short_form_lang,
        )
        transcribe_cfg._internal = InternalTranscribeConfig()
        transcribe_cfg._internal.temp_dir = tempfile.mkdtemp()

        print("Creating dataloader")
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        
        print("Creating dataloader")
        dataloader = model.module._transcribe_input_processing(audio_files, transcribe_cfg, transcripts, durations=durations,
                                                        shuffle=False if config.distributed else True,
                                                        language_id = short_form_lang,
                                                        sampler="ddp" if config.distributed else None)

        importance = get_zero_params(model.module, device=device)
        for epoch in range(config.epochs + 1): ### last epoch to calculate ewc
            torch.distributed.barrier()
            model.train()
            if lang_idx > -1:
                if config.mixed_precision:
                    scaler = GradScaler()  # for AMP
                    
                
                if epoch == config.epochs:
                    model.module.joint.store_sub_enc = False
                    model.module.joint.store_sub_logits = True
                    model.module.ctc_decoder.return_logits_ = True
                    model.module.joint.detach_sub_enc = False
                else:
                    model.module.joint.store_sub_enc = False
                    model.module.joint.store_sub_logits = False
                    model.module.ctc_decoder.return_logits_ = False
                    model.module.joint.detach_sub_enc = False
                
                for batch in tqdm(dataloader, desc=f"[Rank {rank}] Lang: {lang} Epoch: {epoch + 1}"):
                    # print("before, batch",torch.cuda.memory_summary())
                    batch = move_to_device(batch, device)
                    optimizer.zero_grad()
                    # print("after batch",torch.cuda.memory_summary())
                    with autocast(device_type="cuda", enabled=config.mixed_precision):
                        loss, monitor = model.module.training_step(batch, [short_form_lang]*len(batch[0]))
                        
                        if checkpoint is not None:
                            mass_loss = penalty(model.module, main_importance, checkpoint)
                            monitor['mass_loss'] = mass_loss.item()
                            loss = loss + mass_loss*config.cl_config.mas_lambda
                    # print("loss calclated",torch.cuda.memory_summary())
                    
                    if (epoch < config.epochs):
                        # penalty, penalty_avg  =  get_penalty_grads(config, main_fish, get_params(model.module), checkpoint)
                        # monitor['ewc_penalty'] = penalty_avg
                        # set_grads(model.module,penalty)
                        
                        if config.mixed_precision:
                            scaler.scale(loss).backward()
                            # print("after loss backward", torch.cuda.memory_summary())
                            if epoch < config.epochs:
                                scaler.step(optimizer)
                                scaler.update()
                        else:
                            loss.backward()
                            if epoch < config.epochs:
                                optimizer.step()
                        
                        if is_main_process():
                            for key, value in monitor.items():
                                logger.log({f"train/{key}_{lang}": value, "epoch": epoch, "lang": lang_idx})
                                
                    else:
                        decoder_logits = (model.module.ctc_decoder.decoder_logits.flatten(end_dim=-2) ** 2).sum(dim=-1).mean()
                        
                        rnn_logits = 0
                        for i in model.module.joint.store_list:
                            rnn_logits += (i.flatten(end_dim=-2) ** 2).sum(dim=-1).mean()
                        rnn_logits /= len(model.module.joint.store_list)
                        loss = (rnn_logits*(1-config.cl_config.mas_ctx) + decoder_logits*config.cl_config.mas_ctx)
                        loss.backward()
                        
                        for n, p in model.module.named_parameters():
                            if p.requires_grad:
                                if p.grad is not None:
                                    importance[n] += p.grad.abs().detach()
                                    
                                    
                        del decoder_logits, rnn_logits
                         
                      
                
                    del loss, batch, monitor
                    gc.collect()
                    torch.cuda.empty_cache()
                    # print("After deleting")
                    # check_garbage()
                
                if epoch == config.epochs:
                    for k in importance.keys():
                        importance[k] /= len(dataloader)
                        
                    main_importance = importance         
                    checkpoint = get_params_clone(model.module)
                  
                if is_main_process():
                    if epoch < config.epochs:
                        logger.log_epoch_average()
   
            if is_main_process() and epoch == config.epochs-1:
                if config.save_weights:
                    print("Saving weights")
                    save_model(model.module, os.path.join(config.output_dir, run_id, f"model_{lang}.pth"))
                # Evaluation after training each language

                print("Validation eval")
                perf_dict = {}
                for prev_lang in tqdm(LANGUAGES[:lang_idx+1]):
                    perf_dict[prev_lang] = run_eval(logger, "val", model.module, val_set, noisy_val_set, config, epoch, lang_idx, prev_lang, short_form_lang)
                insert_perf(val_performance, perf_dict)
                for modes in ["ctc", "rnnt"]:
                    print(f"Computing {modes} curves")
                    print("val_performance", val_performance)
                    bwt_curves = compute_bwt_new(val_performance, f"{modes}_avg_wer")
                    print("bwt_curves", bwt_curves)
                    log_bwt_curves_wandb(bwt_curves)

                print("Test eval")
                perf_dict = {}
                for prev_lang in tqdm(LANGUAGES[:lang_idx+1]):
                    perf_dict[prev_lang] = run_eval(logger, "test", model.module, test_set, noisy_test_set, config, epoch, lang_idx, prev_lang, short_form_lang)
                insert_perf(test_performance, perf_dict)

                for modes in ["ctc", "rnnt"]:
                    print(f"Computing {modes} curves")
                    print("test_performance", test_performance)
                    bwt_curves = compute_bwt_new(test_performance, f"{modes}_avg_wer")
                    print("bwt_curves", bwt_curves)
                    log_bwt_curves_wandb(bwt_curves)

                logger.reset()

         
                

    cleanup_distributed()


if __name__ == "__main__":
    # sys.settrace(gpu_profile)
    train()
    cleanup_distributed()
