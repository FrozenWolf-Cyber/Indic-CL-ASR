# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import tempfile
import librosa
import warnings

from tqdm.auto import tqdm
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')
warnings.filterwarnings("ignore", category=FutureWarning, module='librosa')
warnings.filterwarnings("ignore", message=".*Audioread support is deprecated.*")
warnings.filterwarnings("ignore", message=".*FutureWarning: get_duration() keyword argument 'filename' has bee.*")
warnings.filterwarnings("ignore", message=".*This alias will be removed in version 1..*")
warnings.filterwarnings("ignore", message=".*ibrosa.core.get_duration.*")
from typing import Any, List, Optional, Tuple
import ffmpeg

def get_duration(file_path):
    probe = ffmpeg.probe(file_path)
    return float(probe['format']['duration'])


import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, InterCTCMixin, TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import TranscriptionReturnType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging, model_utils


import copy
import json
import os
import tempfile
from typing import Any, List, Optional, Tuple

import torch
from pytorch_lightning import Trainer

from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.utils import logging, model_utils

import torch
import torch

import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
import gc
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.asr.data.audio_to_text_dali import AudioToCharDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.parts.preprocessing.parsers import make_parser
from nemo.utils import logging, logging_mode
from nemo.collections.asr.data import audio_to_text_dataset

TranscriptionReturnType = Union[List[str], List['Hypothesis'], Tuple[List[str]], Tuple[List['Hypothesis']]]
GenericTranscriptionType = Union[List[Any], List[List[Any]], Tuple[Any], Tuple[List[Any]], Dict[str, List[Any]]]

@dataclass
class InternalTranscribeConfig:
    # Internal values
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    training_mode: bool = False
    logging_level: Optional[Any] = None

    # Preprocessor values
    dither_value: float = 0.0
    pad_to_value: int = 0

    # Scratch space
    temp_dir: Optional[str] = None

@dataclass
class TranscribeConfig:
    batch_size: int = 4
    return_hypotheses: bool = False
    num_workers: Optional[int] = None
    channel_selector: ChannelSelectorType = None
    augmentor: Optional[DictConfig] = None
    verbose: bool = True
    logprobs: bool = False
    language_id: str = None

    # Utility
    partial_hypothesis: Optional[List[Any]] = None

    _internal: Optional[InternalTranscribeConfig] = None
    
def get_value_from_transcription_config(trcfg, key, default):
    if hasattr(trcfg, key):
        return getattr(trcfg, key)
    else:
        logging.debug(
            f"Using default value of {default} for {key} because it is not present in the transcription config {trcfg}."
        )
        return default
    
def move_to_device(batch, device):
    """
    Recursively move all tensors in `batch` to `device`.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return [move_to_device(x, device) for x in batch]
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    else:
        raise TypeError(f"Unsupported type: {type(batch)}")
    

class TranscriptionTensorDataset(Dataset):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.audio_tensors = config['audio_tensors']
        self.channel_selector = config['channel_selector']
        self.augmentor_cfg = config.get('augmentor', None)
        self.sample_rate = config['sample_rate']

        if self.augmentor_cfg is not None:
            self.augmentor = process_augmentations(self.augmentor_cfg, global_rank=0, world_size=1)
        else:
            self.augmentor = None

        self.length = len(self.audio_tensors)

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError(f"Index {index} out of range for dataset of size {self.length}")

        return self.get_item(index)

    def __len__(self):
        return self.length

    def get_item(self, index):
        samples = self.audio_tensors[index]

        if self.augmentor is not None:
            logging.warning(
                "Audio Augmentations are being applied during inference by moving the tensor onto CPU. "
                "This is highly inefficient and therefore not recommended.",
                mode=logging_mode.ONCE,
            )

            original_dtype = samples.dtype
            samples = samples.to(device='cpu', dtype=torch.float32).numpy()
            segment = AudioSegment(
                samples, self.sample_rate, target_sr=self.sample_rate, channel_selector=self.channel_selector
            )
            samples = self.augmentor.perturb(segment)
            samples = torch.tensor(samples.samples, dtype=original_dtype)

        # Calculate seq length
        seq_len = torch.tensor(samples.shape[0], dtype=torch.long)

        # Dummy text tokens
        text_tokens = torch.tensor([0], dtype=torch.long)
        text_tokens_len = torch.tensor(1, dtype=torch.long)

        return (samples, seq_len, text_tokens, text_tokens_len)
    

class EncDecHybridRNNTCTCModel(EncDecRNNTModel, ASRBPEMixin, InterCTCMixin):
    """Base class for hybrid RNNT/CTC models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        if 'aux_ctc' not in self.cfg:
            raise ValueError(
                "The config need to have a section for the CTC decoder named as aux_ctc for Hybrid models."
            )
        with open_dict(self.cfg.aux_ctc):
            if "feat_in" not in self.cfg.aux_ctc.decoder or (
                not self.cfg.aux_ctc.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self.cfg.aux_ctc.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self.cfg.aux_ctc.decoder or not self.cfg.aux_ctc.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.aux_ctc.decoder.num_classes < 1 and self.cfg.aux_ctc.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.aux_ctc.decoder.num_classes, len(self.cfg.aux_ctc.decoder.vocabulary)
                    )
                )
                self.cfg.aux_ctc.decoder["num_classes"] = len(self.cfg.aux_ctc.decoder.vocabulary)

        self.ctc_decoder = EncDecRNNTModel.from_config_dict(self.cfg.aux_ctc.decoder)
        self.ctc_loss_weight = self.cfg.aux_ctc.get("ctc_loss_weight", 0.5)

        self.ctc_loss = CTCLoss(
            num_classes=self.ctc_decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
        )

        ctc_decoding_cfg = self.cfg.aux_ctc.get('decoding', None)
        if ctc_decoding_cfg is None:
            ctc_decoding_cfg = OmegaConf.structured(CTCDecodingConfig)
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg

        self.ctc_decoding = CTCDecoding(self.cfg.aux_ctc.decoding, vocabulary=self.ctc_decoder.vocabulary)
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
        )

        # setting the RNNT decoder as the default one
        self.cur_decoder = "rnnt"

        # setting up interCTC loss (from InterCTCMixin)
        self.setup_interctc(decoder_name='ctc_decoder', loss_name='ctc_loss', wer_name='ctc_wer')


    @torch.no_grad()
    def transcribe(
        self,
        audio,
        batch_size: int = 4,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        verbose: bool = True,
        logprobs: bool = False,
        language_id: str = None,
        channel_selector  = None,
        augmentor = None,
        override_config = None,
        **config_kwargs,
    ):

        transcribe_cfg = TranscribeConfig(
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
                logprobs=logprobs,
                language_id=language_id,
                **config_kwargs,
            )
    
        transcribe_cfg._internal = InternalTranscribeConfig()

        # Hold the results here
        results = None  # type: GenericTranscriptionType
        try:
            generator = self.transcribe_generator(audio, override_config=transcribe_cfg, language_id=language_id)
            for processed_outputs in generator:
                # Store results
                if isinstance(processed_outputs, list):
                    # Create a results of the same type as each element in processed_outputs
                    if results is None:
                        results = []
                        # if list of inner list of results, copy structure
                        if isinstance(processed_outputs[0], list):
                            for _ in processed_outputs:
                                results.append([])
                    # If nested list structure
                    if isinstance(processed_outputs[0], list):
                        for i, processed_output in enumerate(processed_outputs):
                            results[i].extend(processed_output)
                    else:
                        # If flat list structure
                        results.extend(processed_outputs)
                elif isinstance(processed_outputs, dict):
                    # Create a results of the same type as each element in processed_outputs
                    if results is None:
                        results = processed_outputs
                    else:
                        for k, v in processed_outputs.items():
                            results[k].extend(v)
                elif isinstance(processed_outputs, tuple):
                    # Create a results of the same type as each element in processed_outputs
                    if results is None:
                        results = tuple([[] for _ in processed_outputs])
                    # If nested list structure
                    if isinstance(processed_outputs[0], list):
                        for i, processed_output in enumerate(processed_outputs):
                            results[i].extend(processed_output)
                    else:
                        # If flat list structure
                        if len(processed_outputs) != len(results):
                            raise RuntimeError(
                                f"The number of elements in the result ({len(results)}) does not "
                                f"match the results of the current batch ({len(processed_outputs)})."
                            )
                        for i, processed_output in enumerate(processed_outputs):
                            results[i].append(processed_output)
                else:
                    raise NotImplementedError(
                        "Given output result for transcription is not supported. "
                        "Please return a list of results, list of list of results, "
                        "a dict of list of results, or "
                        "a tuple of list of results."
                    )
        except StopIteration:
            pass
        return results


    def transcribe_generator(self, audio, override_config: Optional[TranscribeConfig], language_id):

        if override_config is None:
            override_config = TranscribeConfig()

        if not hasattr(override_config, '_internal'):
            raise ValueError(
                "`transcribe_cfg must have an `_internal` argument, which must be of an object of type "
                "InternalTranscribeConfig or its subclass."
            )

        # Add new internal config
        if override_config._internal is None:
            override_config._internal = InternalTranscribeConfig()
        else:
            # Check if internal config is valid
            if not isinstance(override_config._internal, InternalTranscribeConfig):
                raise ValueError(
                    "`transcribe_cfg._internal` must be of an object of type InternalTranscribeConfig or "
                    "its subclass"
                )

        transcribe_cfg = override_config
        try:
            # Initialize and assert the transcription environment
            self._transcribe_on_begin(audio, transcribe_cfg)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                transcribe_cfg._internal.temp_dir = tmpdir
                dataloader = self._transcribe_input_processing(audio, transcribe_cfg, transcripts=None, language_id= language_id)
                if hasattr(transcribe_cfg, 'verbose'):
                    verbose = transcribe_cfg.verbose
                else:
                    verbose = True
                for test_batch in tqdm(dataloader, desc="Transcribing", disable=not verbose):
                    # Move batch to device
                    test_batch = move_to_device(test_batch, transcribe_cfg._internal.device)
                    # Run forward pass
                    model_outputs = self._transcribe_forward(test_batch, transcribe_cfg)
                    processed_outputs = self._transcribe_output_processing(model_outputs, transcribe_cfg)
                    # clear up memory
                    del test_batch, model_outputs
                    # Yield results if generator
                    yield processed_outputs
                    del processed_outputs
        finally:
            # set mode back to its original value
            self._transcribe_on_end(transcribe_cfg)

    def _transcribe_input_manifest_processing(
        
        self, audio_files: List[str], temp_dir: str, trcfg: TranscribeConfig, durations=None, transcripts = None, language_id='ta'
    ) -> Dict[str, Any]:
        """
        Internal function to process the input audio filepaths and return a config dict for the dataloader.
        Specializes to ASR models which can have Encoder-Decoder-Joint architectures.
        Args:
            audio_files: A list of string filepaths for audio files.
            temp_dir: A temporary directory to store intermediate files.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        Returns:
            A config dict that is used to setup the dataloader for transcription.
        """
        transcript_also = True
        if transcripts is None:
            transcripts = len(audio_files)*['']
            transcript_also = False
        
        if durations is None:
            durations = len(audio_files)*[0]    
        
        with open(os.path.join(temp_dir, 'manifest.json'), 'w', encoding='utf-8') as fp:
            for audio_file, transcript, duration_ in tqdm(zip(audio_files, transcripts, durations), total=len(audio_files)):
                if isinstance(audio_file, str):
                    if transcript_also:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            # duration = get_duration(audio_file)
                        entry = {'audio_filepath': audio_file, 'duration': duration_, 'text': transcript, 'lang': language_id}
                    else:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                        
                    fp.write(json.dumps(entry) + '\n')
                elif isinstance(audio_file, dict):
                    fp.write(json.dumps(audio_file) + '\n')
                else:
                    raise ValueError(
                        f"Input `audio` is of type {type(audio_file)}. "
                        "Only `str` (path to audio file) or `dict` are supported as input."
                    )
        ds_config = {
            'paths2audio_files': audio_files,
            'batch_size': get_value_from_transcription_config(trcfg, 'batch_size', 4),
            'temp_dir': temp_dir,
            'num_workers': get_value_from_transcription_config(trcfg, 'num_workers', 0),
            'channel_selector': get_value_from_transcription_config(trcfg, 'channel_selector', None),
            'text_field': get_value_from_transcription_config(trcfg, 'text_field', 'text'),
            'lang_field': get_value_from_transcription_config(trcfg, 'lang_field', 'lang'),
        }
        augmentor = get_value_from_transcription_config(trcfg, 'augmentor', None)
        if augmentor:
            ds_config['augmentor'] = augmentor
        return ds_config

    """
    Transcribe Execution Flow
    """
    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        """
        Internal function to setup the model for transcription. Perform all setup and pre-checks here.
        Args:
            audio: Of type `GenericTranscriptionType`
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        if audio is None:
            return {}
        if isinstance(audio, (str, np.ndarray, torch.Tensor)):
            audio = [audio]
        if isinstance(audio, list) and len(audio) == 0:
            return {}
        _params = next(self.parameters())
        if trcfg._internal.device is None:
            trcfg._internal.device = _params.device
        if trcfg._internal.dtype is None:
            trcfg._internal.dtype = _params.dtype
        # Set num_workers
        num_workers = get_value_from_transcription_config(trcfg, 'num_workers', default=0)
        if num_workers is None:
            _batch_size = get_value_from_transcription_config(trcfg, 'batch_size', default=4)
            num_workers = min(_batch_size, os.cpu_count() - 1)
        # Assign num_workers if available as key in trcfg
        if hasattr(trcfg, 'num_workers'):
            trcfg.num_workers = num_workers
        # Model's mode and device
        trcfg._internal.training_mode = self.training
        # Switch model to evaluation mode
        if hasattr(self, 'preprocessor'):
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'dither'):
                trcfg._internal.dither_value = self.preprocessor.featurizer.dither
                self.preprocessor.featurizer.dither = 0.0
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'pad_to'):
                trcfg._internal.pad_to_value = self.preprocessor.featurizer.pad_to
                self.preprocessor.featurizer.pad_to = 0
        # Switch model to evaluation mode
        self.eval()
        # Disable logging
        trcfg._internal.logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

    def _transcribe_input_processing(self, audio, trcfg: TranscribeConfig, transcripts=None, durations=None, language_id='ta', shuffle=False, sampler=None):
        """
        Internal function to process the input audio data and return a DataLoader. This function is called by
        `transcribe()` and `transcribe_generator()` to setup the input data for transcription.
        Args:
            audio: Of type `GenericTranscriptionType`
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        Returns:
            A DataLoader object that is used to iterate over the input audio data.
        """
        if isinstance(audio, (list, tuple)):
            if len(audio) == 0:
                raise ValueError("Input `audio` is empty")
        else:
            audio = [audio]
        # Check if audio is a list of strings (filepaths or manifests)
        if isinstance(audio[0], str):
            audio_files = list(audio)
            tmp_dir = trcfg._internal.temp_dir
            print("Creating Manifest")
            ds_config = self._transcribe_input_manifest_processing(audio_files, tmp_dir, trcfg, transcripts=transcripts, durations=durations, language_id=language_id)
            print("Creating Dataloader")
            ds_config['shuffle'] = shuffle
            ds_config['sampler'] = sampler
            temp_dataloader = self._setup_transcribe_dataloader(ds_config)
            return temp_dataloader
        # Check if audio is a list of numpy or torch tensors
        elif isinstance(audio[0], (np.ndarray, torch.Tensor)):
            audio_tensors = list(audio)
            # Convert numpy tensors to torch tensors
            if any([isinstance(_tensor, np.ndarray) for _tensor in audio_tensors]):
                audio_tensors = [
                    torch.as_tensor(audio_tensor) if isinstance(audio_tensor, np.ndarray) else audio_tensor
                    for audio_tensor in audio_tensors
                ]
            tmp_dir = trcfg._internal.temp_dir
            ds_config = self._transcribe_input_tensor_processing(audio_tensors, tmp_dir, trcfg)
            temp_dataloader = self._setup_transcribe_tensor_dataloader(ds_config, trcfg)
            return temp_dataloader
        else:
            raise ValueError(
                f"Input `audio` is of type {type(audio[0])}. "
                "Only `str` (path to audio file), `np.ndarray`, and `torch.Tensor` "
                "are supported as input."
            )
    def _transcribe_input_tensor_processing(
        self, audio_tensors: List[Union[np.ndarray, torch.Tensor]], temp_dir: str, trcfg: TranscribeConfig
    ):
        """
        Internal function to process the input audio tensors and return a config dict for the dataloader.
        Args:
            audio_tensors: A list of numpy or torch tensors. The user must ensure that they satisfy the correct
                sample rate and channel format.
            temp_dir: A temporary directory to store intermediate files.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        Returns:
            A config dict that is used to setup the dataloader for transcription.
        """
        # Check if sample rate is set
        sample_rate = None
        if hasattr(self, 'cfg') and 'sample_rate' in self.cfg:
            sample_rate = self.cfg.sample_rate
        elif hasattr(self, 'sample_rate'):
            sample_rate = self.sample_rate
        if sample_rate is None:
            raise RuntimeError(
                "Provided `audio` data contains numpy or torch tensors, however the class "
                "does not have `sample_rate` attribute. Please set `sample_rate` attribute to the model explicitly."
            )
        ds_config = {
            'audio_tensors': audio_tensors,
            'batch_size': get_value_from_transcription_config(trcfg, 'batch_size', 4),
            'temp_dir': temp_dir,
            'num_workers': get_value_from_transcription_config(trcfg, 'num_workers', 0),
            'channel_selector': get_value_from_transcription_config(trcfg, 'channel_selector', None),
            'sample_rate': sample_rate,
        }
        augmentor = get_value_from_transcription_config(trcfg, 'augmentor', None)
        if augmentor:
            ds_config['augmentor'] = augmentor
        return ds_config



    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        """
        Internal function to perform the model's custom forward pass to return outputs that are processed by
        `_transcribe_output_processing()`.
        This function is called by `transcribe()` and `transcribe_generator()` to perform the model's forward pass.
        Args:
            batch: A batch of input data from the data loader that is used to perform the model's forward pass.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        Returns:
            The model's outputs that are processed by `_transcribe_output_processing()`.
        """
            # CTC Path

        if self.cur_decoder == "rnnt":
            encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])
            output = dict(encoded=encoded, encoded_len=encoded_len)
            return output

        # CTC Path
        encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])
        if "multisoftmax" not in self.cfg.decoder:
            language_ids = None
        else:
            language_ids = [trcfg.language_id] * len(batch[0])
            
        logits = self.ctc_decoder(encoder_output=encoded, language_ids=language_ids)
        output = dict(logits=logits, encoded_len=encoded_len, language_ids=language_ids)
        
        del encoded
        return output

    def _transcribe_output_processing(
        self, outputs, trcfg: TranscribeConfig
    ) -> Tuple[List['Hypothesis'], List['Hypothesis']]:
        if self.cur_decoder == "rnnt":
            encoded = outputs.pop('encoded')
            encoded_len = outputs.pop('encoded_len')

            if "multisoftmax" not in self.cfg.decoder:
                language_ids = None
            else:
                language_ids = [trcfg.language_id] * len(encoded)

            best_hyp, all_hyp = self.decoding.rnnt_decoder_predictions_tensor(
                encoded,
                encoded_len,
                return_hypotheses=trcfg.return_hypotheses,
                partial_hypotheses=trcfg.partial_hypothesis,
                lang_ids=language_ids,
            )

            # cleanup memory
            del encoded, encoded_len

            hypotheses = []
            all_hypotheses = []

            hypotheses += best_hyp
            if all_hyp is not None:
                all_hypotheses += all_hyp
            else:
                all_hypotheses += best_hyp

            return (hypotheses, all_hypotheses)

        # CTC Path
        logits = outputs.pop('logits')
        encoded_len = outputs.pop('encoded_len')
        language_ids = outputs.pop('language_ids')
        best_hyp, all_hyp = self.ctc_decoding.ctc_decoder_predictions_tensor(
            logits, encoded_len, return_hypotheses=trcfg.return_hypotheses, lang_ids=language_ids
        )
        logits = logits.cpu()
        if trcfg.return_hypotheses:
            # dump log probs per file
            for idx in range(logits.shape[0]):
                best_hyp[idx].y_sequence = logits[idx][: encoded_len[idx]]
                if best_hyp[idx].alignments is None:
                    best_hyp[idx].alignments = best_hyp[idx].y_sequence
        # DEPRECATED?
        if trcfg.logprobs:
            logits_list = []
            for logit, elen in zip(logits, encoded_len):
                logits_list.append(logit[:elen])
            return logits_list
        del logits, encoded_len
        hypotheses = []
        all_hypotheses = []
        hypotheses += best_hyp
        if all_hyp is not None:
            all_hypotheses += all_hyp
        else:
            all_hypotheses += best_hyp
        return (hypotheses, all_hypotheses)




    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        """
        Internal function to teardown the model after transcription. Perform all teardown and post-checks here.
        Args:
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        # set mode back to its original value
        self.train(mode=trcfg._internal.training_mode)
        if hasattr(self, 'preprocessor'):
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'dither'):
                self.preprocessor.featurizer.dither = trcfg._internal.dither_value
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'pad_to'):
                self.preprocessor.featurizer.pad_to = trcfg._internal.pad_to_value
        if trcfg._internal.logging_level is not None:
            logging.set_verbosity(trcfg._internal.logging_level)

    def _setup_transcribe_tensor_dataloader(self, config: Dict, trcfg: TranscribeConfig) -> DataLoader:
        """
        Internal function to setup the dataloader for transcription. This function is called by
        `transcribe()` and `transcribe_generator()` to setup the input data for transcription.
        Args:
            config: A config dict that is used to setup the dataloader for transcription. It can be generated either
                by `_transcribe_input_manifest_processing()` or `_transcribe_input_tensor_processing()`.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.
        Returns:
            A DataLoader object that is used to iterate over the input audio data.
        """
        dataset = TranscriptionTensorDataset(config)
        # Import collate function here to avoid circular imports
        from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
        # Calculate pad id
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'pad_id'):
            pad_id = self.tokenizer.pad_id
        elif hasattr(self, 'transcribe_pad_id'):
            logging.info("Pad id is explicitly set to `model.transcribe_pad_id` = {}".format(self.transcribe_pad_id))
            pad_id = self.transcribe_pad_id
        else:
            logging.info(
                "Pad id is being set to 0 because it could not be resolved from the tokenizer. "
                "This can happen for various reasons, especially for character based models. "
                "If pad id is incorrect, please provide the pad id explicitly by setting "
                "`model.transcribe_pad_id`."
            )
            pad_id = 0
        return DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=False,
            drop_last=False,
            collate_fn=partial(_speech_collate_fn, pad_id=pad_id),)
        
    def change_vocabulary(
        self,
        new_vocabulary: List[str],
        decoding_cfg: Optional[DictConfig] = None,
        ctc_decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning a pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
                this is target alphabet.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            ctc_decoding_cfg: A config for CTC decoding, which is optional and can be used to change decoding type.

        Returns: None

        """
        super().change_vocabulary(new_vocabulary=new_vocabulary, decoding_cfg=decoding_cfg)

        # set up the new tokenizer for the CTC decoder
        if hasattr(self, 'ctc_decoder'):
            if self.ctc_decoder.vocabulary == new_vocabulary:
                logging.warning(
                    f"Old {self.ctc_decoder.vocabulary} and new {new_vocabulary} match. Not changing anything."
                )
            else:
                if new_vocabulary is None or len(new_vocabulary) == 0:
                    raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
                decoder_config = self.ctc_decoder.to_config_dict()
                new_decoder_config = copy.deepcopy(decoder_config)
                new_decoder_config['vocabulary'] = new_vocabulary
                new_decoder_config['num_classes'] = len(new_vocabulary)

                del self.ctc_decoder
                self.ctc_decoder = EncDecHybridRNNTCTCModel.from_config_dict(new_decoder_config)
                del self.ctc_loss
                self.ctc_loss = CTCLoss(
                    num_classes=self.ctc_decoder.num_classes_with_blank - 1,
                    zero_infinity=True,
                    reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
                )

                if ctc_decoding_cfg is None:
                    # Assume same decoding config as before
                    logging.info("No `ctc_decoding_cfg` passed when changing decoding strategy, using internal config")
                    ctc_decoding_cfg = self.cfg.aux_ctc.decoding

                # Assert the decoding config with all hyper parameters
                ctc_decoding_cls = OmegaConf.structured(CTCDecodingConfig)
                ctc_decoding_cls = OmegaConf.create(OmegaConf.to_container(ctc_decoding_cls))
                ctc_decoding_cfg = OmegaConf.merge(ctc_decoding_cls, ctc_decoding_cfg)

                self.ctc_decoding = CTCDecoding(decoding_cfg=ctc_decoding_cfg, vocabulary=self.ctc_decoder.vocabulary)

                self.ctc_wer = WER(
                    decoding=self.ctc_decoding,
                    use_cer=self.ctc_wer.use_cer,
                    log_prediction=self.ctc_wer.log_prediction,
                    dist_sync_on_step=True,
                )

                # Update config
                with open_dict(self.cfg.aux_ctc):
                    self.cfg.aux_ctc.decoding = ctc_decoding_cfg

                with open_dict(self.cfg.aux_ctc):
                    self.cfg.aux_ctc.decoder = new_decoder_config

                ds_keys = ['train_ds', 'validation_ds', 'test_ds']
                for key in ds_keys:
                    if key in self.cfg:
                        with open_dict(self.cfg[key]):
                            self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

                logging.info(f"Changed the tokenizer of the CTC decoder to {self.ctc_decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig = None, decoder_type: str = None):
        """
        Changes decoding strategy used during RNNT decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            decoder_type: (str) Can be set to 'rnnt' or 'ctc' to switch between appropriate decoder in a
                model having RNN-T and CTC decoders. Defaults to None, in which case RNN-T decoder is
                used. If set to 'ctc', it raises error if 'ctc_decoder' is not an attribute of the model.
        """
        if decoder_type is None or decoder_type == 'rnnt':
            self.cur_decoder = "rnnt"
            return super().change_decoding_strategy(decoding_cfg=decoding_cfg)

        assert decoder_type == 'ctc' and hasattr(self, 'ctc_decoder')
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.aux_ctc.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.ctc_decoding = CTCDecoding(decoding_cfg=decoding_cfg, vocabulary=self.ctc_decoder.vocabulary)

        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.ctc_wer.use_cer,
            log_prediction=self.ctc_wer.log_prediction,
            dist_sync_on_step=True,
        )

        self.ctc_decoder.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.aux_ctc):
            self.cfg.aux_ctc.decoding = decoding_cfg

        self.cur_decoder = "ctc"
        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.aux_ctc.decoding)}")

    # PTL-specific methods
    def training_step(self, batch, lang_ids, return_probs=False):
        signal, signal_len, transcript, transcript_len = batch
        del batch
        gc.collect(); gc.collect()
        torch.cuda.empty_cache()
        language_ids = lang_ids

        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal
        gc.collect(); gc.collect()
        torch.cuda.empty_cache()
        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
        del target_length, states, signal_len
        gc.collect(); gc.collect()
        torch.cuda.empty_cache()
        compute_wer = True

        # If fused Joint-Loss-WER is not used
        monitor = {}

        loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
                language_ids=language_ids
            )
        monitor['training_batch_wer'] = wer
        del wer, _, decoder
        gc.collect(); gc.collect()
        torch.cuda.empty_cache()

        log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=language_ids)
        del encoded
        ctc_loss = self.ctc_loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        monitor['train_rnnt_loss'] = loss_value.item()
        monitor['train_ctc_loss'] = ctc_loss.item()
        
        loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
        self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                    lang_ids=language_ids,
                )
        ctc_wer, _, _ = self.ctc_wer.compute()
        self.ctc_wer.reset()
        monitor['training_batch_wer_ctc'] = ctc_wer.item()
        # tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})
        del ctc_wer, ctc_loss, _
        if not return_probs:
            del log_probs
        gc.collect(); gc.collect()
        torch.cuda.empty_cache()

        monitor['train_loss'] = loss_value.item()
        ### delete rest and reset cuda cache
        del encoded_len, transcript, transcript_len, language_ids
        gc.collect(); gc.collect()
        torch.cuda.empty_cache()
        
        if return_probs:
            return loss_value, monitor, log_probs
        
        
        return loss_value, monitor



    # PTL-specific methods
    def training_step_custom(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        if "multisoftmax" not in self.cfg.decoder: #CTEMO
            signal, signal_len, transcript, transcript_len = batch
            language_ids = None
        else:
            signal, signal_len, transcript, transcript_len, sample_ids, language_ids = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        # if hasattr(self, '_trainer') and self._trainer is not None:
        #     log_every_n_steps = self._trainer.log_every_n_steps
        #     sample_id = self._trainer.global_step
        # else:
        #     log_every_n_steps = 1
        #     sample_id = batch_nb

        # if (sample_id + 1) % log_every_n_steps == 0:
        #     compute_wer = False
        # else:
        compute_wer = False

        # If fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder, language_ids=language_ids) #CTEMO
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # tensorboard_logs = {
            #     'learning_rate': self._optimizer.param_groups[0]['lr'],
            #     'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            # }

            if compute_wer:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                # tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:  # If fused Joint-Loss-WER is used
            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
                language_ids=language_ids
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # tensorboard_logs = {
            #     'learning_rate': self._optimizer.param_groups[0]['lr'],
            #     'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            # }

            # if compute_wer:
                # tensorboard_logs.update({'training_batch_wer': wer})

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=language_ids)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            # tensorboard_logs['train_rnnt_loss'] = loss_value
            # tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                if "multisoftmax" in self.cfg.decoder:
                    self.ctc_wer.update(
                        predictions=log_probs,
                        targets=transcript,
                        targets_lengths=transcript_len,
                        predictions_lengths=encoded_len,
                        lang_ids=language_ids,
                    )
                else:
                    self.ctc_wer.update(
                        predictions=log_probs,
                        targets=transcript,
                        targets_lengths=transcript_len,
                        predictions_lengths=encoded_len,
                    )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                # tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})

        # note that we want to apply interctc independent of whether main ctc
        # loss is used or not (to allow rnnt + interctc training).
        # assuming ``ctc_loss_weight=0.3`` and interctc is applied to a single
        # layer with weight of ``0.1``, the total loss will be
        # ``loss = 0.9 * (0.3 * ctc_loss + 0.7 * rnnt_loss) + 0.1 * interctc_loss``
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )
        # tensorboard_logs.update(additional_logs)
        # tensorboard_logs['train_loss'] = loss_value
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Log items
        # self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return loss_value


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO: add support for CTC decoding
        if "multisoftmax" not in self.cfg.decoder: #CTEMO
            signal, signal_len, transcript, transcript_len = batch
            language_ids = None
        else:
            signal, signal_len, transcript, transcript_len, sample_ids, language_ids = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        if "multisoftmax" in self.cfg.decoder: #CTEMO
            best_hyp_text, all_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False, lang_ids=language_ids
            )
        else:
            best_hyp_text, all_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
            )
        # breakpoint()
        sample_id = sample_ids.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp_text))

    def validation_pass(self, batch, batch_idx, dataloader_idx):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        if "multisoftmax" not in self.cfg.decoder:
            signal, signal_len, transcript, transcript_len = batch
            language_ids=None
        else:
            signal, signal_len, transcript, transcript_len, sample_ids, language_ids = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}
        loss_value = None

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder, language_ids=language_ids) # CTEMO

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )
                tensorboard_logs['val_loss'] = loss_value

            if "multisoftmax" in self.cfg.decoder: #CTEMO
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                    lang_ids=language_ids,
                )
            else:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
                language_ids=language_ids #CTEMO
            )
            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=language_ids) #CTEMO
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_rnnt_loss'] = loss_value
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['val_loss'] = loss_value
        self.ctc_wer.update(
            predictions=log_probs, targets=transcript, targets_lengths=transcript_len, predictions_lengths=encoded_len, lang_ids=language_ids #CTEMO
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            compute_loss=self.compute_eval_loss,
            log_wer_num_denom=True,
            log_prefix="val_",
        )
        if self.compute_eval_loss:
            # overriding total loss value. Note that the previous
            # rnnt + ctc loss is available in metrics as "val_final_loss" now
            tensorboard_logs['val_loss'] = loss_value
        tensorboard_logs.update(additional_logs)
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        return tensorboard_logs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        tensorboard_logs = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(tensorboard_logs)
        else:
            self.validation_step_outputs.append(tensorboard_logs)

        return tensorboard_logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}
        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['val_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['val_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['val_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom
        metrics = {**val_loss_log, 'log': tensorboard_logs}
        self.finalize_interctc_metrics(metrics, outputs, prefix="val_")
        return metrics

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_loss_log = {'test_loss': test_loss_mean}
        else:
            test_loss_log = {}
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**test_loss_log, 'test_wer': wer_num.float() / wer_denom}

        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['test_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['test_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['test_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        metrics = {**test_loss_log, 'log': tensorboard_logs}
        self.finalize_interctc_metrics(metrics, outputs, prefix="test_")
        return metrics

    # EncDecRNNTModel is exported in 2 parts
    def list_export_subnets(self):
        if self.cur_decoder == 'rnnt':
            return ['encoder', 'decoder_joint']
        else:
            return ['self']

    @property
    def output_module(self):
        if self.cur_decoder == 'rnnt':
            return self.decoder
        else:
            return self.ctc_decoder

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results
