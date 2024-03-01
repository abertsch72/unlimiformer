
import os
import sys

import torch.cuda
from transformers.utils import logging

sys.path.insert(0, os.getcwd())

from dataclasses import dataclass, field

from transformers.trainer_utils import IntervalStrategy
from transformers import Seq2SeqTrainingArguments

logger = logging.get_logger('swed_logger')

@dataclass
class TrainingOverridesArguments(Seq2SeqTrainingArguments):
    """
    To use if, it requires evaluation_strategy == IntervalStrategy.STEPS
    """
    eval_steps_override: float = field(default=0, metadata={"help": "a fraction, to set the the save_steps w.r.t to number of steps in "
                                                                    "a single epoch. changes eval_steps. 0 to disable (default)"})
    save_steps_override: float = field(default=0, metadata={"help": "a fraction, to set the the save_steps w.r.t to number of steps in "
                                                                    "a single epoch. changes save_steps. must be a multiple of eval_steps"
                                                                    " (or eval_steps_override if given). 0 to disable (default)"})

    eval_fraction: float = field(default=1, metadata={
        "help": "A float in (0,1] that corresponds to how much of the eval set to use during evaluations "
                "(same subset all the time) or an integer >= 2 which amounts to the absolute number of training "
                "samples to use. 1. to disable it and use the entire eval set "})

    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models). If AUTH_TOKEN is set as an environment variable, would use that"
        },
    )

    fp16_padding: bool = field(
        default=False,
        metadata={"help": "Whether to use padding for fp16"},
    )


    def __post_init__(self):
        super(TrainingOverridesArguments, self).__post_init__()
        if self.eval_steps_override > 0 or self.save_steps_override > 0:
            if self.evaluation_strategy != IntervalStrategy.STEPS:
                raise ValueError(
                    f"using eval/save steps override requires evaluation strategy to be  {IntervalStrategy.STEPS}"
                )
            if self.save_steps_override == 0 or self.eval_steps_override == 0:
                raise ValueError(
                    f"using eval/save steps override requires both overrides to be non zero"
                )
            diff = (self.save_steps_override / self.eval_steps_override) % 1
            if min(1-diff, diff) > 1e-5:  # we do it like that to support fractions modulo as well, with loss of precision
                raise ValueError(
                    f"using eval/save steps override requires save steps override to be a multiple of eval_steps_override"
                )
        if self.use_auth_token and 'AUTH_TOKEN' in os.environ:
            self.use_auth_token = os.getenv('AUTH_TOKEN')

    @property
    def effective_batch_size(self):
        if not hasattr(self, '_ebs'):
            n_gpu = self.n_gpu if torch.cuda.is_available() else 1  # may be on cpu
            self._ebs = self.per_device_train_batch_size * self.gradient_accumulation_steps * n_gpu
            logger.warning(f'Training with {self.per_device_train_batch_size} per_device_train_size, {self.n_gpu} gpus and '
                        f'{self.gradient_accumulation_steps} gradient accumulation steps, resulting in {self._ebs} effective batch size')
        return self._ebs

    def apply_overrides(self, dataset_size):
        # Uri:
        return
        
        if self.eval_steps_override == 0:
            return
        es, ss = self.eval_steps, self.save_steps
        total_steps_per_epoch = dataset_size / self.effective_batch_size  # note that this may not be an  integer
        eval_steps = int(total_steps_per_epoch * self.eval_steps_override)
        if eval_steps >= self.logging_steps:
            if eval_steps % self.logging_steps != 0:
                logger.warning(f'Eval steps override would result in eval every {eval_steps} steps, but it is not a '
                            f'multiple of logging steps ({self.logging_steps}) so changing to '
                            f'{eval_steps + self.logging_steps - eval_steps % self.logging_steps}')
                eval_steps = eval_steps + self.logging_steps - eval_steps % self.logging_steps
        elif eval_steps < self.logging_steps:
            logger.warning(f'Eval steps override would result in eval every {eval_steps} steps, but it is not a '
                        f'multiple of logging steps ({self.logging_steps}) so changing to {self.logging_steps}')
            eval_steps = self.logging_steps
        self.eval_steps = eval_steps

        save_steps = int(total_steps_per_epoch * self.save_steps_override)
        if save_steps < eval_steps or save_steps % eval_steps != 0:
            logger.warning(f'Save steps override would result in eval every {save_steps} steps, but it is not a '
                        f'multiple of eval steps ({eval_steps}) so changing to '
                        f'{save_steps + eval_steps - save_steps % self.eval_steps}')
            save_steps = save_steps + eval_steps - save_steps % self.eval_steps
        self.save_steps = save_steps

        logger.warning(f'Using overrides with dataset of size {dataset_size} and effective batch size of '
                    f'{self.effective_batch_size}, moving from (eval_steps, save_steps) '
                    f'of {(es, ss)} to {(self.eval_steps, self.save_steps)}')
