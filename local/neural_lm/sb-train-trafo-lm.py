#!/usr/bin/env python3
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

import sys
import torch
import speechbrain as sb
import itertools
from hyperpyyaml import load_hyperpyyaml
from lm_dataset import construct_static_lm_data, EndlessDynBatchIterable, EndlessDynamicLMData
import time



class LM(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the sentence batches to the output probabilities."""
        batch = batch.to(self.device)
        tokens_bos, _ = batch.tokens_bos
        logits = self.hparams.model(tokens_bos)
        pred = self.hparams.log_softmax(logits)
        return pred

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        batch = batch.to(self.device)
        tokens_eos, tokens_len = batch.tokens_eos
        loss = self.hparams.compute_cost(
            predictions, tokens_eos, length=tokens_len
        )
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        (loss / self.hparams.accu_steps).backward()

        if self.step % self.hparams.accu_steps == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)


            if isinstance(
                self.hparams.lr_annealing, sb.nnet.schedulers.NoamScheduler
            ) or isinstance(
                self.hparams.lr_annealing,
                sb.nnet.schedulers.CyclicCosineScheduler,
            ):
                self.hparams.lr_annealing(self.optimizer)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss

    def on_fit_start(self):
        super().on_fit_start()
        self.optimizer.param_groups[0]['capturable'] = True

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            if not (
                isinstance(
                    self.hparams.lr_annealing, sb.nnet.schedulers.NoamScheduler
                )
                or isinstance(
                    self.hparams.lr_annealing,
                    sb.nnet.schedulers.CyclicCosineScheduler,
                )
            ):
                old_lr, new_lr = self.hparams.lr_annealing(stage_loss)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            else:
                old_lr = self.hparams.lr_annealing.current_lr

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats, min_keys=["loss"],
            )



if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    train_data = EndlessDynamicLMData(
            hparams["train_data"],
            hparams["tokenizer"],
            hparams["bos_index"],
            hparams["eos_index"],
            hparams["dynbatch_kwargs"],
    )

    valid_data = construct_static_lm_data(
            hparams["valid_data"],
            hparams["tokenizer"],
            hparams["bos_index"],
            hparams["eos_index"],
    )
    
    lm_brain = LM(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    lm_brain.fit(
        lm_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # evaluation
    test_stats = lm_brain.evaluate(
        valid_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
