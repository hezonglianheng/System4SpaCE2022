import torch
import logging
import config
import metrics
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import numpy as np


def test_epoch(test_loader, model):
    losses = 0
    true_tags = []
    pred_tags = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch_data, batch_labels = batch
            batch_mask = batch_data.gt(0)
            outputs = model(batch_data, batch_mask,
                            labels=batch_labels)
            losses += outputs.loss
            batch_outputs = outputs.logits

            batch_outputs = batch_outputs.detach().cpu().numpy()
            batch_labels = batch_labels.cpu().numpy()
            batch_pred_labels = np.argmax(
                batch_outputs,
                axis=-1
            )
            for i in range(len(batch_pred_labels)):
                pred_tags.append(batch_pred_labels[i])
            true_tags.extend(batch_labels)

    assert len(pred_tags) == len(true_tags)
    for i in range(len(true_tags)):
        assert len(pred_tags[i]) == len(true_tags[i])
    f1 = metrics.main(true_tags, pred_tags)
    return f1


def train_epoch(train_loader, model,
                optimizer, scheduler, epoch):
    losses = 0
    model.train()

    for batch in tqdm(train_loader):
        batch_data, batch_labels = batch
        batch_mask = batch_data.gt(0)
        outputs = model(batch_data, batch_mask,
                        labels=batch_labels)
        train_loss = outputs.loss
        losses += train_loss

        model.zero_grad()
        train_loss.backward()
        clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=config.clip_grad
        )

        optimizer.step()
        scheduler.step()

    losses = float(losses) / len(train_loader)
    logging.info(
        'epoch: {}, train loss: {}'.format(epoch, losses)
    )


def main(
        train_loader, test_loader, model,
        optimizer, scheduler
):
    best_f1 = 0
    patience_count = 0
    for epoch in range(1, config.epoch_num+1):
        train_epoch(train_loader, model,
                    optimizer, scheduler, epoch)
        epoch_f1 = test_epoch(test_loader, model)
        logging.info(
            'epoch: {}, step1 f1: {}'.format(epoch, epoch_f1)
        )
        f1_improve = epoch_f1 - best_f1
        if f1_improve > 1e-5:
            best_f1 = epoch_f1
            model.save_pretrained(config.step1_model_dir)
            logging.info('best model saved!')
            if f1_improve < config.patience:
                patience_count += 1
            else:
                patience_count = 0
        else:
            patience_count += 1

        if (epoch == config.epoch_num) or (
            epoch >= config.min_epoch_num and
            patience_count == config.patience_num
        ):
            logging.info(
                'best step 1 f1: {}'.format(best_f1)
            )
            break
    logging.info('training for step1 completed.')
