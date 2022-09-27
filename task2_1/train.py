import torch
import config
from tqdm import tqdm
import torch.nn as nn
import logging
import numpy as np
import metrics


def train_epoch(
        epoch, train_loader, model,
        optimizer, scheduler
):
    losses = 0
    model.train()

    for batch in tqdm(train_loader):
        batch_sentence, batch_labels = batch
        batch_mask = batch_sentence.gt(0)
        outputs = model(
            batch_sentence,
            batch_mask,
            labels=batch_labels
        )
        train_loss = outputs.loss
        losses += train_loss

        model.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=config.clip_grad
        )

        optimizer.step()
        scheduler.step()

    losses = float(losses) / len(train_loader)
    logging.info(
        'epoch: {}, train loss: {}'.format(epoch, losses)
    )


def test_epoch(test_loader, model, _type):
    losses = 0
    true_labels = []
    pred_labels = []
    qids = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch_sentence, batch_labels, batch_qids = batch
            batch_mask = batch_sentence.gt(0)
            outputs = model(batch_sentence, batch_mask,
                            labels=batch_labels)
            losses += outputs.loss
            batch_outputs = outputs.logits

            # size: (batch_size, seq_length, num_labels)
            batch_outputs = batch_outputs.detach().cpu().numpy()
            # size: (batch_size, seq_length)
            batch_tags = batch_labels.cpu().numpy()
            # size: (batch_size, seq_length)
            batch_pred_tags = np.argmax(
                batch_outputs,
                axis=-1
            )
            for i in range(len(batch_pred_tags)):
                pred_labels.append(batch_pred_tags[i])
            true_labels.extend(batch_tags)
            qids.extend(batch_qids)

    assert len(true_labels) == len(pred_labels)
    for i in range(len(true_labels)):
        assert len(true_labels[i]) == len(pred_labels[i])

    quota = metrics.main(pred_labels, qids, _type)
    losses = losses / len(test_loader)
    return losses, quota


def main(
        train_loader, test_loader, model,
        optimizer, scheduler, _type
):
    best_f1 = 0.0
    patience_count = 0
    for epoch in range(1, config.epoch_num+1):
        train_epoch(
            epoch, train_loader, model,
            optimizer, scheduler
        )
        test_loss, test_quota = test_epoch(
            test_loader, model, _type
        )
        logging.info(
            'type: {}\n'.format(_type) +
            'epoch: {}\n'.format(epoch) +
            'test loss: {}\n'.format(test_quota) +
            'wrong type accuracy: {}\n'.format(test_quota[0]) +
            'token marking f1: {}\n'.format(test_quota[1]) +
            'token classification f1: {}'.format(test_quota[2])
        )
        f1_improve = test_quota[-1] - best_f1
        if f1_improve > 1e-5:
            best_f1 = test_quota[-1]
            if _type == 'A':
                model.save_pretrained(config.model4a_dir)
            elif _type == 'B':
                model.save_pretrained(config.model4b_dir)
            elif _type == 'C':
                model.save_pretrained(config.model4c_dir)
            else:
                raise AttributeError(
                    'the type: {}, is wrong.'.format(_type)
                )
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
                'best text classification f1: {}'.format(best_f1)
            )
            break

    logging.info('train finished.')
