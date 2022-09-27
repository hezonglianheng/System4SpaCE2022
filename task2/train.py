import numpy as np
import torch
import config
import logging
from tqdm import tqdm
import torch.nn as nn
import metrics


def train_epoch(
        epoch, train_loader, model, optimizer, scheduler
):
    losses = 0
    model.train()
    for sample in tqdm(train_loader):
        sample_data, sample_labels = sample
        sample_mask = sample_data.gt(0)
        outputs = model(sample_data, sample_mask,
                        labels=sample_labels)
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
    logging.info('epoch: {}, train loss: {}.'.format(
        epoch, losses
    ))


def train(
        train_loader, test_loader, model,
        optimizer, scheduler
):
    best_f1 = 0
    patience_count = 0
    for epoch in range(1, config.epoch_num+1):
        train_epoch(epoch, train_loader, model,
                    optimizer, scheduler)
        test_loss, quota = test(test_loader, model)
        logging.info(
            'epoch: {}, '.format(epoch) +
            'test loss: {}, '.format(test_loss) +
            'wrong type accuracy: {}, '.format(quota[0]) +
            'token marking f1: {}, '.format(quota[1]) +
            'token classification f1: {}'.format(quota[2])
            )
        f1_improve = quota[2] - best_f1
        if f1_improve > 1e-5:
            best_f1 = quota[2]
            model.save_pretrained(config.result_dir)
            logging.info('best model is saved!')
            if f1_improve < config.patience:
                patience_count += 1
            else:
                patience_count = 0
        else:
            patience_count += 1

        # 早停
        if(
            patience_count >= config.patience_num
            and epoch >= config.min_epoch_num
        ) or epoch == config.epoch_num:
            logging.info(
                'best token classification f1: {}'.format(best_f1)
            )
            break
    logging.info('training finished!')


def test(test_loader, model):
    true_tags = []
    predict_tags = []
    qids = []
    losses = 0
    model.eval()

    with torch.no_grad():
        for sample in tqdm(test_loader):
            sample_data, sample_labels, sample_qids = sample
            sample_mask = sample_data.gt(0)
            outputs = model(sample_data, sample_mask,
                            labels=sample_labels)
            losses += outputs.loss
            batch_outputs = outputs.logits

            # size: (batch_size, seq_length, num_labels)
            batch_outputs = batch_outputs.detach().cpu().numpy()
            # size: (batch_size, seq_length)
            batch_tags = sample_labels.cpu().numpy()
            # size: (batch_size, seq_length)
            batch_pred_tags = np.argmax(batch_outputs, axis=-1)
            for i in range(len(batch_pred_tags)):
                predict_tags.append(batch_pred_tags[i])
            true_tags.extend(batch_tags)
            qids.extend(sample_qids)

    assert len(true_tags) == len(predict_tags) == len(qids)
    for i in range(len(true_tags)):
        assert len(true_tags[i]) == len(predict_tags[i])

    quota = metrics.main(predict_tags, qids)
    losses = float(losses) / len(test_loader)
    return losses, quota
