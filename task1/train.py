import numpy as np
import torch
import config
import logging
from tqdm import tqdm
import torch.nn as nn
import metrics


def train(train_loader, test_loader, model, optimizer,
          scheduler):
    best_acc = 0
    patience_count = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer,
                    scheduler, epoch)
        test_acc, test_loss = evaluate(test_loader,
                                       model)
        logging.info('Epoch: {}, accuracy: {}'.format(epoch,
                                                      test_acc))
        acc_improve = test_acc - best_acc
        if acc_improve > 1e-5:
            best_acc = test_acc
            model.save_pretrained(config.result_dir)
            logging.info('Best model is saved at the directory named experiment.')
            if acc_improve < config.patience:
                patience_count += 1
            else:
                patience_count = 0
        else:
            patience_count += 1

        # early stopping
        if (patience_count >= config.patience_num and
            epoch >= config.min_epoch_num) or \
                epoch == config.epoch_num:
            logging.info('Best accuracy: {}'.format(best_acc))
            break

    logging.info('Training finished.')


def train_epoch(train_loader, model, optimizer, scheduler,
                epoch):
    losses = 0
    model.train()
    for sample in tqdm(train_loader):
        sample_data, sample_label = sample
        sample_musk = sample_data.gt(0)
        outputs = model(sample_data, sample_musk,
                        labels=sample_label)
        train_loss = outputs.loss
        losses += train_loss

        model.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm(
            parameters=model.parameters(),
            max_norm=config.clip_grad)

        optimizer.step()
        scheduler.step()

    losses = float(losses) / len(train_loader)
    logging.info('Epoch: {}, train loss: {}'.format(epoch,
                                                    losses))


def evaluate(test_loader, model, mode='test'):
    true_tags = []
    predict_tags = []
    losses = 0
    model.eval()

    with torch.no_grad():
        for sample in tqdm(test_loader):
            sample_data, sample_label = sample
            sample_musk = sample_data.gt(0)
            outputs = model(sample_data, sample_musk,
                            labels=sample_label)
            losses += outputs.loss
            batch_output = outputs.logits

            batch_output = batch_output.detach().cpu().numpy()  # (batch_size, num_labels)
            batch_tags = sample_label.to('cpu').numpy()

            predict_tags.extend(np.argmax(batch_output, axis=-1))
            true_tags.extend(batch_tags)

    assert len(true_tags) == len(predict_tags)

    accuracy = metrics.get_accuracy(true_tags, predict_tags,
                                    mode)
    losses = float(losses) / len(test_loader)
    return accuracy, losses
