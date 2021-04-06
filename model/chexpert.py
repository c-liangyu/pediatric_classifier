from numpy.lib.stride_tricks import broadcast_to
import torch.nn as nn
import numpy as np
import time
import cv2
import os
import shutil
from torch.nn.functional import threshold
import tqdm
import pickle
import torch
import wandb
from metrics import AUC_ROC
from data.utils import transform
from model.models import save_dense_backbone, load_dense_backbone, save_resnet_backbone, load_resnet_backbone, Ensemble, AverageMeter
from model.utils import get_models, get_str, tensor2numpy, get_optimizer, load_ckp, lrfn, get_metrics
from confidence_interval import boostrap_ci


class CheXpert_model():
    id_obs = [2, 5, 6, 8, 10]
    # id_obs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    id_leaf = [2, 4, 5, 6, 7, 8]
    id_parent = [0, 1, 3, 9, 10, 11, 12, 13]
    M = np.array([[0, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0]])

    def __init__(self, cfg, loss_func, metrics=None):
        """CheXpert class contains all functions used for training and testing our models

        Args:
            cfg (dict): configuration file.
            loss_func (torch.nn.Module): loss function of the model.
            metrics (dict, optional): metrics use to evaluate model performance. Defaults to None.
        """
        self.cfg = cfg
        if self.cfg.type == 'pediatric':
            self.cfg.num_classes = 13*[1]
        elif self.cfg.type == 'sub_pediatric':
            self.cfg.num_classes = 10*[1]
        elif self.cfg.type == 'chexmic':
            self.cfg.num_classes = 14*[1]
        else:
            self.cfg.num_classes = [1]
            # self.cfg.disease_classes = ['No finding']
        if self.cfg.device == 'cpu':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:"+str(self.cfg.device))
        self.model, self.childs_cut = get_models(self.cfg)
        self.loss_func = loss_func
        if metrics is not None:
            self.metrics = metrics
            self.metrics['loss'] = self.loss_func
        else:
            self.metrics = {'loss': self.loss_func}
        self.optimizer = get_optimizer(self.model.parameters(), self.cfg)
        # self.model.to(self.device)
        if cfg.distributed:
            self.model.cuda()
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.cfg.device], broadcast_buffers=False)
        elif cfg.parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        else:
            self.model.cuda()

        self.thresh_val = torch.Tensor(
            [0.5]*len(self.cfg.num_classes)).float().cuda()

    def freeze_backbone(self):
        """Freeze model backbone
        """
        ct = 0
        if self.cfg.distributed:
            model_part = self.model.module
        elif self.cfg.parallel:
            model_part = self.model.module
        else:
            model_part = self.model

        loop = model_part.children() if len(list(model_part.children())
                                            ) > 1 else list(model_part.children())[0].children()

        for child in loop:
            ct += 1
            if ct < self.childs_cut:
                for param in child.parameters():
                    param.requires_grad = False

    def save_backbone(self, ckp_path):
        if self.cfg.distributed:
            model_part = self.model.module
        elif self.cfg.parallel:
            model_part = self.model.module
        else:
            model_part = self.model

        if self.cfg.backbone == 'dense' or self.cfg.backbone == 'densenet':
            save_dense_backbone(model_part, ckp_path)
        elif self.cfg.backbone == 'resnet':
            save_resnet_backbone(model_part, ckp_path)

    def load_backbone(self, ckp_path, strict=True):
        if self.cfg.distributed:
            model_part = self.model.module
        elif self.cfg.parallel:
            model_part = self.model.module
        else:
            model_part = self.model

        if self.cfg.backbone == 'dense' or self.cfg.backbone == 'densenet':
            load_dense_backbone(model_part, ckp_path, self.device, strict)
        elif self.cfg.backbone == 'resnet':
            load_resnet_backbone(model_part, ckp_path, self.device, strict)

    def load_ckp(self, ckp_path, strict=True):
        """Load checkpoint

        Args:
            ckp_path (str): path to checkpoint

        Returns:
            int, int: current epoch, current iteration
        """
        return load_ckp(self.model, ckp_path, self.device, self.cfg.distributed, self.cfg.parallel, strict)

    def save_ckp(self, ckp_path, epoch, iter):
        """Save checkpoint

        Args:
            ckp_path (str): path to saved checkpoint
            epoch (int): current epoch
            iter (int): current iteration
        """
        if os.path.exists(os.path.dirname(ckp_path)):
            torch.save(
                {'epoch': epoch+1,
                 'iter': iter+1,
                 'state_dict': self.model.module.state_dict() if (self.cfg.distributed or self.cfg.parallel) else self.model.state_dict()},
                ckp_path
            )
        else:
            print("Save path not exist!!!")

    def predict(self, image):
        """Run prediction

        Args:
            image (torch.Tensor): images to predict. Shape (batch size, C, H, W)

        Returns:
            torch.Tensor: model prediction
        """
        torch.set_grad_enabled(False)
        self.model.eval()
        with torch.no_grad() as tng:
            preds = self.model(image)
            if not isinstance(self.model, Ensemble):
                preds = nn.Sigmoid()(preds)

        return preds

    def predict_from_file(self, image_file):
        """Run prediction from image path

        Args:
            image_file (str): image path

        Returns:
            numpy array: model prediction in numpy array type
        """
        image_gray = cv2.imread(image_file, 0)
        image = transform(image_gray, self.cfg)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)

        return tensor2numpy(nn.Sigmoid()(self.predict(image)))

    def predict_loader(self, loader, ensemble=False, cal_loss=False):
        """Run prediction on a given dataloader.

        Args:
            loader (torch.utils.data.Dataloader): a dataloader
            ensemble (bool, optional): use FAEL for prediction. Defaults to False.

        Returns:
            torch.Tensor, torch.Tensor: prediction, labels
        """
        preds_stack = []
        labels_stack = []
        running_loss = []
        ova_len = loader.dataset._num_image
        loop = tqdm.tqdm(enumerate(loader), total=len(loader))
        img_ids = []
        for i, data in loop:
            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            if self.cfg.tta:
                # imgs = torch.cat(imgs, dim=0)
                list_imgs = [imgs[:, j] for j in range(imgs.shape[1])]
                imgs = torch.cat(list_imgs, dim=0)
                preds = self.predict(imgs)
                batch_len = labels.shape[0]
                list_preds = [preds[batch_len*j:batch_len *
                                    (j+1)] for j in range(len(list_imgs))]
                preds = torch.stack(list_preds, dim=0).mean(dim=0)
            else:
                preds = self.predict(imgs)
            if self.cfg.conditional_training and (self.cfg.type == 'pediatric' or self.cfg.type == 'chexmic'):
                preds = preds[:, loader.dataset.id_leaf]
                labels = labels[:, loader.dataset.id_leaf]
            iter_len = imgs.size()[0]
            if ensemble:
                preds = torch.mm(preds, self.ensemble_weights)
                labels = labels[:, loader.dataset.id_obs]
            preds_stack.append(preds)
            labels_stack.append(labels)
            if cal_loss:
                running_loss.append(self.metrics['loss'](
                    preds, labels).item()*iter_len/ova_len)
        preds_stack = torch.cat(preds_stack, 0)
        labels_stack = torch.cat(labels_stack, 0)
        running_loss = sum(running_loss)
        return preds_stack, labels_stack, running_loss

    def train(self, train_loader, val_loader, epochs=10, iter_log=100, use_lr_sch=False, resume=False, ckp_dir='./experiment/checkpoint',
              eval_metric='loss'):
        """Run training

        Args:
            train_loader (torch.utils.data.Dataloader): dataloader use for training
            val_loader (torch.utils.data.Dataloader): dataloader use for validation
            epochs (int, optional): number of training epochs. Defaults to 120.
            iter_log (int, optional): logging iteration. Defaults to 100.
            use_lr_sch (bool, optional): use learning rate scheduler. Defaults to False.
            resume (bool, optional): resume training process. Defaults to False.
            ckp_dir (str, optional): path to checkpoint directory. Defaults to './experiment/checkpoint'.
            writer (torch.utils.tensorboard.SummaryWriter, optional): tensorboard summery writer. Defaults to None.
            eval_metric (str, optional): name of metric for validation. Defaults to 'loss'.
        """
        # wandb.init(name=self.cfg.log_dir,
        #            project='Pediatric Multi-label Classifier',
        #            entity='dolphin')
        if use_lr_sch:
            lr_sch = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lrfn)
            # lr_sch = torch.optim.lr_scheduler.StepLR(
            #     self.optimizer, int(epochs*1/2), self.cfg.lr/10)
            lr_hist = []
        else:
            lr_sch = None
        best_metric = 0.0

        if self.cfg.conditional_training and (self.cfg.type == 'pediatric' or self.cfg.type == 'chexmic'):
            self.thresh_val = self.thresh_val[train_loader.dataset.id_leaf]
        if os.path.exists(ckp_dir) != True:
            os.mkdir(ckp_dir)
        if resume:
            epoch_resume, iter_resume = self.load_ckp(
                os.path.join(ckp_dir, 'latest.ckpt'))
        else:
            epoch_resume = 1
            iter_resume = 0
        scaler = None
        if self.cfg.mix_precision:
            print('Train with mix precision!')
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(epoch_resume-1, epochs):
            start = time.time()
            running_loss = AverageMeter()
            n_iter = len(train_loader)
            torch.set_grad_enabled(True)
            self.model.train()
            batch_weights = (1/iter_log)*np.ones(n_iter)
            step_per_epoch = n_iter // iter_log
            if n_iter % iter_log:
                step_per_epoch += 1
                batch_weights[-(n_iter % iter_log):] = 1 / (n_iter % iter_log)
                iter_per_step = iter_log * \
                    np.ones(step_per_epoch, dtype=np.int16)
                iter_per_step[-1] = n_iter % iter_log
            else:
                iter_per_step = iter_log * \
                    np.ones(step_per_epoch, dtype=np.int16)
            i = 0
            for step in range(step_per_epoch):
                loop = tqdm.tqdm(
                    range(iter_per_step[step]), total=iter_per_step[step])
                iter_loader = iter(train_loader)
                for iteration in loop:
                    data = next(iter_loader)
                    imgs, labels = data[0].to(
                        self.device), data[1].to(self.device)
                    # r = np.random.rand(1)
                    # use_cutmix = self.cfg.beta > 0 and r < self.cfg.cutmix_prob
                    # if use_cutmix:
                    #     # generate mixed sample
                    #     lam = np.random.beta(self.cfg.beta, self.cfg.beta)
                    #     rand_index = torch.randperm(imgs.size()[0]).cuda()
                    #     target_a = labels
                    #     target_b = labels[rand_index]
                    #     bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                    #     imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    #     # adjust lambda to exactly match pixel ratio
                    #     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))

                    if self.cfg.mix_precision:
                        with torch.cuda.amp.autocast():
                            preds = self.model(imgs)
                            if self.cfg.conditional_training and (self.cfg.type == 'pediatric' or self.cfg.type == 'chexmic'):
                                preds = preds[:, train_loader.dataset.id_leaf]
                                labels = labels[:,
                                                train_loader.dataset.id_leaf]
                            loss = self.metrics['loss'](preds, labels)
                            # if use_cutmix:
                            #     loss = self.metrics['loss'](preds, target_a) * lam + self.metrics['loss'](preds, target_b) * (1. - lam)
                            # else:
                            #     loss = self.metrics['loss'](preds, labels)
                    else:
                        preds = self.model(imgs)
                        if self.cfg.conditional_training and (self.cfg.type == 'pediatric' or self.cfg.type == 'chexmic'):
                            preds = preds[:, train_loader.dataset.id_leaf]
                            labels = labels[:, train_loader.dataset.id_leaf]
                        loss = self.metrics['loss'](preds, labels)
                        # if use_cutmix:
                        #     loss = self.metrics['loss'](preds, target_a) * lam + self.metrics['loss'](preds, target_b) * (1. - lam)
                        # else:
                        #     loss = self.metrics['loss'](preds, labels)
                    preds = nn.Sigmoid()(preds)
                    running_loss.update(loss.item(), imgs.shape[0])
                    self.optimizer.zero_grad()
                    if self.cfg.mix_precision:
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    i += 1

                # if wandb:
                #     wandb.log(
                #         {'loss/train': running_metrics['loss']}, step=(epoch*n_iter)+(i+1))
                s = "Epoch [{}/{}] Iter [{}/{}]:\n".format(
                    epoch+1, epochs, i+1, n_iter)
                s += "{}_{} {:.3f}\n".format('train', 'loss', running_loss.avg)
                print(s)
                running_metrics_test = self.test(
                    val_loader, False)
                torch.set_grad_enabled(True)
                self.model.train()
                s = get_str(running_metrics_test, 'val', s)
                # if wandb:
                #     for key in running_metrics_test.keys():
                #         if key != 'loss':
                #             for j, disease_class in enumerate(np.array(train_loader.dataset.disease_classes)[train_loader.dataset.id_leaf]):
                #                 wandb.log(
                #                     {key+'/'+disease_class: running_metrics_test[key][j]}, step=(epoch*n_iter)+(i+1))
                #         else:
                #             wandb.log(
                #                 {'loss/val': running_metrics_test['loss']}, step=(epoch*n_iter)+(i+1))
                if self.cfg.type != 'chexmic':
                    metric_eval = running_metrics_test[eval_metric]
                else:
                    metric_eval = running_metrics_test[eval_metric][self.id_obs]
                s = s[:-1] + "- mean_"+eval_metric + \
                    " {:.3f}".format(metric_eval.mean())
                self.save_ckp(os.path.join(
                    ckp_dir, 'latest.ckpt'), epoch, i)
                running_loss.reset()
                end = time.time()
                s += " ({:.1f}s)".format(end-start)
                print(s)
                if metric_eval.mean() > best_metric:
                    best_metric = metric_eval.mean()
                    shutil.copyfile(os.path.join(ckp_dir, 'latest.ckpt'), os.path.join(
                        ckp_dir, 'best.ckpt'))
                    print('new checkpoint saved!')
                start = time.time()
            if lr_sch is not None:
                lr_sch.step()
                print('current lr: {:.4f}'.format(lr_sch.get_lr()[0]))
        if lr_sch is not None:
            return lr_hist
        else:
            return None

    def test(self, loader, ensemble=False, get_ci=False, n_boostrap=10000):
        """Run testing

        Args:
            loader (torch.utils.data.Dataloader): dataloader use for testing
            ensemble (bool, optional): use FAEL for prediction. Defaults to False.

        Returns:
            dict: metrics use to evaluate model performance.
        """
        # if self.cfg.conditional_training and (self.cfg.type == 'pediatric' or self.cfg.type == 'chexmic'):
        #     self.thresh_val = self.thresh_val[loader.dataset.id_leaf]
        preds_stack, labels_stack, running_loss = self.predict_loader(
            loader, ensemble, cal_loss=True)

        running_metrics = get_metrics(
            preds_stack, labels_stack, self.metrics, self.thresh_val)
        running_metrics['loss'] = running_loss
        if get_ci:
            ci_dict = self.eval_CI(labels_stack, preds_stack, n_boostrap)
            return running_metrics, ci_dict

        return running_metrics

    def thresholding(self, loader):
        auc_opt = AUC_ROC()
        preds, labels, _ = self.predict_loader(loader)
        thresh_val = auc_opt(preds, labels, thresholding=True)
        print(f"List optimal threshold {thresh_val}")
        self.thresh_val = torch.Tensor(thresh_val).float().cuda()

    def eval_CI(self, labels, preds, n_boostrap=1000, csv_path=None):

        return boostrap_ci(labels, preds, self.metrics, n_boostrap, self.thresh_val, csv_path)

    def FAEL(self, loader, val_loader, type='basic', init_lr=0.01, log_iter=100, steps=20, lambda1=0.1, lambda2=2):
        """Run fully associative ensemble learning (FAEL)

        Args:
            loader (torch.utils.data.Dataloader): dataloader use for training FAEL model
            val_loader (torch.utils.data.Dataloader): dataloader use for validating FAEL model
            type (str, optional): regularization type (basic/binary constraint). Defaults to 'basic'.
            init_lr (float, optional): initial learning rate. Defaults to 0.01.
            log_step (int, optional): logging step. Defaults to 100.
            steps (int, optional): total steps. Defaults to 20.
            lambda1 (float, optional): l2 regularization parameter. Defaults to 0.1.
            lambda2 (int, optional): binary constraint parameter. Defaults to 2.

        Returns:
            dict: metrics use to evaluate model performance.
        """
        n_class = len(self.cfg.num_classes)
        w = np.random.rand(n_class, len(self.id_obs))
        iden_matrix = np.diag(np.ones(n_class))
        lr = init_lr
        start = time.time()
        for i, data in enumerate(tqdm.tqdm(loader, total=log_iter*steps)):
            imgs, labels = data[0].to(self.device), data[1]
            preds = self.predict(imgs)
            preds = tensor2numpy(preds)
            labels = tensor2numpy(labels)
            labels = labels[:, self.id_obs]
            if type == 'basic':
                grad = (preds.T.dot(preds) + lambda1*iden_matrix).dot(w) - \
                    preds.T.dot(labels)
            elif type == 'b_constraint':
                grad = (preds.T.dot(preds) + lambda1*iden_matrix + lambda2 *
                        self.M.T.dot(self.M)).dot(w) + - preds.T.dot(labels)
            else:
                raise Exception("Not support this type!!!")
            w -= lr*grad
            if (i+1) % log_iter == 0:
                # print(w)
                end = time.time()
                print('iter {:d} time takes: {:.3f}s'.format(i+1, end-start))
                start = time.time()
                if (i+1)//log_iter == steps:
                    break
        if self.cfg.mix_precision:
            self.ensemble_weights = torch.from_numpy(
                w).type(torch.HalfTensor).to(self.device)
        else:
            self.ensemble_weights = torch.from_numpy(w).float().to(self.device)
        print('Done Essemble!!!')
        metrics = self.test(val_loader, ensemble=True)

        return metrics

    def save_FAEL_weight(self, path):
        """save FAEL weight

        Args:
            path (str): path to saved weight.
        """
        with open(path, 'wb') as f:
            pickle.dump(tensor2numpy(self.ensemble_weights.float()), f)

    def load_FAEL_weight(self, path):
        """load FAEL weight

        Args:
            path (str): path to saved weight
            mix_precision (bool, optional): use mix precision for prediction. Defaults to False.
        """
        with open(path, 'rb') as f:
            w = pickle.load(f)
        if self.cfg.mix_precision:
            self.ensemble_weights = torch.from_numpy(
                w).type(torch.HalfTensor).to(self.device)
        else:
            self.ensemble_weights = torch.from_numpy(w).float().to(self.device)