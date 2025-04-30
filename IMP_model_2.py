import torch

from utils_imp import *
from weighted_ce_loss import *
from typing import Union, Tuple
import model


def complexVar(real: torch.Tensor,
               imag: torch.Tensor,
               dim: Union[int, Tuple[int, ...]] = None,
               keepdim: bool = False) -> torch.Tensor:
    """
    Compute the population variance of a complex-valued tensor given
    separate real and imaginary parts along specified dimension(s).

    Variances defined by:
        Var(Z) = E[|Z - E[Z]|^2]
               = E[(a - μₐ)^2 + (b - μ_b)^2]

    Args:
        real     (torch.Tensor): Real-part tensor.
        imag     (torch.Tensor): Imag-part tensor (same shape as real).
        dim       (int or tuple): Dimension(s) along which to reduce.
        keepdim  (bool): Whether to retain reduced dimensions.

    Returns:
        torch.Tensor: Real-valued variance, reduced over `dim`.
    """
    # 1) Compute means for real and imaginary parts
    mean_real = real.mean(dim=dim, keepdim=True)
    mean_imag = imag.mean(dim=dim, keepdim=True)

    # 2) Compute squared deviations and sum
    sq_dev = (real - mean_real) ** 2 + (imag - mean_imag) ** 2

    # 3) Average over specified dimensions
    var = sq_dev.mean(dim=dim, keepdim=keepdim)
    return var

class InfiniteMixturePrototype2(nn.Module):
    def __init__(self, cfg, args):
        super(InfiniteMixturePrototype2, self).__init__()

        self.encoder = getattr(model, cfg['model']['name'])(
            cifarnet_config=args.cifarnet_config, **cfg['model']['args'])

        learn_sigma_l = 10.0
        #
        # self.in_channels = in_channels
        # self.num_prototypes = num_prototypes

        log_sigma_l = torch.log(torch.FloatTensor([learn_sigma_l]))

        if learn_sigma_l:
            self.log_sigma_l = nn.Parameter(log_sigma_l, requires_grad=True)
        else:
            self.log_sigma_l = Variable(log_sigma_l, requires_grad=True).cuda()

        hid_dim, z_dim = 64, 64

    def _add_cluster(self, nClusters, protos, radii, cluster_type='unlabeled', ex=None):
        """
        Args:
            nClusters: number of clusters
            protos: [B, nClusters, D] cluster protos
            radii: [B, nClusters] cluster radius,
            cluster_type: ['labeled','unlabeled'], the type of cluster we're adding
            ex: the example to add
        Returns:
            updated arguments
        """
        nClusters += 1
        bsize = protos.size()[0]
        dimension = protos.size()[2]

        zero_count = Variable(torch.zeros(bsize, 1)).cuda()

        d_radii = Variable(torch.ones(bsize, 1), requires_grad=False).cuda()

        if cluster_type == 'labeled':
            d_radii = d_radii * torch.exp(self.log_sigma_l)
        else:
            d_radii = d_radii * torch.exp(self.log_sigma_u)

        if ex is None:
            new_proto = self.base_distribution.data.cuda()
        else:
            # new_proto = ex.unsqueeze(0).unsqueeze(0).cuda()
            new_proto = ex.unsqueeze(0).unsqueeze(2).cuda()

        # protos = th.cat([protos, new_proto], dim=1)
        protos = torch.cat([protos, new_proto], dim=2)
        radii = torch.cat([radii, d_radii], dim=1)
        return nClusters, protos, radii

    def estimate_lambda(self, tensor_proto, semi_supervised):
        # estimate lambda by mean of shared sigmas
        # rho = tensor_proto[0].var(dim=0)
        rho = complexVar(tensor_proto[0, 0, ...], tensor_proto[0, 1, ...], dim=0)
        # rho = tensor_proto[0].var(dim=1)
        rho = rho.mean()

        if semi_supervised:
            sigma = (torch.exp(self.log_sigma_l).data[0] + torch.exp(self.log_sigma_u).data[0]) / 2.
        else:
            sigma = torch.exp(self.log_sigma_l).data[0]

        alpha = 0.01
        lamda = -2 * sigma.cpu().numpy() * np.log(alpha) + sigma.cpu().numpy() * np.log(
            1 + rho.detach().cpu().numpy() / sigma.cpu().numpy())

        return np.abs(lamda)

    def _compute_protos(self, h, probs):
        """Compute the prototypes
        Args:
            h: [B, N, D] encoded inputs
            probs: [B, N, nClusters] soft assignment
        Returns:
            cluster protos: [B, nClusters, D]
        """

        h = torch.unsqueeze(h, 2)       # [B, N, 1, D]
        probs = torch.unsqueeze(probs, 3)       # [B, N, nClusters, 1]
        prob_sum = torch.sum(probs, 1)  # [B, nClusters, 1]
        zero_indices = (prob_sum.view(-1) == 0).nonzero()
        if torch.numel(zero_indices) != 0:
            values = torch.masked_select(torch.ones_like(prob_sum), torch.eq(prob_sum, 0.0))
            prob_sum = prob_sum.put_(zero_indices, values)
        # protos = torch.einsum('ijkmn, ijpq->ijmpn', h, probs)
        protos_real = h[:, :, :, 0, :] * probs
        protos_imag = h[:, :, :, 1, :] * probs
        protos = torch.stack([protos_real, protos_imag], dim=2)
        # protos = h*probs    # [B, N, nClusters, D]
        # protos = torch.sum(protos, 1)/prob_sum
        protos = torch.sum(protos, 1)/prob_sum.unsqueeze(1)
        return protos

    def _compute_distances_complex(self, protos, example):
        dist = torch.sum((example[0] - protos[:, 0, ...])**2 + (example[1] - protos[:, 1, ...])**2, dim=2)
        # dist = torch.sum((example - protos)**2, dim=2)
        return dist

    def delete_empty_clusters(self, tensor_proto, prob, radii, targets, eps=1e-3):
        column_sums = torch.sum(prob[0], dim=0).data
        good_protos = column_sums > eps
        idxs = torch.nonzero(good_protos).squeeze()
        return tensor_proto[:, :, idxs, :], radii[:, idxs], targets[idxs]

    def loss(self, logits, targets, labels):
        """Loss function to "or" across the prototypes in the class:
        take the loss for the closest prototype in the class and all negatives.
        inputs:
            logits [B, N, nClusters] of nll probs for each cluster
            targets [B, N] of target clusters
        outputs:
            weighted cross entropy such that we have an "or" function
            across prototypes in the class of each query
        """
        targets = targets.cuda()
        # determine index of closest in-class prototype for each query
        # target_logits = th.ones_like(logits.data) * float('-Inf')
        target_logits = torch.ones_like(logits) * float('-Inf')
        # target_logits[targets] = logits.data[targets]
        target_logits[targets] = logits[targets]
        _, best_targets = torch.max(target_logits, dim=1)
        # mask out everything...
        # weights = th.zeros_like(logits.data)
        weights = torch.zeros_like(logits)
        # ...then include the closest prototype in each class and unlabeled)
        unique_labels = np.unique(labels.cpu().numpy())
        for l in unique_labels:
            class_mask = labels == l
            # class_logits = th.ones_like(logits.data) * float('-Inf')
            class_logits = torch.ones_like(logits) * float('-Inf')
            # batch_idx = torch.arange(class_logits.size(0))
            row, col = class_mask.repeat(logits.size(0), 1).nonzero(as_tuple=True)
            class_logits[row, col] = logits[row, col]
            # class_logits[class_mask.repeat(logits.size(0), 1)] = logits[class_mask].data.view(logits.size(0), -1).squeeze(1)
            _, best_in_class = torch.max(class_logits, dim=1)
            weights[range(0, targets.size(0)), best_in_class] = 1.
        # loss = weighted_loss(logits, Variable(best_targets), Variable(weights))
        loss = weighted_loss(logits, best_targets, weights)
        return loss.mean()

    def forward(self, x, y, x_val=None, y_val=None, train_flag=True):

        y, _ = y.sort()
        y = y.unsqueeze(0)


        h = self.encoder(x)
        if x_val is not None:
            h_val = self.encoder(x_val)
            y_val = y_val.unsqueeze(0)

        # batch = self._process_batch(sample, super_classes=super_classes)
        if train_flag:
            self.nClusters = len(np.unique(y.data.cpu().numpy()))
            self.nInitialClusters = self.nClusters

            # run data through network
            # h_train = self._run_forward(x)
            # create probabilities for points
            _, idx = np.unique(y.squeeze().data.cpu().numpy(), return_inverse=True)
            prob_train = one_hot(y, self.nClusters).cuda()

            # make initial radii for labeled clusters
            bsize = h.size()[0]
            self.radii = Variable(torch.ones(bsize, self.nClusters)).cuda() * torch.exp(self.log_sigma_l)

            self.support_labels = torch.arange(0, self.nClusters).cuda().long()

            # compute initial prototypes from labeled examples
            self.protos = self._compute_protos(h, prob_train)

            # estimate lamda
            # lamda = self.estimate_lambda(self.protos.data, False)
            lamda = self.estimate_lambda(self.protos, False)

            # tensor_proto = self.protos.data
            tensor_proto = self.protos
            # iterate over labeled examples to reassign first
            for i, ex in enumerate(h[0]):
                idxs = torch.nonzero(y.data[0, i] == self.support_labels)[0]
                # distances = self._compute_distances(tensor_proto[:, idxs, :], ex.data)
                distances = self._compute_distances_complex(tensor_proto[:, :, idxs, :], ex.data)
                if (torch.min(distances) > lamda):
                    # self.nClusters, tensor_proto, self.radii = self._add_cluster(self.nClusters, tensor_proto, self.radii,
                    #                                                    cluster_type='labeled', ex=ex.data)
                    self.nClusters, tensor_proto, self.radii = self._add_cluster(self.nClusters, tensor_proto, self.radii,
                                                                       cluster_type='labeled', ex=ex)
                    # self.support_labels = th.cat([self.support_labels, y[0, i].data], dim=0)
                    self.support_labels = torch.cat([self.support_labels, y[0, i].reshape([1])], dim=0)

            # perform partial reassignment based on newly created labeled clusters
            if self.nClusters > self.nInitialClusters:
                # support_targets = y.data[0, :, None] == self.support_labels
                support_targets = y[0, :, None] == self.support_labels
                # prob_train = assign_cluster_radii_limited(Variable(tensor_proto), x, self.radii, support_targets)
                prob_train = assign_cluster_radii_limited(tensor_proto, h, self.radii, support_targets)

            self.protos = Variable(tensor_proto).cuda()
            self.protos = self._compute_protos(h, Variable(prob_train.data, requires_grad=False).cuda())
            self.protos, self.radii, self.support_labels = self.delete_empty_clusters(self.protos, prob_train,
                                                                            self.radii, self.support_labels)

            ###

            logits = compute_logits_radii(self.protos, h_val, self.radii).squeeze()

            # labels = y.data
            labels = y_val
            labels[labels >= self.nInitialClusters] = -1

            support_targets = labels[0, :, None] == self.support_labels
            loss = self.loss(logits, support_targets, self.support_labels)

            # map support predictions back into classes to check accuracy
            _, support_preds = torch.max(logits.data, dim=1)
            y_pred = self.support_labels[support_preds]

            # acc_val = th.eq(y_pred, labels[0]).float().mean()
            acc_val = torch.eq(y_pred, labels[0]).float().sum()

        else:

            # self.nClusters = len(np.unique(y.data.cpu().numpy()))
            # self.nInitialClusters = self.nClusters
            #
            # # run data through network
            # # h_train = self._run_forward(x)
            # # create probabilities for points
            # _, idx = np.unique(y.squeeze().data.cpu().numpy(), return_inverse=True)
            # prob_train = one_hot(y, self.nClusters).cuda()
            #
            # # make initial radii for labeled clusters
            # bsize = x.size()[0]
            # self.radii = Variable(th.ones(bsize, self.nClusters)).cuda() * th.exp(self.log_sigma_l)
            #
            # self.support_labels = th.arange(0, self.nClusters).cuda().long()
            #
            # # compute initial prototypes from labeled examples
            # self.protos = self._compute_protos(x, prob_train)
            #
            # # estimate lamda
            # # lamda = self.estimate_lambda(self.protos.data, False)
            # lamda = self.estimate_lambda(self.protos, False)
            #
            # # tensor_proto = self.protos.data
            # tensor_proto = self.protos
            # # iterate over labeled examples to reassign first
            # for i, ex in enumerate(x[0]):
            #     idxs = th.nonzero(y.data[0, i] == self.support_labels)[0]
            #     # distances = self._compute_distances(tensor_proto[:, idxs, :], ex.data)
            #     distances = self._compute_distances_complex(tensor_proto[:, :, idxs, :], ex.data)
            #     if (th.min(distances) > lamda):
            #         # self.nClusters, tensor_proto, self.radii = self._add_cluster(self.nClusters, tensor_proto, self.radii,
            #         #                                                    cluster_type='labeled', ex=ex.data)
            #         self.nClusters, tensor_proto, self.radii = self._add_cluster(self.nClusters, tensor_proto, self.radii,
            #                                                            cluster_type='labeled', ex=ex)
            #         # self.support_labels = th.cat([self.support_labels, y[0, i].data], dim=0)
            #         self.support_labels = th.cat([self.support_labels, y[0, i].reshape([1])], dim=0)
            #
            # # perform partial reassignment based on newly created labeled clusters
            # if self.nClusters > self.nInitialClusters:
            #     # support_targets = y.data[0, :, None] == self.support_labels
            #     support_targets = y[0, :, None] == self.support_labels
            #     # prob_train = assign_cluster_radii_limited(Variable(tensor_proto), x, self.radii, support_targets)
            #     prob_train = assign_cluster_radii_limited(tensor_proto, x, self.radii, support_targets)
            #
            # self.protos = Variable(tensor_proto).cuda()
            # self.protos = self._compute_protos(x, Variable(prob_train.data, requires_grad=False).cuda())
            # self.protos, self.radii, self.support_labels = self.delete_empty_clusters(self.protos, prob_train,
            #                                                                 self.radii, self.support_labels)

            # h_test = self._run_forward(x)

            logits = compute_logits_radii(self.protos, h, self.radii).squeeze()

            # labels = y.data
            labels = y
            labels[labels >= self.nInitialClusters] = -1

            support_targets = labels[0, :, None] == self.support_labels
            loss = self.loss(logits, support_targets, self.support_labels)

            # map support predictions back into classes to check accuracy
            _, support_preds = torch.max(logits.data, dim=1)
            y_pred = self.support_labels[support_preds]

            acc_val = torch.eq(y_pred, labels[0]).float().sum()

        return loss, acc_val


class InfiniteMixturePrototype(nn.Module):
    def __init__(self, cfg, args):
        super(InfiniteMixturePrototype, self).__init__()

        self.encoder = getattr(model, cfg['model']['name'])(
            cifarnet_config=args.cifarnet_config, **cfg['model']['args'])

        learn_sigma_l = 0.1
        #
        # self.in_channels = in_channels
        # self.num_prototypes = num_prototypes

        log_sigma_l = torch.log(torch.FloatTensor([learn_sigma_l]))

        if learn_sigma_l:
            self.log_sigma_l = nn.Parameter(log_sigma_l, requires_grad=True)
        else:
            self.log_sigma_l = Variable(log_sigma_l, requires_grad=True).cuda()

        hid_dim, z_dim = 64, 64

    def _add_cluster(self, nClusters, protos, radii, cluster_type='unlabeled', ex=None):
        """
        Args:
            nClusters: number of clusters
            protos: [B, nClusters, D] cluster protos
            radii: [B, nClusters] cluster radius,
            cluster_type: ['labeled','unlabeled'], the type of cluster we're adding
            ex: the example to add
        Returns:
            updated arguments
        """
        nClusters += 1
        bsize = protos.size()[0]
        dimension = protos.size()[2]

        zero_count = Variable(torch.zeros(bsize, 1)).cuda()

        d_radii = Variable(torch.ones(bsize, 1), requires_grad=False).cuda()

        if cluster_type == 'labeled':
            d_radii = d_radii * torch.exp(self.log_sigma_l)
        else:
            d_radii = d_radii * torch.exp(self.log_sigma_u)

        if ex is None:
            new_proto = self.base_distribution.data.cuda()
        else:
            # new_proto = ex.unsqueeze(0).unsqueeze(0).cuda()
            new_proto = ex.unsqueeze(0).unsqueeze(2).cuda()

        # protos = th.cat([protos, new_proto], dim=1)
        protos = torch.cat([protos, new_proto], dim=2)
        radii = torch.cat([radii, d_radii], dim=1)
        return nClusters, protos, radii

    def estimate_lambda(self, tensor_proto, semi_supervised):
        # estimate lambda by mean of shared sigmas
        # rho = tensor_proto[0].var(dim=0)
        rho = complexVar(tensor_proto[0, 0, ...], tensor_proto[0, 1, ...], dim=0)
        # rho = tensor_proto[0].var(dim=1)
        rho = rho.mean()

        if semi_supervised:
            sigma = (torch.exp(self.log_sigma_l).data[0] + torch.exp(self.log_sigma_u).data[0]) / 2.
        else:
            sigma = torch.exp(self.log_sigma_l).data[0]

        alpha = 0.01
        lamda = -2 * sigma.cpu().numpy() * np.log(alpha) + sigma.cpu().numpy() * np.log(
            1 + rho.detach().cpu().numpy() / sigma.cpu().numpy())

        return np.abs(lamda)

    def _compute_protos(self, h, probs):
        """Compute the prototypes
        Args:
            h: [B, N, D] encoded inputs
            probs: [B, N, nClusters] soft assignment
        Returns:
            cluster protos: [B, nClusters, D]
        """

        h = torch.unsqueeze(h, 2)       # [B, N, 1, D]
        probs = torch.unsqueeze(probs, 3)       # [B, N, nClusters, 1]
        prob_sum = torch.sum(probs, 1)  # [B, nClusters, 1]
        zero_indices = (prob_sum.view(-1) == 0).nonzero()
        if torch.numel(zero_indices) != 0:
            values = torch.masked_select(torch.ones_like(prob_sum), torch.eq(prob_sum, 0.0))
            prob_sum = prob_sum.put_(zero_indices, values)
        # protos = torch.einsum('ijkmn, ijpq->ijmpn', h, probs)
        protos_real = h[:, :, :, 0, :] * probs
        protos_imag = h[:, :, :, 1, :] * probs
        protos = torch.stack([protos_real, protos_imag], dim=2)
        # protos = h*probs    # [B, N, nClusters, D]
        # protos = torch.sum(protos, 1)/prob_sum
        protos = torch.sum(protos, 1)/prob_sum.unsqueeze(1)
        return protos

    def _compute_distances_complex(self, protos, example):
        dist = torch.sum((example[0] - protos[:, 0, ...])**2 + (example[1] - protos[:, 1, ...])**2, dim=2)
        # dist = torch.sum((example - protos)**2, dim=2)
        return dist

    def delete_empty_clusters(self, tensor_proto, prob, radii, targets, eps=1e-3):
        column_sums = torch.sum(prob[0], dim=0).data
        good_protos = column_sums > eps
        idxs = torch.nonzero(good_protos).squeeze()
        return tensor_proto[:, :, idxs, :], radii[:, idxs], targets[idxs]

    def loss(self, logits, targets, labels):
        """Loss function to "or" across the prototypes in the class:
        take the loss for the closest prototype in the class and all negatives.
        inputs:
            logits [B, N, nClusters] of nll probs for each cluster
            targets [B, N] of target clusters
        outputs:
            weighted cross entropy such that we have an "or" function
            across prototypes in the class of each query
        """
        targets = targets.cuda()
        # determine index of closest in-class prototype for each query
        # target_logits = th.ones_like(logits.data) * float('-Inf')
        target_logits = torch.ones_like(logits) * float('-Inf')
        # target_logits[targets] = logits.data[targets]
        target_logits[targets] = logits[targets]
        _, best_targets = torch.max(target_logits, dim=1)
        # mask out everything...
        # weights = th.zeros_like(logits.data)
        weights = torch.zeros_like(logits)
        # ...then include the closest prototype in each class and unlabeled)
        unique_labels = np.unique(labels.cpu().numpy())
        for l in unique_labels:
            class_mask = labels == l
            # class_logits = th.ones_like(logits.data) * float('-Inf')
            class_logits = torch.ones_like(logits) * float('-Inf')
            # batch_idx = torch.arange(class_logits.size(0))
            row, col = class_mask.repeat(logits.size(0), 1).nonzero(as_tuple=True)
            class_logits[row, col] = logits[row, col]
            # class_logits[class_mask.repeat(logits.size(0), 1)] = logits[class_mask].data.view(logits.size(0), -1).squeeze(1)
            _, best_in_class = torch.max(class_logits, dim=1)
            weights[range(0, targets.size(0)), best_in_class] = 1.
        # loss = weighted_loss(logits, Variable(best_targets), Variable(weights))
        loss = weighted_loss(logits, best_targets, weights)
        return loss.mean()
    @torch.no_grad()
    def dp_means_clustering(self, train_loader):

        all_embs, all_lbls = [], []
        self.encoder.eval()

        for x,y in train_loader:
            z = self.encoder(x.cuda())
            all_embs.append(z.detach().cpu())
            all_lbls.append(y)


        all_embs = torch.cat(all_embs, dim=1)
        all_lbls = torch.cat(all_lbls, dim=0)

        self.all_embs = all_embs.cuda()
        self.all_lbls = all_lbls.unsqueeze(0).cuda()

        self.nClusters = len(np.unique(self.all_lbls.data.cpu().numpy()))
        self.nInitialClusters = self.nClusters

        _, idx = np.unique(self.all_lbls.data.cpu().numpy(), return_inverse=True)
        prob_train = one_hot(self.all_lbls, self.nClusters).cuda()

        bsize = self.all_embs.size()[0]
        self.radii = Variable(torch.ones(bsize, self.nClusters)).cuda() * torch.exp(self.log_sigma_l)

        self.support_labels = torch.arange(0, self.nClusters).cuda().long()

        # compute initial prototypes from labeled examples
        self.protos = self._compute_protos(self.all_embs, prob_train)

        lamda = self.estimate_lambda(self.protos, False)

        # tensor_proto = self.protos.data
        tensor_proto = self.protos
        # iterate over labeled examples to reassign first
        for i, ex in enumerate(self.all_embs[0]):
            idxs = torch.nonzero(self.all_lbls.squeeze().data[i] == self.support_labels)[0]
            # distances = self._compute_distances(tensor_proto[:, idxs, :], ex.data)
            distances = self._compute_distances_complex(tensor_proto[:, :, idxs, :], ex.data)
            if (torch.min(distances) > lamda):
                # self.nClusters, tensor_proto, self.radii = self._add_cluster(self.nClusters, tensor_proto, self.radii,
                #                                                    cluster_type='labeled', ex=ex.data)
                self.nClusters, tensor_proto, self.radii = self._add_cluster(self.nClusters, tensor_proto, self.radii,
                                                                             cluster_type='labeled', ex=ex)
                # self.support_labels = th.cat([self.support_labels, y[0, i].data], dim=0)
                self.support_labels = torch.cat([self.support_labels, self.all_lbls[0, i].reshape([1])], dim=0)

        # perform partial reassignment based on newly created labeled clusters
        if self.nClusters > self.nInitialClusters:
            # support_targets = y.data[0, :, None] == self.support_labels
            support_targets = self.all_lbls[0, :, None] == self.support_labels
            # prob_train = assign_cluster_radii_limited(Variable(tensor_proto), x, self.radii, support_targets)
            prob_train = assign_cluster_radii_limited(tensor_proto, self.all_embs, self.radii, support_targets)

        self.protos = Variable(tensor_proto).cuda()
        self.protos = self._compute_protos(self.all_embs, Variable(prob_train.data, requires_grad=False).cuda())
        self.protos, self.radii, self.support_labels = self.delete_empty_clusters(self.protos, prob_train,
                                                                                  self.radii, self.support_labels)

    def forward(self, x, y, x_val=None, y_val=None, train_flag=True):

        h = self.encoder(x)

        y = y.unsqueeze(0)

        logits = compute_logits_radii(self.protos, h, self.radii).squeeze()

        # labels = y.data
        labels = y
        labels[labels >= self.nInitialClusters] = -1

        support_targets = labels[0, :, None] == self.support_labels
        loss = self.loss(logits, support_targets, self.support_labels)

        # map support predictions back into classes to check accuracy
        _, support_preds = torch.max(logits.data, dim=1)
        y_pred = self.support_labels[support_preds]

        acc_val = torch.eq(y_pred, labels[0]).float().sum()

        return loss, acc_val