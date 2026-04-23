"""Episode samplers for node-, link-, and graph-classification tasks.

Each sampler returns a dict of tensors shaped for :class:`MochiModel` to consume.
The :class:`MultiTaskEpisodeSampler` wraps one sampler per dataset and draws an
episode from a weighted mixture of task types.
"""

from __future__ import annotations

import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════════
# Per-task samplers
# ══════════════════════════════════════════════════════════════════════════


class _ClassBalancedSampler:
    """Shared K-shot / Q-query sampling logic for NC and GC."""

    def __init__(self, y, task: str):
        self.y = y
        self.task = task
        self.classes = torch.unique(y, sorted=True).tolist()
        self.class_idx = {c: (y == c).nonzero(as_tuple=True)[0] for c in self.classes}

    def sample(self, n_way=None, k_shot=5, q_query=15):
        cs = (self.classes
              if n_way is None or n_way >= len(self.classes)
              else np.random.choice(self.classes, n_way, replace=False).tolist())

        s_idx, q_idx, s_labels, q_labels = [], [], [], []
        for new_c, orig_c in enumerate(cs):
            pool = self.class_idx[orig_c]
            perm = torch.randperm(len(pool))
            need = k_shot + q_query
            if need > len(pool):
                perm = torch.cat([perm, torch.randint(len(pool), (need - len(pool),))])
            s_idx.append(pool[perm[:k_shot]])
            q_idx.append(pool[perm[k_shot:k_shot + q_query]])
            s_labels += [new_c] * k_shot
            q_labels += [new_c] * q_query

        s_idx = torch.cat(s_idx)
        q_idx = torch.cat(q_idx)
        perm = torch.randperm(s_idx.size(0))
        s_idx = s_idx[perm]
        s_labels_t = torch.tensor(s_labels, dtype=torch.long)[perm]

        return dict(
            support_idx=s_idx,
            support_y=s_labels_t,
            query_idx=q_idx,
            query_y=torch.tensor(q_labels, dtype=torch.long),
            n_way=len(cs),
            task=self.task,
        )


class NodeEpisodeSampler(_ClassBalancedSampler):
    """N-way K-shot node classification."""
    def __init__(self, y):
        super().__init__(y, task="nc")


class GraphEpisodeSampler(_ClassBalancedSampler):
    """N-way K-shot graph classification."""
    def __init__(self, y):
        super().__init__(y, task="gc")


class LinkEpisodeSampler:
    """Binary support/query edge classification (pos vs. neg).

    The sampler holds the edge set of the training adjacency (``handler.trn_mat``)
    and draws ``k_shot`` positives + ``k_shot`` negatives for support, then the
    same for query.
    """

    def __init__(self, handler):
        self.rows = handler.trn_mat.row.astype(np.int64)
        self.cols = handler.trn_mat.col.astype(np.int64)
        self.node_num = handler.trn_mat.shape[0]
        self.edge_set = set(zip(self.rows.tolist(), self.cols.tolist()))

    def _sample_negatives(self, n):
        neg_src, neg_dst = [], []
        while len(neg_src) < n:
            s = np.random.randint(0, self.node_num)
            d = np.random.randint(0, self.node_num)
            if s != d and (s, d) not in self.edge_set:
                neg_src.append(s)
                neg_dst.append(d)
        return np.array(neg_src, dtype=np.int64), np.array(neg_dst, dtype=np.int64)

    def sample(self, k_shot=32, q_query=64):
        total_pos = k_shot + q_query
        perm = np.random.permutation(len(self.rows))[:total_pos]
        if len(perm) < total_pos:
            extra = np.random.randint(0, len(self.rows), total_pos - len(perm))
            perm = np.concatenate([perm, extra])
        pos_src = self.rows[perm]
        pos_dst = self.cols[perm]

        neg_src, neg_dst = self._sample_negatives(k_shot + q_query)

        sup_src = np.concatenate([pos_src[:k_shot], neg_src[:k_shot]])
        sup_dst = np.concatenate([pos_dst[:k_shot], neg_dst[:k_shot]])
        sup_y = np.concatenate([np.ones(k_shot), np.zeros(k_shot)])

        qry_src = np.concatenate([pos_src[k_shot:], neg_src[k_shot:]])
        qry_dst = np.concatenate([pos_dst[k_shot:], neg_dst[k_shot:]])
        qry_y = np.concatenate([np.ones(q_query), np.zeros(q_query)])

        sup_perm = np.random.permutation(len(sup_y))
        qry_perm = np.random.permutation(len(qry_y))

        return dict(
            support_src=torch.tensor(sup_src[sup_perm], dtype=torch.long),
            support_dst=torch.tensor(sup_dst[sup_perm], dtype=torch.long),
            support_y=torch.tensor(sup_y[sup_perm], dtype=torch.long),
            query_src=torch.tensor(qry_src[qry_perm], dtype=torch.long),
            query_dst=torch.tensor(qry_dst[qry_perm], dtype=torch.long),
            query_y=torch.tensor(qry_y[qry_perm], dtype=torch.long),
            n_way=2,
            task="lp",
        )


# ══════════════════════════════════════════════════════════════════════════
# Mixed-task sampler
# ══════════════════════════════════════════════════════════════════════════


class MultiTaskEpisodeSampler:
    """Uniformly (or weighted) samples NC, LP, or GC episodes.

    ``task_weights`` is a ``"nc:lp:gc"`` string — ``"1:1:1"`` by default. Missing
    task types are dropped automatically.
    """

    def __init__(self, lp_handlers, nc_labels_list, gc_labels_list,
                 task_weights: str = "1:1:1"):
        self.lp_samplers = [LinkEpisodeSampler(h) for h in lp_handlers] if lp_handlers else []
        self.nc_samplers = [NodeEpisodeSampler(y) for y, _ in nc_labels_list] if nc_labels_list else []
        self.gc_samplers = [GraphEpisodeSampler(y) for y, _ in gc_labels_list] if gc_labels_list else []

        w = [float(x) for x in task_weights.split(":")]
        available, weights = [], []
        if self.nc_samplers:
            available.append("nc"); weights.append(w[0] if len(w) > 0 else 1.0)
        if self.lp_samplers:
            available.append("lp"); weights.append(w[1] if len(w) > 1 else 1.0)
        if self.gc_samplers:
            available.append("gc"); weights.append(w[2] if len(w) > 2 else 1.0)

        total = sum(weights) or 1.0
        self.task_types = available
        self.task_probs = [x / total for x in weights]

    def sample(self, k_shot=16, q_query=32, max_classes=64):
        task_type = np.random.choice(self.task_types, p=self.task_probs)
        if task_type == "nc":
            idx = np.random.randint(len(self.nc_samplers))
            s = self.nc_samplers[idx]
            ep = s.sample(n_way=min(max_classes, len(s.classes)), k_shot=k_shot, q_query=q_query)
            return ep, task_type, idx
        if task_type == "lp":
            idx = np.random.randint(len(self.lp_samplers))
            ep = self.lp_samplers[idx].sample(k_shot=k_shot, q_query=q_query)
            return ep, task_type, idx
        # gc
        idx = np.random.randint(len(self.gc_samplers))
        s = self.gc_samplers[idx]
        ep = s.sample(n_way=min(max_classes, len(s.classes)), k_shot=k_shot, q_query=q_query)
        return ep, task_type, idx
