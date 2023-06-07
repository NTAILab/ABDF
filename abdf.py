from bosk.block.auto import auto_block
from bosk.block.base import BaseBlock, BlockInputData, TransformOutputData, BlockT
from bosk.block.meta import BlockMeta, InputSlotMeta, OutputSlotMeta, BlockExecutionProperties, DynamicBlockMetaStub
from bosk.pipeline.builder import FunctionalPipelineBuilder
from bosk.stages import Stages
from bosk.data import CPUData
from bosk.executor.sklearn_interface import BoskPipelineClassifier
from sklearn.ensemble._forest import ForestClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from typing import Type
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from bosk.pipeline import BasePipeline
from bosk.painter.topological import TopologicalPainter
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numba
from numba.types import bool_
import sys

@numba.njit
def find_close_elems(x_1, x_2):
    eps = 10 ** -6
    res = np.zeros(x_1.shape[0], bool_)
    for i in range(x_1.shape[0]):
        for j in range(x_2.shape[0]):
            if np.sum(np.abs(x_1[i] - x_2[j])) < eps:
                res[i] = True
                break
    return res

# forest which gives not only the y estim
# but the x according to the leaf close elements

class EstimForest(BaseBlock):
    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, forest_cls: Type[ForestClassifier], folds_num: int, name = None, **forest_kw):
        self.forest = forest_cls(n_jobs=-1, **forest_kw)
        if name is not None:
            self.name = name
        fold_inputs = [InputSlotMeta(f'X_fold_{i}', Stages(True, False)) for i in range(folds_num)]
        self.meta = BlockMeta(
            inputs=[
                InputSlotMeta('X', Stages()),
                InputSlotMeta('y', Stages(True, False)),
            ] + fold_inputs,
            outputs=[
                OutputSlotMeta(f'probas_{i}') for i in range(folds_num)
            ] + [
                OutputSlotMeta(f'x_estim_{i}') for i in range(folds_num)
            ]
        )
        self.folds_num = folds_num
        super().__init__()

    def fit(self: BlockT, inputs: BlockInputData) -> BlockT:
        x_np = inputs['X'].data
        y_np = inputs['y'].data
        self.c_num = np.max(y_np).astype(np.int0) + 1
        x_fold_list = [inputs[f'X_fold_{i}'].data.copy() for i in range(self.folds_num)]
        self.forest.fit(x_np, y_np)
        idxes = self.forest.apply(x_np)
        # tree -> leaf -> x list
        idx_to_x = [defaultdict(list) for _ in range(self.forest.n_estimators)]
        idx_to_y = [defaultdict(list) for _ in range(self.forest.n_estimators)]
        one_hot = np.eye(self.c_num)
        for i in range(len(idxes)):
            for j, l_i in enumerate(idxes[i]):
                idx_to_x[j][l_i].append(x_np[i])
                idx_to_y[j][l_i].append(one_hot[y_np[i]])
        # tree -> leaf -> x array FILTERED FOR EACH FOLD
        # x array is list lenght of folds with MEANS
        x_map = []
        y_map = []
        for i in range(self.forest.n_estimators):
            cur_dict_x = dict()
            cur_dict_y = dict()
            x_map.append(cur_dict_x)
            y_map.append(cur_dict_y)
            for leaf_idx in idx_to_x[i]:
                x_arr = np.array(idx_to_x[i][leaf_idx])
                y_arr = np.array(idx_to_y[i][leaf_idx])
                total_x_list = []
                total_y_list = []
                for j in range(self.folds_num):
                    # filter_mask = np.max(np.sum(
                    #         np.abs(x_arr[None, ...] - x_fold_list[j][:, None, :]), axis=-1
                    #     ) < eps, axis=0)
                    filter_mask = find_close_elems(x_arr, x_fold_list[j])
                    if np.max(filter_mask) == False:
                        total_x_list.append(None)
                        total_y_list.append(None)
                        continue
                    total_x_list.append(np.mean(x_arr[filter_mask], axis=0))
                    total_y_list.append(np.mean(y_arr[filter_mask], axis=0))
                cur_dict_x[leaf_idx] = total_x_list
                cur_dict_y[leaf_idx] = total_y_list
        self.x_map = x_map
        self.y_map = y_map
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        x_np = inputs['X'].data
        leaf_idx = self.forest.apply(x_np)
        output_dict = dict()
        for f_i in range(self.folds_num):
            x_estim = np.zeros_like(x_np)
            probas = np.zeros((x_np.shape[0], self.c_num))
            for i in range(len(x_np)):
                trees_used = 0
                for j in range(self.forest.n_estimators):
                    x_filtered = self.x_map[j][leaf_idx[i, j]][f_i]
                    if x_filtered is None:
                        continue
                    y_filtered = self.y_map[j][leaf_idx[i, j]][f_i]
                    x_estim[i] += x_filtered
                    probas[i] += y_filtered
                    trees_used += 1
                if trees_used == 0:
                    postfix = '' if self.name is None else f' in {self.name}'
                    print('None of the trees were used during prediction' + postfix, file=sys.stderr)
                else:
                    x_estim[i] /= trees_used
                    probas[i] /= trees_used
            output_dict[f'probas_{f_i}'] = CPUData(probas)
            output_dict[f'x_estim_{f_i}'] = CPUData(x_estim)
        return output_dict


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class AttentionHead(torch.nn.Module):
    def __init__(self, name="", f_optimize=True):
        super().__init__()
        self.device = torch.device('cpu')
        self.gamma = torch.ones(1, requires_grad=True, device=self.device)
        self.optimizer = torch.optim.AdamW([self.gamma], lr=2e-1)
        self.epochs_num = 300
        self.batch_size = 128
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.name = name
        self.f_optim = f_optimize

    def _get_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.astype(np.float32)).to(self.device)

    def forward(self, x: torch.Tensor, keys, values) -> torch.Tensor:
        x_mat = x[:, None, :] # (n, 1, m)
        metric = torch.sum((x_mat - keys) ** 2, dim=-1) # (n, f)
        att = torch.softmax(-metric * self.gamma, dim=-1) # (n, f)
        pred = torch.sum(att[..., None] * values, dim=1) # (n, c)
        return pred

    def fit(self, X: np.ndarray, X_hat: np.ndarray, y_hat: np.ndarray, y_gt: np.ndarray) -> 'AttentionHead':
        if not self.f_optim:
            return self
        q = self._get_tensor(X) # (n, m)
        k = self._get_tensor(X_hat) # (n, f, m)
        v = self._get_tensor(y_hat) # (n, f, c)
        labels = torch.from_numpy(y_gt).long().to(self.device) # (n)
        dataset = TensorDataset(q, k, v, labels)
        data_loader = DataLoader(dataset, self.batch_size, True)
        self.train()
        # summary = SummaryWriter(comment=self.name)
        for e in range(self.epochs_num):
            cumul_loss = 0
            # best_loss = float('inf')
            for q_b, k_b, v_b, labels_b in data_loader:
                self.optimizer.zero_grad()
                pred = self(q_b, k_b, v_b)
                loss = self.loss_fn(pred, labels_b) / len(dataset)
                loss.backward()
                self.optimizer.step()
                cumul_loss += loss.item()
            # mean_loss = cumul_loss # / len(data_loader)
            # summary.add_scalar('Train Cross Entropy', mean_loss, e)
            # summary.add_scalar('Gamma', self.gamma.item(), e)
        # summary.close()
        return self

    def transform(self, X: np.ndarray, X_hat: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            q, k, v = map(self._get_tensor, (X, X_hat, y_hat))
            return self(q, k, v).numpy()


@auto_block()
class SliceBlock():
    def __init__(self) -> None:
        super().__init__()

    def fit(self, idx):
        self.idx = idx
        return self

    def transform(self, X):
        return X[self.idx]

    
@auto_block()
class StdScalerBlock():
    def __init__(self, with_mean=True, with_std=True) -> None:
        self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        super().__init__()
        
    def fit(self, X) -> 'StdScalerBlock':
        self.scaler.fit(X)
        return self
    
    def transform(self, X) -> np.ndarray:
        return self.scaler.transform(X)


def get_adf_class_pipeline(layers_num: int, forests_num: int, heads_num: int, trees: int) -> BasePipeline:
    b = FunctionalPipelineBuilder()
    FOREST_WRAPPER = lambda name, cls, **kw: b.new(EstimForest, folds_num=heads_num, name=name, forest_cls=cls, n_estimators=trees)(**kw)
    forest_cls_list = [RandomForestClassifier if i % 2 == 0 else ExtraTreesClassifier for i in range(forests_num)]
    ATTENTION_WRAPPER = lambda name, **kw: b.new(AttentionHead, name)(**kw)
    SLICE_WRAPPER = lambda **kw: b.new(SliceBlock)(**kw)
    SCALER_WRAPPER = lambda **kw: b.new(StdScalerBlock)(**kw)
    x_input = b.Input()()
    y_input = b.TargetInput()()
    for i in range(layers_num):
        if i > 0:
            x = output
        else:
            scaler = SCALER_WRAPPER(X=x_input)
            x = scaler
        fold_gen = b.CVTrainIndices(size=heads_num, random_state=None)(
            X=x, y=y_input)
        inp_fold_names = [f'X_fold_{i}' for i in range(heads_num)]
        folds_x = [SLICE_WRAPPER(X=x, idx=fold_gen[str(i)]) for i in range(heads_num)]
        forest_folds_map = dict(zip(inp_fold_names, folds_x))
        forest_list = [FOREST_WRAPPER(f'AF_{j}', forest_cls_list[j], X=x, y=y_input, **forest_folds_map) for j in range(forests_num)]
        slots_names = [f'forest_{i}' for i in range(forests_num)]
        X_hat_list = []
        y_hat_list = []
        for j in range(heads_num):
            output_x_slots = [forest[f'x_estim_{j}'] for forest in forest_list]
            stack_dict_x = dict(zip(slots_names, output_x_slots))
            X_hat_list.append(b.Stack(slots_names, axis=1)(**stack_dict_x))
            output_y_slots = [forest[f'probas_{j}'] for forest in forest_list]
            stack_dict_y = dict(zip(slots_names, output_y_slots))
            y_hat_list.append(b.Stack(slots_names, axis=1)(**stack_dict_y))
        attention_map = dict()
        for j in range(heads_num):
            head_name = f'head_{j}'
            attention_map[head_name] = ATTENTION_WRAPPER(head_name, 
                                                    X=x, X_hat=X_hat_list[j],
                                                    y_hat=y_hat_list[j], y_gt=y_input)
        output_att_slots = list(attention_map.keys())
        output_map = attention_map.copy()
        if i < layers_num - 1:
            output_att_slots.append('X')
            output_map['X'] = x
            output = b.Concat(output_att_slots, axis=1)(**output_map)
        if i == layers_num - 1:
            stack = b.Stack(output_att_slots, axis=1)(**output_map)
            probas = b.Average(axis=1)(X=stack)
            labels = b.Argmax(axis=-1)(X=probas)
    pipeline = b.build(
        inputs={'X': x_input, 'y': y_input},
        outputs={'probas': probas, 'labels': labels}
    )
    # TopologicalPainter(figure_dpi=150, graph_levels_sep=3).from_pipeline(pipeline).render('pipeline.png')
    return BoskPipelineClassifier(pipeline, outputs_map={'pred': 'labels', 'proba': 'probas'})
    

def adf_test():
    pipeline = get_adf_class_pipeline(2, 2, 3, 100)
    X, y = make_moons(200, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    pipeline.fit(X_train, y_train)
    result = pipeline.predict(X_test)
    print('acc:', accuracy_score(y_test, result))
    print('roc-auc:', roc_auc_score(y_test, result))

if __name__ == '__main__':
    adf_test()
