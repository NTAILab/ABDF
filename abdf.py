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
from time import time

@numba.njit
def index_getter(x_tensor, y_tensor, forest_idx):
    x_estim = np.zeros((x_tensor.shape[0], forest_idx.shape[0], x_tensor.shape[-1]), dtype=np.float64)
    probas = np.zeros((x_tensor.shape[0], forest_idx.shape[0], y_tensor.shape[-1]), dtype=np.float64)
    for id_point in range(forest_idx.shape[0]):
        for id_fold in range(x_tensor.shape[0]):
            trees_used = 0
            for id_tree in range(forest_idx.shape[1]):
                x_estim[id_fold, id_point, :] += x_tensor[id_fold, id_tree, forest_idx[id_point, id_tree]].copy()
                probas[id_fold, id_point, :] += y_tensor[id_fold, id_tree, forest_idx[id_point, id_tree]].copy()
                if np.any(y_tensor[id_fold, id_tree, forest_idx[id_point, id_tree]] > 0):
                    trees_used += 1
            if trees_used > 0:
                x_estim[id_fold, id_point] /= trees_used
                probas[id_fold, id_point] /= trees_used
    return x_estim, probas

@numba.njit
def build_tensors(X, y, fold_idxes, apply_res, max_nodes, c_num):
    folds_num = len(fold_idxes)
    trees_num = apply_res.shape[1]
    forest_x_tensor = np.zeros((folds_num, trees_num, max_nodes, X.shape[1]), dtype=np.float32)
    forest_y_tensor = np.zeros((folds_num, trees_num, max_nodes, c_num), dtype=np.float32)
    counter = np.empty((trees_num, max_nodes), dtype=np.int32)
    one_hot = np.eye(c_num)
    for id_fold in range(folds_num):
        cur_fold_idx = fold_idxes[id_fold]
        x_fold = X[cur_fold_idx]
        y_fold = y[cur_fold_idx]
        cur_apply = apply_res[cur_fold_idx]
        counter.fill(0)
        for id_point in range(cur_apply.shape[0]):
            idx_leafs = cur_apply[id_point]
            cur_x = x_fold[id_point]
            cur_y = one_hot[y_fold[id_point]]
            for id_tree in range(trees_num):
                id_leaf = idx_leafs[id_tree]
                forest_x_tensor[id_fold, id_tree, id_leaf] += cur_x
                forest_y_tensor[id_fold, id_tree, id_leaf] += cur_y
                counter[id_tree, id_leaf] += 1
        for id_tree in range(trees_num):
            for id_leaf in range(max_nodes):
                if counter[id_tree, id_leaf] > 0:
                    forest_x_tensor[id_fold, id_tree, id_leaf] /= counter[id_tree, id_leaf]
                    forest_y_tensor[id_fold, id_tree, id_leaf] /= counter[id_tree, id_leaf]
    return forest_x_tensor, forest_y_tensor
        

# forest which gives not only the y estim
# but the x according to the leaf close elements

class EstimForest(BaseBlock):
    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, forest_cls: Type[ForestClassifier], folds_num: int, name = None, **forest_kw):
        self.forest = forest_cls(n_jobs=-1, **forest_kw)
        if name is not None:
            self.name = name
        fold_inputs = [InputSlotMeta(f'fold_idx_{i}', Stages(True, False)) for i in range(folds_num)]
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
        self.x_tensor = None
        self.y_tensor = None
        super().__init__()

    def fit(self: BlockT, inputs: BlockInputData) -> BlockT:
        x_np = inputs['X'].data
        y_np = inputs['y'].data
        c_num = y_np.max().astype(np.int0) + 1
        idx_folds = tuple(inputs[f'fold_idx_{i}'].data.copy() for i in range(self.folds_num))
        self.forest.fit(x_np, y_np)
        max_nodes = 0
        for tree in self.forest.estimators_:
            max_nodes = max(max_nodes, tree.tree_.node_count)
        apply_res = self.forest.apply(x_np)
        self.x_tensor, self.y_tensor = build_tensors(x_np, y_np, idx_folds, apply_res, max_nodes, c_num)
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        x_np = inputs['X'].data
        leaf_idx = self.forest.apply(x_np)
        x_output, probas_output = index_getter(self.x_tensor, self.y_tensor, leaf_idx)
        output_dict = dict()
        for i_fold in range(self.folds_num):
            output_dict[f'x_estim_{i_fold}'] = CPUData(x_output[i_fold].astype(np.float64))
            output_dict[f'probas_{i_fold}'] = CPUData(probas_output[i_fold].astype(np.float64))
        return output_dict


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class AttentionHead(torch.nn.Module):
    def __init__(self, name="", f_optimize=True):
        super().__init__()
        self.device = torch.device('cpu')
        self.gamma = torch.ones(1, requires_grad=True, device=self.device)
        self.optimizer = torch.optim.AdamW([self.gamma], lr=2e-1)
        self.epochs_num = 100
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
        inp_fold_names = [f'fold_idx_{i}' for i in range(heads_num)]
        inp_fold_idxes = [fold_gen[str(i)] for i in range(heads_num)]
        forest_folds_map = dict(zip(inp_fold_names, inp_fold_idxes))
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
    # pipeline.set_random_state(123)
    # TopologicalPainter(figure_dpi=150, graph_levels_sep=3).from_pipeline(pipeline).render('pipeline.png')
    return BoskPipelineClassifier(pipeline, outputs_map={'pred': 'labels', 'proba': 'probas'})
    

def adf_test():
    pipeline = get_adf_class_pipeline(2, 2, 10, 100)
    X, y = make_moons(5000, noise=0.1, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    time_stamp = time()
    pipeline.fit(X_train, y_train)
    fit_time = time() - time_stamp
    print('fit time:', fit_time)
    time_stamp = time()
    result = pipeline.predict(X_test)
    pred_time = time() - time_stamp
    print('pred time:', pred_time)
    print('acc:', accuracy_score(y_test, result))
    print('roc-auc:', roc_auc_score(y_test, result))

if __name__ == '__main__':
    adf_test()
