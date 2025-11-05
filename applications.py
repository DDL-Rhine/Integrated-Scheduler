import collections
import glob
import math
import os
import pandas
import numpy as np 
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator
from goodput import fit_perf_params
from llama_goodput import llama_fit_perf_params
from llama_pp_goodput import pp_llama_fit_perf_params

def get(name):
    return APPLICATIONS[name]


def memoize(f):
    memo = {}
    def helper(*x):
        if x not in memo:
            memo[x] = f(*x)
        return memo[x]
    return helper


class Application(object):
    def __init__(self, trace_dir,
                 init_batch_size=None, max_batch_size=None,
                 min_local_bsz=None, max_local_bsz=None,
                 max_epochs=None, target_metric=None, max_stats_bsz=None):
        self.name = os.path.basename(trace_dir)
        if not os.path.exists(trace_dir): 
            trace_dir = trace_dir.replace('A800', '')
        
        validation = {}
        for path in glob.glob(os.path.join(trace_dir, "validation-*.csv")):
            batch_size = int(path.split("-")[-1].split(".")[0])
            validation[batch_size] = pandas.read_csv(path)
        self.validation = collections.OrderedDict(sorted(validation.items()))
        self.placements = \
            pandas.read_csv(os.path.join(trace_dir, "placements.csv"))
        self.placements["num_nodes"] = \
            self.placements.placement.apply(lambda p: len(str(p)))
        self.placements["num_replicas"] = \
            self.placements.placement.apply(lambda p: sum(map(int, str(p))))
        
        # rewrite_placement_file = os.path.join(trace_dir, 'placements-rewrite.csv')
        # if os.path.exists(rewrite_placement_file): 
        #     self.rewrite_placements = pandas.read_csv(os.path.join(trace_dir, "placements-rewrite.csv"))
        #     self.rewrite_placements["num_nodes"] = \
        #         self.rewrite_placements.placement.apply(lambda p: len(str(p)))
        #     self.placements["num_replicas"] = \
        #         self.rewrite_placements.placement.apply(lambda p: sum(map(int, str(p))))
        
        # cache_placement_file = os.path.join(trace_dir, 'placements-cache.csv')
        # if os.path.exists(cache_placement_file): 
        #     self.cache_placements = pandas.read_csv(os.path.join(trace_dir, "placements-cache.csv"))
        #     self.cache_placements["num_nodes"] = \
        #         self.cache_placements.placement.apply(lambda p: len(str(p)))
        #     self.cache_placements["num_replicas"] = \
        #         self.cache_placements.placement.apply(lambda p: sum(map(int, str(p))))
        
        accelerate_placement_file = os.path.join(trace_dir, 'placements-accelerate.csv')
        if os.path.exists(accelerate_placement_file): 
            self.accelerate_placements = pandas.read_csv(os.path.join(trace_dir, "placements-accelerate.csv"))
            self.accelerate_placements["num_nodes"] = \
                self.accelerate_placements.placement.apply(lambda p: len(str(p)))
            self.accelerate_placements["num_replicas"] = \
                self.accelerate_placements.placement.apply(lambda p: sum(map(int, str(p))))

        int8_placement_file = os.path.join(trace_dir, 'placements-int8.csv')
        if os.path.exists(int8_placement_file): 
            self.int8_placements = pandas.read_csv(os.path.join(trace_dir, "placements-int8.csv"))
            self.int8_placements["num_nodes"] = \
                self.int8_placements.placement.apply(lambda p: len(str(p)))
            self.int8_placements["num_replicas"] = \
                self.int8_placements.placement.apply(lambda p: sum(map(int, str(p))))

        scalability_file = os.path.join(trace_dir, "scalability.csv")
        if os.path.exists(scalability_file): 
            self.scalability = \
                pandas.read_csv(os.path.join(trace_dir, "scalability.csv"))
            self.max_allowable_nodes = -1 
        else: 
            self.scalability = None 
            self.max_allowable_nodes = 4 
        
        accelerate_scalability_file = os.path.join(trace_dir, "scalability-accelerate.csv")
        if os.path.exists(accelerate_scalability_file): 
            self.accelerate_scalability = \
                pandas.read_csv(accelerate_scalability_file)
            self.max_allowable_nodes = -1 

        int8_scalability_file = os.path.join(trace_dir, "scalability-int8.csv")
        if os.path.exists(int8_scalability_file): 
            self.int8_scalability = \
                pandas.read_csv(int8_scalability_file)
            self.max_allowable_nodes = -1 

        # gpu_utili_file = os.path.join(trace_dir, "gpu_util.csv")
        gpu_utili_file = os.path.join(trace_dir, "gpu_util.csv")
        if os.path.exists(gpu_utili_file):
            self.gpu_util = pandas.read_csv(gpu_utili_file)
            
        
        self.init_batch_size = init_batch_size or min(self.validation)
        self.max_batch_size = max_batch_size or max(self.validation)
        self.min_local_bsz = min_local_bsz or self.placements.local_bsz.min()
        if 'squad-llama' in self.name: 
            self.min_local_bsz = 32
        self.max_local_bsz = max_local_bsz or self.placements[self.placements.frozen_layer==0].local_bsz.max()
        self.max_stats_bsz = max_stats_bsz or self.placements.local_bsz.max() 
        assert self.max_batch_size >= self.min_local_bsz
        self.max_epochs = max_epochs or min(map(len, self.validation.values()))
        self.target_metric = target_metric
        # app_path = os.path.join(trace_dir, "app_info.npy")
        # import pdb; pdb.set_trace() 
        self.app_info = np.load(os.path.join(trace_dir, "app_info.npy"), allow_pickle=True).tolist() 
        self._flop_info = self.app_info['flop_info']
        self._param_info = self.app_info['param_info']
        self._layer_num = self.app_info['layer_num'] 
        self._max_frozen_layer = self.placements.frozen_layer.max() 
        if os.path.exists(os.path.join(trace_dir, "freeze_strategy.npy")): 
            self.recommend_freeze_strategies = np.load(os.path.join(trace_dir, "freeze_strategy.npy"), allow_pickle=True).tolist() 

        # memory limit 
        self._max_host_memory = 34089730048 
        self.memory_info = None 
        if os.path.exists(os.path.join(trace_dir, "memory.csv")): 
            self.memory_info = pandas.read_csv(os.path.join(trace_dir, "memory.csv")) 
            self.layer_batch_parteo = self.fetch_layer_batch_pareto() 

        assert len(self._flop_info) == len(self._param_info) 
        # self.perf_params = self.prior_perf_params() 
        self.grad_params = self.prior_grad_params() 
    
    def prior_grad_params(self, ): 
        return self.get_grad_stats(self.init_batch_size, 0, 0, True) 
        

    def prior_perf_params(self, LargeThr=False): 
        if 'squad-llama' in self.name: 
            placement_list = [(4,), (4, 4)]
            frozen_list = [0, 56, 112, 140]
        else: 
            placement_list = [(1,), (2,), (4,), (4,4)]
            frozen_list = [0, 2, 4, 8]
            
        num_nodes, num_replicas, local_bsz, flop_ratios, param_ratios = list(), list(), list(), list(), list() 
        compute_time, step_time = list(), list() 
        for placement in placement_list: 
            for bsz in [self.min_local_bsz, self.max_local_bsz]: 
                for frozen_layer in frozen_list: 
                    num_nodes.append(len(placement)) 
                    num_replicas.append(sum(placement)) 
                    local_bsz.append(bsz)
                    time_info = self.get_throughput(placement, bsz, frozen_layer)
                    compute_time.append(time_info[0] - time_info[1])
                    step_time.append(time_info[0])
                    
                    # import pdb; pdb.set_trace() 
                    flop_ratios.append(self._flop_info[frozen_layer])
                    param_ratios.append(self._param_info[frozen_layer])
        
        if '-pp' in self.name and LargeThr: 
            perf_params = pp_llama_fit_perf_params(num_nodes, num_replicas, local_bsz, compute_time, step_time, flop_ratios=flop_ratios, param_ratios=param_ratios, init_params=None)
        elif 'llama' in self.name and LargeThr: 
            perf_params = llama_fit_perf_params(num_nodes, num_replicas, local_bsz, compute_time, step_time, flop_ratios=flop_ratios, param_ratios=param_ratios, init_params=None)
        else: 
            perf_params = fit_perf_params(num_nodes, num_replicas, local_bsz, compute_time, step_time, flop_ratios=flop_ratios, param_ratios=param_ratios, init_params=None)
        self.max_profiled_replicas = 2
        return perf_params

    def fetch_layer_batch_pareto(self): 
        assert self.memory_info is not None 
        pareto_list = list()
        froze_layer_list = np.unique(self.placements.frozen_layer.values).tolist()
        for frozen_layer in range(self._max_frozen_layer + 1): 
            if frozen_layer in froze_layer_list: 
                info = self.memory_info[self.memory_info.frozen_layer == frozen_layer]
                info = info[info.gpu_memory <= self._max_host_memory]
                if len(info) == 0: 
                    pareto_list.append(self.placements[self.placements.frozen_layer==frozen_layer].local_bsz.min())
                else: 
                    pareto_list.append(min(info.local_bsz.max(), self.placements[self.placements.frozen_layer==frozen_layer].local_bsz.max()))

            else: 
                pareto_list.append(pareto_list[-1])
        # print(self.name, pareto_list)
        return pareto_list


    def update_max_host_memory(self, memory_scale): 
        # self._max_host_memory = int(34089730048 * memory_scale) 
        self._max_host_memory = int(85174583296 * memory_scale)
        assert self.memory_info is not None, 'memory information should provide'
        if self.memory_info is not None: 
            info = self.memory_info[self.memory_info.frozen_layer == 0]
            info = info[info.gpu_memory <= self._max_host_memory]
            if len(info) == 0: 
                max_local_bsz = self.min_local_bsz 
            else: 
                max_local_bsz = min(info.local_bsz.max(), self.max_local_bsz)
            self.max_local_bsz = min(max_local_bsz, self.max_stats_bsz)
            self.layer_batch_parteo = self.fetch_layer_batch_pareto() 
            
        

    def get_max_frozen_layer(self, ): 
        return self._max_frozen_layer 

    def get_flop_info(self,):
        return self._flop_info 

    def get_param_info(self,):
        return self._param_info

    def get_layer_num(self, ):
        return self._layer_num
    
    def _validated_batch_sizes(self, batch_size):
        # Find the lower-bound and upper-bound batch sizes (may be the same).
        lower_bsz = upper_bsz = None
        for bsz in self.validation:
            if bsz <= batch_size:
                lower_bsz = bsz
            if bsz >= batch_size:
                upper_bsz = bsz
                break
        assert lower_bsz is not None and upper_bsz is not None, \
               "{} {}".format(batch_size, list(self.validation))
        assert lower_bsz <= batch_size <= upper_bsz
        return lower_bsz, upper_bsz

    def get_configurations(self, lo_util=0.5, hi_util=0.8):
        # Assuming a cluster of 16 nodes each with 4 GPUs.
        ret = []
        base_jct = None
        base_batch_size = None
        if self.name in ['squad-llama', 'squad-llama-pp', 'squad-llama-3B', 'squad-llama-3B-pp', 'sst2-llama-7B', 'sst2-llama-7B-pp']: 
            num_replicas_list = (4, 8, 12, 16, 24, 32)
        else: 
            num_replicas_list = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48)
            
        for num_replicas in num_replicas_list: 
            if num_replicas * self.min_local_bsz > self.max_batch_size:
                break
            placement = ()
            while sum(placement) < num_replicas:
                placement = (*placement, min(num_replicas - sum(placement), 4))
            best_jct = None
            best_batch_size = None
            for batch_size, valid in self.validation.items(): 
                local_bsz = math.ceil(batch_size / sum(placement) - 1e-8)
                if local_bsz < self.min_local_bsz:
                    continue
                accum_steps = math.ceil(local_bsz / self.max_local_bsz - 1e-8) - 1
                #if sum(placement) == 1 and batch_size > self.init_batch_size:
                #    accum_steps = max(1, accum_steps)
                atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
                epoch = self.get_completion_epoch(batch_size)
                step_time, sync_time = self.get_throughput(placement, atomic_bsz, 0)
                step_time += accum_steps * (step_time - sync_time)
                jct = valid.iteration[epoch] * step_time
                if best_jct is None or jct < best_jct:
                    best_jct = jct
                    best_batch_size = batch_size
            if num_replicas == 1:
                base_jct = best_jct
                base_batch_size = best_batch_size
            elif 'llama' in self.name and num_replicas == 4: 
                base_jct = best_jct
                base_batch_size = best_batch_size
            elif best_jct < 12 * 3600 and \
                    lo_util < base_jct / best_jct / num_replicas < hi_util:
                ret.append((num_replicas, best_batch_size, best_jct))
        if not ret:
            ret.append((1, base_batch_size, base_jct))
        return ret

    def get_best_batch_size(self, num_replicas):
        # Assuming a cluster of 16 nodes each with 4 GPUs.
        if num_replicas * self.min_local_bsz > self.max_batch_size:
            return None
        placement = ()
        while sum(placement) < num_replicas:
            placement = (*placement, min(num_replicas - sum(placement), 4))
        best_jct = None
        best_batch_size = None
        for batch_size, valid in self.validation.items():
            local_bsz = math.ceil(batch_size / sum(placement))
            if local_bsz < self.min_local_bsz:
                continue
            if local_bsz > self.max_local_bsz:
                break
            epoch = self.get_completion_epoch(batch_size)
            step_time, _ = self.get_throughput(placement, local_bsz)
            jct = valid.iteration[epoch] * step_time
            if best_jct is None or jct < best_jct:
                best_jct = jct
                best_batch_size = batch_size
        return best_batch_size

    def get_epoch(self, progress):
        return max(df.progress.searchsorted(progress, "right")
                   for df in self.validation.values())

    @memoize
    def get_progress(self, epoch):
        if epoch == 0:
            return 0.0
        return min(df.progress[epoch - 1] for df in self.validation.values()) # confuse, why min 

    @memoize
    def get_completion_epoch(self, batch_size):
        if self.target_metric is None:
            return self.max_epochs - 1
        best_metric = None
        for epoch in range(self.max_epochs):
            next_metric = self.get_best_metric(batch_size, epoch)
            if best_metric is not None:
                sign = self.target_metric - best_metric
                if sign * (self.target_metric - next_metric) <= 0:
                    # Opposite signs, crossed target metric.
                    return epoch
        return epoch

    @memoize
    def get_iteration(self, batch_size, epoch):
        # Returns the number of iterations after completing a given epoch.
        lower_bsz, upper_bsz = self._validated_batch_sizes(batch_size)
        lower_it = self.validation[lower_bsz].iteration[epoch]
        upper_it = self.validation[upper_bsz].iteration[epoch]
        if lower_bsz == upper_bsz:
            assert lower_it == upper_it
            return lower_it
        # Linear interpolation between lower_bsz and upper_bsz.
        return ((batch_size - lower_bsz) * upper_it +
                (upper_bsz - batch_size) * lower_it) / (upper_bsz - lower_bsz)

    @memoize
    def get_best_metric(self, batch_size, epoch):
        # Returns the best observed validation metric before a given epoch.
        if epoch == 0:
            return None
        lower_bsz, upper_bsz = self._validated_batch_sizes(batch_size)
        if (next(iter(self.validation.values())).metric.values[0] <
            next(iter(self.validation.values())).metric.values[-1]):
            # Validation metric increases.
            lower_val = self.validation[lower_bsz].metric[:epoch].max()
            upper_val = self.validation[upper_bsz].metric[:epoch].max()
        else:
            lower_val = self.validation[lower_bsz].metric[:epoch].min()
            upper_val = self.validation[upper_bsz].metric[:epoch].min()
        if lower_bsz == upper_bsz:
            assert lower_val == upper_val
            return lower_val
        # Linear interpolation between lower_bsz and upper_bsz.
        return ((batch_size - lower_bsz) * upper_val +
                (upper_bsz - batch_size) * lower_val) / (upper_bsz - lower_bsz)


    def get_recommend_frozen_layer(self, epoch): 
        if epoch == self.max_epochs: 
            return self.recommend_freeze_strategies[epoch - 1]
        return self.recommend_freeze_strategies[epoch]

    @memoize
    def get_grad_stats(self, batch_size, epoch, frozen_layer, verbose):
        # Returns the gradient sqr and var estimates during a given epoch.
        lower_bsz, upper_bsz = self._validated_batch_sizes(batch_size)
        # lower_sqr = np.cumsum([getattr(self.validation[lower_bsz], 'layer_{}_grad_sqr'.format(layer))[epoch] for layer in range(self._layer_num - 1, frozen_layer - 1, -1)])[::-1]
        # upper_sqr = np.cumsum([getattr(self.validation[upper_bsz], 'layer_{}_grad_sqr'.format(layer))[epoch] for layer in range(self._layer_num - 1, frozen_layer - 1, -1)])[::-1]
        # lower_var = np.cumsum([getattr(self.validation[lower_bsz], 'layer_{}_grad_var'.format(layer))[epoch] for layer in range(self._layer_num - 1, frozen_layer - 1, -1)])[::-1]
        # upper_var = np.cumsum([getattr(self.validation[upper_bsz], 'layer_{}_grad_var'.format(layer))[epoch] for layer in range(self._layer_num - 1, frozen_layer - 1, -1)])[::-1]
        lower_sqr = np.array([getattr(self.validation[lower_bsz], 'layer_{}_grad_sqr'.format(layer))[epoch] for layer in range(frozen_layer, self._layer_num, 1)])
        upper_sqr = np.array([getattr(self.validation[upper_bsz], 'layer_{}_grad_sqr'.format(layer))[epoch] for layer in range(frozen_layer, self._layer_num, 1)])
        lower_var = np.array([getattr(self.validation[lower_bsz], 'layer_{}_grad_var'.format(layer))[epoch] for layer in range(frozen_layer, self._layer_num, 1)])
        upper_var = np.array([getattr(self.validation[upper_bsz], 'layer_{}_grad_var'.format(layer))[epoch] for layer in range(frozen_layer, self._layer_num, 1)])
        # import pdb; pdb.set_trace() 

        if lower_bsz == upper_bsz:
            # assert lower_sqr == upper_sqr and lower_var == upper_var
            if verbose: 
                return lower_sqr, lower_var
            else:
                return sum(lower_sqr), sum(lower_var)
        # Linear interpolation between lower_bsz and upper_bsz.
        sqr = ((batch_size - lower_bsz) * upper_sqr +
               (upper_bsz - batch_size) * lower_sqr) / (upper_bsz - lower_bsz)
        var = ((batch_size - lower_bsz) * upper_var +
               (upper_bsz - batch_size) * lower_var) / (upper_bsz - lower_bsz)
        if verbose: 
            return sqr, var
        else:
            return sum(lower_sqr), sum(lower_var)
    

    @memoize
    def get_gpu_memory(self, local_bsz, frozen_layer): 
        df = self.memory_info 
        ys = ["gpu_memory"]
        if frozen_layer in df.frozen_layer.values: 
            df = df[df.frozen_layer == frozen_layer]
            interpolator = interp1d(df.local_bsz.values, df[ys].values, axis=0, fill_value='extrapolate')
            ret = interpolator(local_bsz)
            assert sum(ret) == sum(ret)
            return ret 
        if local_bsz in df.local_bsz.values: 
            df = df[df.local_bsz == local_bsz]
            interpolator = interp1d(df.frozen_layer.values, df[ys].values, axis=0, fill_value='extrapolate')
            ret = interpolator(frozen_layer)
            assert sum(ret) == sum(ret)
            return ret
        
        fill_value = min(df[df.frozen_layer > frozen_layer].gpu_memory.values + [0])
        interpolator = LinearNDInterpolator(df[['frozen_layer', 'local_bsz']].values, df[ys].values, fill_value=fill_value)
        ret = interpolator([frozen_layer, local_bsz])[0]
        assert sum(ret) == sum(ret)
        if ret < 0: 
            import pdb; pdb.set_trace() 
        return ret 

    @memoize
    def get_gpu_util(self, local_bsz, frozen_layer): 
        if not hasattr(self, 'gpu_util'):
            return 100 
        df = self.gpu_util 
        ys = ["gpu_util"]
        if frozen_layer in df.frozen_layer.values: 
            df = df[df.frozen_layer == frozen_layer]
            if local_bsz < min(df.local_bsz.values): 
                local_bsz = min(df.local_bsz.values)
            if local_bsz > max(df.local_bsz.values): 
                local_bsz = max(df.local_bsz.values)
            
            interpolator = interp1d(df.local_bsz.values, df[ys].values, axis=0, fill_value='extrapolate')
            ret = interpolator(local_bsz)
            assert sum(ret) == sum(ret)
            return ret.item()
        
        if local_bsz in df.local_bsz.values: 
            df = df[df.local_bsz == local_bsz]
            if frozen_layer < min(df.frozen_layer.values): 
                frozen_layer = min(df.frozen_layer.values)
            if frozen_layer > max(df.frozen_layer.values): 
                frozen_layer = max(df.frozen_layer.values)
            
            interpolator = interp1d(df.frozen_layer.values, df[ys].values, axis=0, fill_value='extrapolate')
            ret = interpolator(frozen_layer)
            assert sum(ret) == sum(ret)
            return ret.item()

        fill_value = min(df[df.frozen_layer > frozen_layer].gpu_util.values)
        interpolator = LinearNDInterpolator(df[['frozen_layer', 'local_bsz']].values, df[ys].values, fill_value=fill_value)
        ret = interpolator([frozen_layer, local_bsz])[0]
        assert sum(ret) == sum(ret)
        if ret < 0: 
            import pdb; pdb.set_trace() 
        return ret.item() 
    def get_sharing_decay(self, local_util, other_util): 
        return min(max((120 - (local_util + other_util)) / 20, 0.1), 1)
        
    @memoize
    def get_throughput(self, placement, local_bsz, frozen_layer):
        # print(self.name, placement, local_bsz, frozen_layer, flush=True)
        # Normalize placement to the lexicographically smallest rotation.
        placement = tuple(filter(None, placement))
        placement = min(placement[i:] + placement[:i]
                        for i in range(len(placement)))
        placement_id = int("".join(map(str, placement)))
        xs = ["num_nodes", "num_replicas", "local_bsz", "frozen_layer"]
        ys = ["step_time", "sync_time"]
        if local_bsz < self.min_local_bsz: 
            local_bsz = self.min_local_bsz
        if placement_id in self.placements.placement.values and frozen_layer in self.placements.frozen_layer.values: 
            df = self.placements[self.placements.placement == placement_id]
            df = df[df.frozen_layer == frozen_layer]
            
            interpolator = interp1d(df.local_bsz.values, df[ys].values, axis=0, fill_value="extrapolate")
            ret = interpolator(local_bsz)
        elif placement_id in self.placements.placement.values:
            # Found in placement traces, interpolate between local_bsz.
            df = self.placements[self.placements.placement == placement_id]
            interpolator = LinearNDInterpolator(df[['local_bsz', 'frozen_layer']].values, df[ys].values, fill_value=100)
            ret = interpolator([local_bsz, frozen_layer])[0]
        else:
            # Interpolate between num_nodes, num_replicas, and local_bsz.
            df = self.placements.groupby(xs)[xs + ys].mean()
            df = pandas.concat([df, self.scalability], ignore_index=True)
            num_nodes, num_replicas = len(placement), sum(placement)
            num_nodes = min(num_nodes, 16)
            # interpolator = NearestNDInterpolator(df[xs].values, df[ys].values, fill_value=100)
            interpolator = NearestNDInterpolator(df[xs].values, df[ys].values)
            ret = interpolator([num_nodes, num_replicas, local_bsz, frozen_layer])[0]
        assert sum(ret) == sum(ret), "{} {} {} {}".format(self.name, placement, local_bsz, frozen_layer)
        return ret


def apply_accelerate(method=None):
    if  method == 'rewrite': 
        for key, app in APPLICATIONS.items(): 
            app.placements = app.rewrite_placements
    elif method == 'cache': 
        for key, app in APPLICATIONS.items(): 
            app.placements = app.cache_placements
    elif method == 'accelerate': 
        for key, app in APPLICATIONS.items(): 
            if hasattr(app, 'accelerate_placements'): 
                app.placements = app.accelerate_placements
            if hasattr(app, 'accelerate_scalability'): 
                app.scalability = app.accelerate_scalability
    elif method == 'int8': 
        for key, app in APPLICATIONS.items(): 
            if hasattr(app, 'int8_placements'): 
                app.placements = app.int8_placements
            if hasattr(app, 'int8_scalability'): 
                app.scalability = app.int8_scalability
    elif method is None: 
        pass 
    else:
        raise NotImplementedError 


def apply_memory_limit(memory_scale): 
    for key, app in APPLICATIONS.items():
        if 'llama' not in key: 
            app.update_max_host_memory(memory_scale) 

def apply_remove_prior(prior): 
    if prior == False: 
        # import pdb; pdb.set_trace() 
        for key, app in APPLICATIONS.items(): 
            app.grad_params = None 
            app.perf_params = None 
            app.max_profiled_replicas = 0

def apply_large_thr(large_thr): 
    for key, app in APPLICATIONS.items():
        app.perf_params = app.prior_perf_params(large_thr)
        
# TRACES_DIR = os.path.join(os.path.dirname(__file__), "traces")
# APPLICATIONS = {
#     "bert": Application(os.path.join(TRACES_DIR, "bert"), max_epochs=2),
#     "cifar10": Application(os.path.join(TRACES_DIR, "cifar10"), max_epochs=100),
#     "ncf": Application(os.path.join(TRACES_DIR, "ncf"), max_epochs=10),
#     "imagenet": Application(os.path.join(TRACES_DIR, "imagenet"), max_epochs=90),
#     "deepspeech2": Application(os.path.join(TRACES_DIR, "deepspeech2"), max_epochs=80),
#     "yolov3": Application(os.path.join(TRACES_DIR, "yolov3"), max_epochs=50, max_local_bsz=8),
# }

TRACES_DIR = os.path.join(os.path.dirname(__file__), "traces", "A800", "frozen")
APPLICATIONS = {
    "cifar10-ResNet18": Application(os.path.join(TRACES_DIR, "cifar10-ResNet18"), max_epochs=100),
    "cifar10-ResNet50": Application(os.path.join(TRACES_DIR, "cifar10-ResNet50"), max_epochs=100),
    "cifar10-MobileNetV2": Application(os.path.join(TRACES_DIR, "cifar10-MobileNetV2"), max_epochs=100),
    "cifar10-VGG19": Application(os.path.join(TRACES_DIR, "cifar10-VGG19"), max_epochs=100),
    "cifar10-GoogLeNet": Application(os.path.join(TRACES_DIR, "cifar10-GoogLeNet"), max_epochs=100),
    "VOC-yolo": Application(os.path.join(TRACES_DIR, "VOC-yolo"), max_epochs=50),
    "WikiText2-bert": Application(os.path.join(TRACES_DIR, "WikiText2-bert"), max_epochs=4),
    "imagenet-ResNet50": Application(os.path.join(TRACES_DIR, "imagenet-ResNet50"), max_epochs=90),
    "imagenet-MobileNetV2": Application(os.path.join(TRACES_DIR, "imagenet-MobileNetV2"), max_epochs=90),
    "imagenet-ResNet18": Application(os.path.join(TRACES_DIR, "imagenet-ResNet18"), max_epochs=90),
    # split line
    "squad-llama-3B": Application(os.path.join(TRACES_DIR, "squad-llama-3B"), max_epochs=4),
    "squad-llama-3B-pp": Application(os.path.join(TRACES_DIR, "squad-llama-3B-pp"), max_epochs=4),
    "sst2-llama-7B": Application(os.path.join(TRACES_DIR, "sst2-llama-7B"), max_epochs=4),
    "sst2-llama-7B-pp": Application(os.path.join(TRACES_DIR, "sst2-llama-7B-pp"), max_epochs=4),
    # pokeman dataset 
    "pokeman-ddpm": Application(os.path.join(TRACES_DIR, "pokeman-ddpm"), max_epochs=5),
}

# TRACES_DIR = os.path.join(os.path.dirname(__file__), "traces")
# APPLICATIONS = {
#     "cifar10-ResNet18": Application(os.path.join(TRACES_DIR, "cifar10"), max_epochs=100),
# }

# print('run application')