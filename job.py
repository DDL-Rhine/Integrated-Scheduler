import math
import numpy as np 
from goodput import GoodputFunction, fit_perf_params, FrozenGoodputFunction, customized_goodput_func
from llama_goodput import LlamaFrozenGoodputFunction, LlamaGoodputFunction, llama_fit_perf_params
from llama_pp_goodput import PPLlamaFrozenGoodputFunction, PPLlamaGoodputFunction, pp_llama_fit_perf_params
from speedup import SpeedupFunction, FrozenSpeedupFunction
import copy 

ConstGPUSharingDecay = 0.1
def is_equal_event(eventA, eventB): 
    return eventA.epoch == eventB.epoch and  eventA.frozen_layer == eventB.frozen_layer and eventA.global_batch_size == eventB.global_batch_size


def FreezeOut(epoch, tot_epoch, tot_layer_num, freeze_parameter): 
    return freeze_parameter[0] * max(0, (epoch + 1) * 1.0 / (tot_epoch + 1) - freeze_parameter[1])


def dynamic_search(self, batch_size, placement, frozen_layer):
    scale = batch_size / self.application.init_batch_size
    # Calculate true (simulated) throughput.
    step_time, sync_time = \
        self.application.get_throughput(placement, self.atomic_bsz, frozen_layer)

    accum_time = step_time - sync_time
    # Calculate true (simulated) efficiency. 
    if True and 'imagenet' in self.name: 
        sub_grad_sqr, sub_grad_var = \
            self.application.get_grad_stats(800, self.epoch, 0, True)
    else: 
        sub_grad_sqr, sub_grad_var = \
            self.application.get_grad_stats(batch_size, self.epoch, 0, True)
    
    grad_sqr, grad_var = sub_grad_sqr, sub_grad_var 


            
    if np.isscalar(sub_grad_sqr): 
        gain = (sub_grad_var + sub_grad_sqr) / (sub_grad_var / scale + sub_grad_sqr)
    else: 
        frozen_scale = 1 - abs(sum(grad_var[:frozen_layer]) / sum(grad_var[frozen_layer:])) 
        if frozen_scale <= 0.2: 
            return 0
        # gain = (sum(sub_grad_var) + abs(sum(sub_grad_sqr))) / (sum(sub_grad_var) / scale + abs(sum(sub_grad_sqr))) # * frozen_scale # * math.sqrt(abs(sum(sub_grad_sqr) / sum(grad_sqr)))
        gain = (sum(grad_var) + abs(sum(grad_sqr))) / (sum(grad_var) / scale + abs(sum(grad_sqr))) * frozen_scale # * frozen_scale # * math.sqrt(abs(sum(sub_grad_sqr) / sum(grad_sqr)))

    # Calculate true (simulated) goodput.
    total_time = step_time + accum_time * self.accum_steps
    goodput = gain / total_time
    return goodput 


class Job(object):

    pretrain = {}

    def __init__(self, name, application, submission_time, batch_size_lower_bound, batch_size_upper_bound, \
                    replica_lower_bound, replica_upper_bound,
                 target_num_replicas=None, target_batch_size=None, frozen_strategy=None, freeze_parameter=None, recommend_frozen_layer=None, deadline_factor=None, deadline=None, jcts=None):
        self.name = name
        self.application = application
        self.submission_time = submission_time
        self.target_num_replicas = target_num_replicas
        self.target_batch_size = target_batch_size
        self.completion_time = None
        self.current_time = 0
        self.rescale_time = 0
        self.placement = ()
        self.atomic_bsz = 0
        self.accum_steps = 0
        self.profile = {}
        self.perf_params = None
        self.grad_params = None
        self.best_metric = None
        self.progress = 0.0
        self.epoch = 0
        self.attained_service = 0
        self.deserved_service = 0
        self.running_time = 0 
        self.num_restarts = None 

        # new added 
        self.max_progress = self.application.get_progress(self.application.max_epochs)
        self.flop_ratios = self.application.get_flop_info()
        self.param_ratios = self.application.get_param_info()
        self.current_frozen_layer = 0
        self.previous_frozen_layer = 0
        self.total_layer = self.application._layer_num 
        self.estimate_speed = 0
        self.progress_per_epoch = self.application.get_progress(1) 
        self.estimate_max_local_bsz = None 
        self.frozen_strategy = frozen_strategy
        self.freeze_parameter = freeze_parameter
        if recommend_frozen_layer is not None: 
            self.recommend_frozen_layer = recommend_frozen_layer
        else: 
            self.recommend_frozen_layer = dict() 
        self.batch_size_lower_bound = batch_size_lower_bound
        self.batch_size_upper_bound = batch_size_upper_bound # TODO: to fix it later 
        self.replica_lower_bound = replica_lower_bound
        self.replica_upper_bound = replica_upper_bound
        
        # GPU sharing
        self.GPUSharingDecay = 1
        self.sharing_placement = None 
        try: 
            from simulator import args 
            self.LargeThr = args.LargeThr
        except Exception as e: 
            self.LargeThr = False
            
        self.sharing_job = None
        
        self.max_progress = self.application.get_progress(self.application.max_epochs)
        # 添加截止时间相关属性
        self.deadline_factor = deadline_factor if deadline_factor else np.random.uniform(1.2, 3.0)
        self.deadline = deadline  # 直接使用传入的deadline
        self.jcts = jcts
        
        
    @property
    def max_profiled_replicas(self):
        return max((k[1] for k in self.profile), default=0)

    def get_goodput_fn(self):
        app = self.application
        if '-pp' in self.name and self.LargeThr: 
            return PPLlamaGoodputFunction(self.perf_params, self.grad_params, app.init_batch_size, \
                frozen_layer=0, flop_info=app.get_flop_info(), param_info=app.get_param_info())
        elif 'llama' in self.name and self.LargeThr: 
            return LlamaGoodputFunction(self.perf_params, self.grad_params, app.init_batch_size, \
                frozen_layer=0, flop_info=app.get_flop_info(), param_info=app.get_param_info())
            
        return GoodputFunction(self.perf_params, self.grad_params, app.init_batch_size, \
            frozen_layer=0, flop_info=app.get_flop_info(), param_info=app.get_param_info())

    def get_speedup_fn(self):
        if self.perf_params is None:
            return lambda n, r: r
        return SpeedupFunction(self.get_goodput_fn(), self.batch_size_upper_bound,
                               self.get_local_bsz_range(),
                               accumulation=True)
    
    def get_local_bsz_range(self): 
        app = self.application 
        return (app.min_local_bsz, app.max_local_bsz if self.estimate_max_local_bsz is None else max(app.max_local_bsz, self.estimate_max_local_bsz))

    def update_local_bsz(self, placement):
        app = self.application
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size 
        if batch_size is None and self.perf_params is None:
            batch_size = max(app.init_batch_size, app.min_local_bsz * num_replicas)
        if batch_size is None:
            goodput_fn = self.get_goodput_fn()
            _, self.atomic_bsz, self.accum_steps = goodput_fn.optimize(
                num_nodes, num_replicas, self.batch_size_upper_bound,
                self.get_local_bsz_range(), accumulation=True)
        else:
            local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
            self.accum_steps = math.ceil(local_bsz / app.max_local_bsz - 1e-8) - 1
            if num_replicas == 1 and batch_size > app.init_batch_size:
                self.accum_steps = max(1, self.accum_steps)
            self.atomic_bsz = math.ceil(local_bsz / (self.accum_steps + 1) - 1e-8)
        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = min(self.atomic_bsz, int(self.batch_size_upper_bound / count))
        if self.frozen_strategy == 'FreezeOut' and self.freeze_parameter is not None : 
            self.current_frozen_layer = \
                FreezeOut(self.epoch, self.application.max_epochs, self.application.get_layer_num(), self.freeze_parameter) 
        elif self.frozen_strategy == 'None': 
            self.current_frozen_layer = 0 
        elif self.frozen_strategy == 'Dynamic' and self.recommend_frozen_layer is not None : 
            if True: 
                if self.epoch not in self.recommend_frozen_layer: 
                    if self.epoch == self.application.max_epochs: 
                        self.recommend_frozen_layer[self.epoch] = (0, 0)
                    else: 
                        max_goodput = -1 
                        max_layer = int(self.application.get_layer_num() * 2 * (self.epoch / self.application.max_epochs))
                        max_layer = min(self.application.get_max_frozen_layer(), max_layer)
                        base_goodput = None 
                        for layer in range(0, max_layer + 1, 4): 
                            goodput = dynamic_search(self, batch_size, placement, layer)
                            if goodput > max_goodput: 
                                max_goodput = goodput
                                max_layer = layer 
                            if base_goodput is None: 
                                base_goodput = goodput
                            
                        self.recommend_frozen_layer[self.epoch] = (max_layer, max_goodput)
                    
                self.current_frozen_layer = self.recommend_frozen_layer[self.epoch][0]
        # import pdb; pdb.set_trace() 
        self.current_frozen_layer = int(min(self.current_frozen_layer, self.application.get_max_frozen_layer()))
        

    def update_params(self, num_nodes, num_replicas, local_bsz,
                      step_time, sync_time, grad_sqr, grad_var, max_memory, gpu_util, frozen_layer):
        self.grad_params = (grad_sqr, grad_var)
        if (num_nodes, num_replicas, local_bsz, frozen_layer) in self.profile:
            return
        self.profile[num_nodes, num_replicas, local_bsz, frozen_layer] = step_time, sync_time, max_memory, gpu_util
        num_nodes = np.array([key[0] for key in self.profile])
        num_replicas = np.array([key[1] for key in self.profile])
        frozen_layer = np.array([key[3] for key in self.profile])
        flop_ratios = np.array([self.flop_ratios[key[3]] for key in self.profile])
        param_ratios = np.array([self.param_ratios[key[3]] for key in self.profile])
        local_bsz = np.array([key[2] for key in self.profile])
        step_time = np.array([val[0] for val in self.profile.values()])
        sync_time = np.array([val[1] for val in self.profile.values()])
        
        compute_time = step_time - sync_time
        if '-pp' in self.name and self.LargeThr: 
            self.perf_params = pp_llama_fit_perf_params(
                num_nodes, num_replicas, local_bsz, compute_time, step_time, flop_ratios=flop_ratios, param_ratios=param_ratios, init_params=self.perf_params)
        elif 'llama' in self.name and self.LargeThr: 
            self.perf_params = llama_fit_perf_params(
                num_nodes, num_replicas, local_bsz, compute_time, step_time, flop_ratios=flop_ratios, param_ratios=param_ratios, init_params=self.perf_params)
        else: 
            self.perf_params = fit_perf_params(
                num_nodes, num_replicas, local_bsz, compute_time, step_time, flop_ratios=flop_ratios, param_ratios=param_ratios, init_params=self.perf_params)
            
    def step(self, seconds, deserved_gpu, interference=0.0):
        deserved_gpu = max(self.max_profiled_replicas, deserved_gpu)
        if self.target_batch_size is not None: 
            deserved_gpu = min((self.target_batch_size // self.application.min_local_bsz), deserved_gpu)
        self.deserved_service += seconds * deserved_gpu
        if not self.placement:
            # No resources are allocated to this job.
            self.current_time += seconds
            return
        delay = min(self.rescale_time, seconds)
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.running_time += delay
        self.rescale_time -= delay
        seconds -= delay
        while seconds > 0 and self.completion_time is None:
            assert self.epoch < self.application.max_epochs
            # Calculate current job configurations.
            placement = tuple(filter(None, self.placement))
            if 'llama' in self.name: 
                num_replicas = sum(placement)
                if num_replicas <= 4: 
                    placement = tuple([num_replicas])
                else: 
                    placement = tuple([4 for _ in range(num_replicas//4)])
                
            num_nodes, num_replicas = len(placement), sum(placement)
            # local_bsz = self.atomic_bsz
            batch_size = num_replicas * self.atomic_bsz * (self.accum_steps + 1)
            scale = batch_size / self.application.init_batch_size
            # Calculate true (simulated) throughput.
            # if 'squad-llama' in self.name: 
            #     step_time, sync_time = \
            #         self.application.get_throughput(placement, max(self.atomic_bsz, 32), self.current_frozen_layer)
            # else: 
            step_time, sync_time = \
                self.application.get_throughput(placement, self.atomic_bsz, self.current_frozen_layer)
            
            accum_time = step_time - sync_time
            
            grad_sqr, grad_var = \
                self.application.get_grad_stats(batch_size, self.epoch, 0, True)

            if True and 'imagenet' in self.name: 
                grad_sqr, grad_var = \
                    self.application.get_grad_stats(800, self.epoch, 0, True)
            
            if self.current_frozen_layer == 0: 
                grad_sqr = abs(sum(grad_sqr))
                grad_var = abs(sum(grad_var))
                if np.isscalar(grad_sqr): 
                    gain = (grad_var + grad_sqr) / (grad_var / scale + grad_sqr) 
            else: 
                frozen_scale = 1 - abs(sum(grad_var[:self.current_frozen_layer]) / sum(grad_var[self.current_frozen_layer:]))
                if frozen_scale <= 0.2: 
                    self.current_frozen_layer = 0 
                    continue 
                gain = (sum(grad_var) + abs(sum(grad_sqr))) / (sum(grad_var) / scale + abs(sum(grad_sqr))) * frozen_scale # * frozen_scale # * math.sqrt(abs(sum(sub_grad_sqr) / sum(grad_sqr)))
                grad_sqr = abs(sum(grad_sqr))
                grad_var = abs(sum(grad_var))
                

            gpu_util = self.application.get_gpu_util(self.atomic_bsz, self.current_frozen_layer)
            if gpu_util > 50.0 and self.sharing_placement is not None: 
                self.GPUSharingDecay = ConstGPUSharingDecay # to simulate the wrong prediction 
            elif self.sharing_placement is not None: 
                sharing_gpu_util = self.sharing_job.application.get_gpu_util(self.sharing_job.atomic_bsz, self.sharing_job.current_frozen_layer)
                if sharing_gpu_util > 50: 
                    self.GPUSharingDecay = ConstGPUSharingDecay
                    
            
            # Update the estimated throughput/efficiency parameters.
            self.update_params(num_nodes, num_replicas, self.atomic_bsz,
                               step_time, sync_time, grad_sqr, grad_var, -1, gpu_util, 0)
            # Calculate true (simulated) goodput.
            total_time = step_time + accum_time * self.accum_steps
            goodput = gain / total_time * (1.0 - interference) * self.GPUSharingDecay

            # Update current epoch and progress.
            next_progress = self.application.get_progress(self.epoch + 1)
            if self.progress + goodput * seconds < next_progress:
                # Used up the entire time interval without finishing an epoch.
                self.progress += goodput * seconds
                self.current_time += seconds
                self.attained_service += seconds * sum(self.placement)
                self.running_time += seconds
                seconds = 0
            else:
                # Crossed an epoch boundary before finishing the time interval.
                self.epoch += 1
                delta = round(float((next_progress - self.progress) / goodput))
                # delta = float((next_progress - self.progress) / goodput) # more accurate way 
                assert delta <= seconds
                completion_epoch = \
                    self.application.get_completion_epoch(batch_size)
                if self.epoch > completion_epoch:
                    self.completion_time = self.current_time + delta
                self.progress = next_progress
                self.best_metric = \
                    self.application.get_best_metric(batch_size, self.epoch)
                self.current_time += delta
                self.attained_service += delta * sum(self.placement)
                self.running_time += seconds
                seconds -= delta
                # Re-scale batch size between epochs.
            self.update_local_bsz(self.placement) 
        
        self.current_time += seconds  # Add any remaining time.
        self.current_time = int(round(self.current_time))

    def reallocate(self, placement, sharing_job=None):
        if placement:
            if sum(placement) == 0.5: 
                self.sharing_placement = placement
                self.sharing_job = sharing_job
                self.GPUSharingDecay = 0.90
                from simulator import args 
                if hasattr(args, 'GPUSharingThr') and args.GPUSharingThr > 0: 
                    self.GPUSharingDecay = args.GPUSharingThr / 100
                self.placement = (1,)
            else: 
                self.sharing_placement = None 
                self.sharing_job = None 
                self.GPUSharingDecay = 1
                self.placement = tuple(placement)
            
            self.update_local_bsz(self.placement)
            self.rescale_time = 30  # Start re-scale countdown.
            from simulator import args 
            if hasattr(args, 'disable_rescale') and args.disable_rescale:
                self.rescale_time = 0 # TODO
            if hasattr(args, 'rescale_time'): 
                self.rescale_time = int(args.rescale_time) 

            if self.num_restarts is None:
                self.num_restarts = 0
            else:
                self.num_restarts += 1
        else:  # De-allocate all resources.
            self.placement = ()
            self.atomic_bsz = 0


def PipeTransformerMethod(layer_num, alpha, epoch):
    second_term = 0.0
    for e in range(2, epoch + 1):
        second_term += ((layer_num * alpha) / pow(1 - alpha, e))
    return pow(1 - alpha, epoch) * ((layer_num * alpha) / (1 - alpha) + second_term)




class FrozenJob(Job):
    pretrain = {}

    def __init__(self, name, application, submission_time, batch_size_lower_bound, batch_size_upper_bound, replica_lower_bound, replica_upper_bound, 
                frozen_alpha, freeze_parameter = None, fixed_batch_size=None, shrink_range=False, elastic=None, 
                target_num_replicas=None, target_batch_size=None, reproduce_record=False, batch_fixed=False, deadline_factor=None, deadline=None, jcts=None):
        super(FrozenJob, self).__init__(name, application, submission_time, target_num_replicas, target_batch_size, replica_lower_bound=replica_lower_bound, replica_upper_bound=replica_upper_bound, deadline_factor=deadline_factor, deadline=deadline, jcts=jcts)
        self.disable_frozen_operation = True 
        self.disable_cache_operation = True 
        self.disable_graph_optimization = True 
        self.frozen_alpha = frozen_alpha
        self.fixed_batch_size = fixed_batch_size
        self.elastic = elastic 
        self.perf_params = copy.deepcopy(application.perf_params)
        self.grad_params = copy.deepcopy(application.grad_params)
        self.reproduce_record = reproduce_record
        if self.reproduce_record: 
            self.event_list = list() 
        self.batch_fixed = batch_fixed
        last_digit = int(self.name[-1])
        self.shrink_range = (shrink_range and last_digit % 2 == 0)
        self.batch_size_lower_bound = batch_size_lower_bound
        self.batch_size_upper_bound = batch_size_upper_bound
        self.GPUSharingDecay = 1
        self.sharing_placement = None 
        self.replica_lower_bound = replica_lower_bound
        self.replica_upper_bound = replica_upper_bound
        self.frozen_strategy = None 
        try: 
            from simulator import args
            self.LargeThr = args.LargeThr
        except Exception as e: 
            self.LargeThr = False
        
        self.freeze_parameter = freeze_parameter 
        if args.penalty == False: 
            self.frozen_strategy = 'FreezeOut'
            self.freeze_parameter = freeze_parameter 
        
        
        
    @property
    def max_profiled_replicas(self):
        return max(max((k[1] for k in self.profile), default=0), self.application.max_profiled_replicas)

    def candidate_frozen_set(self, ): 
        if not self.elastic in ['layer']: 
            return [0]
        max_layer_range = self.application.get_max_frozen_layer()
        if self.shrink_range: 
            max_layer_range //= 2
        if 'llama' in self.name: 
            return [i * 7 for i in range(21)]
        else: 
            return [i for i in range(max_layer_range - 1)]
        

    def get_frozen_goodput_fn(self, ): 
        app = self.application 
        if self.frozen_strategy == 'FreezeOut' and self.freeze_parameter is not None: 
            candidate_layer = FreezeOut(self.epoch, self.application.max_epochs, self.application.get_layer_num(), self.freeze_parameter)
            candidate_layer = int(min(candidate_layer, self.application.get_max_frozen_layer()))
            frozen_set = np.unique([0, self.current_frozen_layer, candidate_layer])
        else: 
            frozen_set = self.candidate_frozen_set()
        
        if '-pp' in self.name and self.LargeThr: 
            return PPLlamaFrozenGoodputFunction(
                self.perf_params, self.grad_params, app.init_batch_size, fixed_batch_size=self.fixed_batch_size, \
                        application=self.application, current_progress=self.progress, current_epoch=self.epoch, 
                        frozen_layer=self.current_frozen_layer, frozen_set=frozen_set, flop_info=app.get_flop_info(), \
                        param_info=app.get_param_info(), estimate_speed=self.estimate_speed, frozen_alpha=self.frozen_alpha, \
                        attained_service=self.attained_service, elastic=self.elastic
            )
        elif 'llama' in self.name and self.LargeThr: 
            # import pdb; pdb.set_trace() 
            return LlamaFrozenGoodputFunction(
                self.perf_params, self.grad_params, app.init_batch_size, fixed_batch_size=self.fixed_batch_size, \
                        application=self.application, current_progress=self.progress, current_epoch=self.epoch, 
                        frozen_layer=self.current_frozen_layer, frozen_set=frozen_set, flop_info=app.get_flop_info(), \
                        param_info=app.get_param_info(), estimate_speed=self.estimate_speed, frozen_alpha=self.frozen_alpha, \
                        attained_service=self.attained_service, elastic=self.elastic
            )
        return FrozenGoodputFunction(self.perf_params, self.grad_params, app.init_batch_size, fixed_batch_size=self.fixed_batch_size, \
                        application=self.application, current_progress=self.progress, current_epoch=self.epoch, 
                        frozen_layer=self.current_frozen_layer, frozen_set=frozen_set, flop_info=app.get_flop_info(), \
                        param_info=app.get_param_info(), estimate_speed=self.estimate_speed, frozen_alpha=self.frozen_alpha, \
                        attained_service=self.attained_service, elastic=self.elastic)

    def get_gpu_util_fn(self, ): 
        app = self.application
        placement = (1,)
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size
        if hasattr(self, 'elastic') and self.elastic in ['static']: 
            batch_size = self.fixed_batch_size 
        if self.batch_fixed: 
            batch_size = self.fixed_batch_size
        
        if batch_size is None and self.perf_params is None:
            batch_size = max(app.init_batch_size, app.min_local_bsz * num_replicas)
        if batch_size is None:
            goodput_fn = self.get_goodput_fn()
            _, atomic_bsz, accum_steps, frozen_layer = goodput_fn.optimize(
                num_nodes, num_replicas, self.batch_size_upper_bound,
                self.get_local_bsz_range(), accumulation=True)
        else: 
            goodput_fn = self.get_goodput_fn()
            _, atomic_bsz, accum_steps, frozen_layer = goodput_fn.optimize(
                num_nodes, num_replicas, self.batch_size_upper_bound,
                self.get_local_bsz_range(), accumulation=True)
        count = num_replicas * (self.accum_steps + 1)
        atomic_bsz = min(atomic_bsz, int(self.batch_size_upper_bound / count))
        return self.application.get_gpu_util(atomic_bsz, frozen_layer)
        
    def get_frozen_speedup_fn(self):
        if self.perf_params is None:
            return lambda n, r: r
        app = self.application 

        return FrozenSpeedupFunction(self.get_goodput_fn(), self.get_gpu_util_fn(), max_batch_size=self.batch_size_upper_bound, 
                                fixed_batch_size=self.fixed_batch_size,
                                atomic_bsz_range=self.get_local_bsz_range(),
                               accumulation=True, mem_size=32, elastic=self.elastic)


    def update_frozen_layer(self, frozen_layer): 
        if self.elastic in ['layer']: 
            self.current_frozen_layer = frozen_layer 
        assert self.elastic in ['static', 'batch', 'layer']

    def frozen_step(self, seconds, deserved_gpu, interference=0.0):
        deserved_gpu = max(self.max_profiled_replicas, deserved_gpu)
        if self.target_batch_size is not None: 
            deserved_gpu = min((self.target_batch_size // self.application.min_local_bsz), deserved_gpu)
        self.deserved_service += seconds * deserved_gpu
        
        if not self.placement:
            # No resources are allocated to this job.
            self.current_time += seconds
            return
        
        delay = min(self.rescale_time, seconds)
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        deserved_gpu = max(self.max_profiled_replicas, deserved_gpu)
        if self.target_batch_size is not None: 
            deserved_gpu = min((self.target_batch_size // self.application.min_local_bsz), deserved_gpu)
        self.running_time += delay
        self.rescale_time -= delay
        seconds -= delay
        placement = tuple(filter(None, self.placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        while seconds > 0 and self.completion_time is None:
            assert self.epoch < self.application.max_epochs
            # Calculate current job configurations.
            if self.elastic in ['static']: 
                batch_size = self.fixed_batch_size 
            else: 
                batch_size = num_replicas * self.atomic_bsz * (self.accum_steps + 1) 

            scale = batch_size / self.application.init_batch_size
            
            # Calculate true (simulated) throughput.
            if 'llama' in self.name: 
                num_replicas = sum(placement)
                if num_replicas < 4: 
                    placement = tuple([min(num_replicas, 2)])
                else: 
                    placement = tuple([4 for _ in range(num_replicas//4)])
                num_nodes, num_replicas = len(placement), sum(placement)
            
            step_time, sync_time = \
                self.application.get_throughput(placement, self.atomic_bsz, self.current_frozen_layer)

            accum_time = step_time - sync_time
            # Calculate true (simulated) efficiency. 
            if True and 'imagenet' in self.name: 
                sub_grad_sqr, sub_grad_var = \
                    self.application.get_grad_stats(800, self.epoch, 0, True)
            else: 
                sub_grad_sqr, sub_grad_var = \
                    self.application.get_grad_stats(batch_size, self.epoch, 0, True)
            
            grad_sqr, grad_var = sub_grad_sqr, sub_grad_var 
            
            if np.isscalar(sub_grad_sqr): 
                gain = (sub_grad_var + sub_grad_sqr) / (sub_grad_var / scale + sub_grad_sqr)
            else:
                frozen_var = sum(abs(sub_grad_var[:self.current_frozen_layer]))
                non_frozen_var =  sum(sub_grad_var[self.current_frozen_layer:])
                frozen_sqr = 0 
                non_frozen_sqr = 0 
                frozen_scale = customized_goodput_func(frozen_var=frozen_var, frozen_sqr=frozen_sqr, \
                    non_frozen_var=non_frozen_var, non_frozen_sqr=non_frozen_sqr)
                
                
                if frozen_scale <= 0.2: 
                    self.update_frozen_layer(0)
                    continue 
                gain = (sum(grad_var) + abs(sum(grad_sqr))) / (sum(grad_var) / scale + abs(sum(grad_sqr))) * frozen_scale # * frozen_scale # * math.sqrt(abs(sum(sub_grad_sqr) / sum(grad_sqr)))
            
                
            if 'llama' not in self.name: 
                max_memory = min(self.application.get_gpu_memory(self.atomic_bsz, self.current_frozen_layer), self.application._max_host_memory) # for simulation
                gpu_util = self.application.get_gpu_util(self.atomic_bsz, self.current_frozen_layer)
                if self.sharing_placement is not None: 
                    sharing_gpu_util = self.sharing_job.application.get_gpu_util(self.sharing_job.atomic_bsz, self.sharing_job.current_frozen_layer)
                    self.GPUSharingDecay = self.application.get_sharing_decay(gpu_util, sharing_gpu_util)
            else: 
                max_memory, gpu_util = 0, 100
            
            # Update the estimated throughput/efficiency parameters.
            self.update_params(num_nodes, num_replicas, self.atomic_bsz,
                               step_time, sync_time, grad_sqr, grad_var, max_memory, gpu_util, self.current_frozen_layer)
            # Calculate true (simulated) goodput.
            total_time = step_time + accum_time * self.accum_steps
            goodput = gain / total_time * (1.0 - interference) * self.GPUSharingDecay
            # Update current epoch and progress.
            next_progress = self.application.get_progress(self.epoch + 1)
            
            if self.progress + goodput * seconds < next_progress:
                # Used up the entire time interval without finishing an epoch.
                self.progress += goodput * seconds
                self.current_time += seconds
                self.attained_service += seconds * sum(self.placement)
                self.running_time += seconds
                seconds = 0
            else:
                # Crossed an epoch boundary before finishing the time interval.
                self.epoch += 1
                delta = round(float((next_progress - self.progress) / goodput))
                # delta = float((next_progress - self.progress) / goodput)
                assert delta <= seconds
                completion_epoch = \
                    self.application.get_completion_epoch(batch_size)
                if self.epoch > completion_epoch:
                    self.completion_time = self.current_time + delta
                self.progress = next_progress
                self.best_metric = \
                    self.application.get_best_metric(batch_size, self.epoch)
                self.current_time += delta
                self.attained_service += delta * sum(self.placement)
                self.running_time += delta
                seconds -= delta
                
                # Re-scale batch size between epochs.
            self.update_local_bsz(self.placement)
            self.update_local_bsz_range()
        batch_size = num_replicas * self.atomic_bsz * (self.accum_steps + 1) 
        
        self.current_time += seconds  # Add any remaining time.
        self.current_time = int(round(self.current_time))
        # self.estimate_speed = 0.1 * self.estimate_speed + 0.9 * (self.progress - init_progress) / self.progress_per_epoch / (self.attained_service - init_attained_service + 1e-3)

    def update_local_bsz_range(self): 
        from simulator import args 
        if hasattr(args, 'disable_memory_scale') and args.disable_memory_scale: 
            return 
        
        if self.profile is not None and len(self.profile) > 0: 
            # self.profile[num_nodes, num_replicas, local_bsz, frozen_layer] = step_time, sync_time, max_memory
            close_layer, max_bsz, max_memory = -100000, -1, -1
            
            for key, value in self.profile.items(): 
                if key[-1] <= self.current_frozen_layer: 
                    if self.current_frozen_layer - key[-1] < self.current_frozen_layer - close_layer: 
                        close_layer = key[-1]
                        max_bsz = key[-2]
                        max_memory = value[-1]
                    elif self.current_frozen_layer - key[-1] == self.current_frozen_layer - close_layer: 
                        if key[-2] > max_bsz: 
                            max_bsz = key[-2]
                            max_memory = value[-1]
                
            
            # import pdb; pdb.set_trace() 
            if max_memory > 0 and 'llama' not in self.name: 
                max_host_memory = self.application._max_host_memory
                max_batch_size = int(max_host_memory / max_memory * max_bsz * 0.95)
                self.estimate_max_local_bsz = min(self.application.max_stats_bsz, max(self.application.max_local_bsz, max_batch_size)) 
                # avoid memory error 
                self.estimate_max_local_bsz = min(self.application.layer_batch_parteo[self.current_frozen_layer], self.estimate_max_local_bsz)
                self.estimate_max_local_bsz = max(self.estimate_max_local_bsz, self.application.min_local_bsz)

    def frozen_update_local_bsz(self, placement):
        app = self.application
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size
        if hasattr(self, 'elastic') and self.elastic in ['static']: 
            batch_size = self.fixed_batch_size 
        if self.batch_fixed: 
            batch_size = self.fixed_batch_size
        
        self.update_local_bsz_range()
        
        if batch_size is None and self.perf_params is None:
            batch_size = max(app.init_batch_size, app.min_local_bsz * num_replicas)
        if batch_size is None:
            goodput_fn = self.get_goodput_fn()
            _, self.atomic_bsz, self.accum_steps, frozen_layer = goodput_fn.optimize(
                num_nodes, num_replicas, self.batch_size_upper_bound,
                self.get_local_bsz_range(), accumulation=True)
            self.previous_frozen_layer = self.current_frozen_layer
            self.update_frozen_layer(frozen_layer) # TODO
        else: 
            goodput_fn = self.get_goodput_fn()
            _, self.atomic_bsz, self.accum_steps, frozen_layer = goodput_fn.optimize(
                num_nodes, num_replicas, self.batch_size_upper_bound,
                self.get_local_bsz_range(), accumulation=True)
            self.previous_frozen_layer = self.current_frozen_layer
            self.update_frozen_layer(frozen_layer) # TODO
        # else:
        #     local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
        #     self.accum_steps = math.ceil(local_bsz / app.max_local_bsz - 1e-8) - 1
        #     if num_replicas == 1 and batch_size > app.init_batch_size:
        #         self.accum_steps = max(1, self.accum_steps)
        #     self.atomic_bsz = math.ceil(local_bsz / (self.accum_steps + 1) - 1e-8)
        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = min(self.atomic_bsz, int(self.batch_size_upper_bound / count))


    def frozen_reallocate(self, placement, sharing_job=None):
        if placement:
            if sum(placement) == 0.5: 
                self.sharing_placement = placement
                self.sharing_job = sharing_job
                self.GPUSharingDecay = 0.90
                from simulator import args 
                if hasattr(args, 'GPUSharingThr') and args.GPUSharingThr > 0: 
                    self.GPUSharingDecay = args.GPUSharingThr / 100
                self.placement = (1,)
                assert self.sharing_job is not None 
            else: 
                self.sharing_placement = None 
                self.sharing_job = None 
                self.GPUSharingDecay = 1
                self.placement = tuple(placement)
            
            
            self.update_local_bsz(self.placement)
            self.rescale_time = 30  # Start re-scale countdown.
            from simulator import args 
            if hasattr(args, 'disable_rescale') and args.disable_rescale:
                self.rescale_time = 0 # TODO
            if hasattr(args, 'rescale_time'): 
                self.rescale_time = int(args.rescale_time) 
                
            if self.num_restarts is None:
                self.num_restarts = 0
            else:
                self.num_restarts += 1
        else:  # De-allocate all resources.
            self.sharing_placement = None 
            self.placement = ()
            self.atomic_bsz = 0
    
    
    get_goodput_fn = get_frozen_goodput_fn
    get_speedup_fn = get_frozen_speedup_fn
    step = frozen_step
    update_local_bsz = frozen_update_local_bsz
    reallocate = frozen_reallocate