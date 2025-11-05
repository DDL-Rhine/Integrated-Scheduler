# !pip install gekko
# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import copy
import logging
import numpy as np

from collections import OrderedDict
from mip import *
import numpy as np
import copy
import math


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class SimplePolluxPolicy(object):
    def __init__(self):
        self._prev_states = None
        self._prev_jobs = None
        self._prev_nodes = None
        # Utilization thresholds for cluster autoscaling.
        self._min_util = 0.35
        self._max_util = 0.65

    def _allocations_to_state(self, allocations, jobs, nodes):
        jobs_index = {key: idx for idx, key in enumerate(jobs)}
        nodes_index = {key: idx for idx, key in enumerate(nodes)}
        state = np.zeros((len(jobs), len(nodes)), dtype=np.int32)
        for job_key, alloc in allocations.items():
            for node_key in (key for key in alloc if key in nodes_index):
                state[jobs_index[job_key], nodes_index[node_key]] += 1
        return state

    def _state_to_allocations(self, state, jobs, nodes):
        allocations = {}
        for job_idx, job_key in enumerate(jobs):
            for node_idx, node_key in enumerate(nodes):
                count = state[job_idx, node_idx]
                allocations.setdefault(job_key, []).extend([node_key] * count)
        return allocations

    def _adapt_prev_states(self, jobs, nodes):
        # Adapt the previously saved optimization states to initialize the
        # current genetic algorithm states.
        #shape = (len(self._prev_states), len(jobs), 2 * len(nodes))
        shape = (len(self._prev_states), len(jobs), len(nodes))
        states = np.zeros(shape, dtype=np.int)
        jobs_src = [i for i, key in enumerate(self._prev_jobs) if key in jobs]
        jobs_dst = [i for i, key in enumerate(jobs) if key in self._prev_jobs]
        placeholder = len(self._prev_nodes)  # Next placeholder node to copy.
        # Set allocations for physical (non-placeholder) nodes.
        nodes_index = {key: i for i, key in enumerate(self._prev_nodes)}
        for i, key in enumerate(nodes):
            if key in nodes_index:
                states[:, jobs_dst, i] = \
                    self._prev_states[:, jobs_src, nodes_index[key]]
            elif placeholder < self._prev_states.shape[2]:
                # New node, use allocations for a previous placeholder node.
                states[:, jobs_dst, i] = \
                    self._prev_states[:, jobs_src, placeholder]
                placeholder += 1
        # Set allocations for placeholder nodes.
        #for i in range(len(nodes), 2 * len(nodes)):
        #    if placeholder < self._prev_states.shape[2]:
        #        states[:, jobs_dst, i] = \
        #            self._prev_states[:, jobs_src, placeholder]
        #        placeholder += 1
        return states

    def _select_result(self, values, max_nodes):
        if np.amin(values[:, 1]) > max_nodes:
            return None
        return np.argmin(np.where(values[:, 1] <= max_nodes, values[:, 0], 0))

    def _desired_nodes(self, utilities, values, nodes):
        idx = self._select_result(values, len(nodes))
        if idx is not None and \
                self._min_util <= utilities[idx] <= self._max_util:
            return len(nodes)
        target_util = (self._min_util + self._max_util) / 2
        best_util = np.inf
        best_val = 0.0
        best_nodes = len(nodes)
        for util, (val, num_nodes) in zip(utilities, values):
            if util > best_util and val < best_val:
                best_util = util
                best_val = val
                best_nodes = num_nodes
            elif util < best_util and val > best_val:
                continue
            elif abs(util - target_util) < abs(best_util - target_util):
                best_util = util
                best_val = val
                best_nodes = num_nodes
        return int(best_nodes)

    def optimize(self, jobs, nodes, base_allocations, node_template):
        """
        Run one optimization cycle of the Pollux scheduling policy.

        Arguments:
            jobs (dict): map from job keys to `JobInfo` objects which
                correspond to the incomplete jobs which should be optimized.
            nodes (dict): map from node keys to `NodeInfo` objects which
                correspond to the existing nodes in the cluster.
            base_allocations (dict): map from job keys to their current
                resource allocations, in the form of a list of a node key for
                each replica.
            node_template (NodeInfo): represents a node which can be requested,
                used to decide the cluster size for cluster auto-scaling.

        Returns:
            dict: map from job keys to their optimized resource allocations,
                in the form of a list of a node key for each replica.
        """

        # A job is considered pinned if it's non-preemptible *and* already has
        # an allocation.
        def ispinned(key, job):
            return not job.preemptible and base_allocations.get(key, []) != []

        # We sort the jobs based on min_replicas and then creation_timestamp,
        # so jobs wanting lower or no min_replicas guarantees are prioritized
        # ahead of those wanting higher min_replicas guarantees to avoid
        # underutilization of cluster. Within a same min_replicas value, they
        # will follow FIFO order. Pinned jobs are aggregated at front because
        # they already have an allocation and won't affect allocations of the
        # rest of the jobs.
        jobs = OrderedDict(sorted(jobs.items(),
                                  key=lambda kv: (not ispinned(kv[0], kv[1]),
                                                  kv[1].attained_service,
                                                  kv[1].creation_timestamp)))
        nodes = OrderedDict(  # Sort preemptible nodes last.
            sorted(nodes.items(), key=lambda kv: (kv[1].preemptible, kv[0])))
        #base_state = np.concatenate(
        #    (self._allocations_to_state(base_allocations, jobs, nodes),
        #     np.zeros((len(jobs), len(nodes)), dtype=np.int)), axis=1)
        base_state = \
            self._allocations_to_state(base_allocations, jobs, nodes)

        if self._prev_states is None:
            states = np.expand_dims(base_state, 0)
        else:
            states = self._adapt_prev_states(jobs, nodes)

        problem = Problem(list(jobs.values()), list(nodes.values()), base_state)
        solved_state = problem.solve() 
        allocations = np.zeros((len(jobs), len(nodes)), np.int32)
        cluster_state = np.array([node.resources['nvidia.com/gpu'] for node in nodes.values()])
        job_to_allocations = [(job_id, job, allocated_gpu) for job_id, (job, allocated_gpu) in enumerate(zip(jobs, solved_state))]
        job_to_allocations = sorted(job_to_allocations, key=lambda x: -x[2])
        print(job_to_allocations)
        for job_id, _, allocated_gpu in job_to_allocations: 
            job_to_allocation = [0 for node in nodes]
            while allocated_gpu > 0:
                node_id = np.argmax(cluster_state)
                free_gpu_num = cluster_state[node_id]
                if free_gpu_num == 0: 
                    import pdb; pdb.set_trace()
                if free_gpu_num >= allocated_gpu: 
                    job_to_allocation[node_id] = allocated_gpu
                    cluster_state[node_id] -= allocated_gpu
                    allocated_gpu = 0
                else: 
                    job_to_allocation[node_id] = free_gpu_num
                    cluster_state[node_id] -= free_gpu_num
                    allocated_gpu -= free_gpu_num
            for node_id, gpu_num in enumerate(job_to_allocation): 
                allocations[job_id][node_id] = gpu_num 
        return self._state_to_allocations(allocations, jobs, nodes), 16 


class Problem(object):
    def __init__(self, jobs, nodes, base_state):
        """
        Multi-objective optimization problem used by PolluxPolicy to determine
        resource allocations and desired cluster size. Optimizes for the best
        performing cluster allocation using only the first N nodes. The cluster
        performance and N are the two objectives being optimized, resulting in
        a set of Pareto-optimal solutions.

        The optimization states are a 3-D array of replica assignments with
        shape (pop_size x num_jobs x num_nodes). The element at k, j, n encodes
        the number of job j replicas assigned to node n, in the kth solution.

        Arguments:
            jobs (list): list of JobInfo objects describing the incomplete jobs
                which need to be scheduled.
            nodes (list): list of NodeInfo objects describing the nodes in the
                cluster, in decreasing order of allocation preference.
            base_state (numpy.array): base optimization state corresponding to
                the current cluster allocations. Shape: (num_jobs x num_nodes).
        """
        assert base_state.shape == (len(jobs), len(nodes))
        self._jobs = jobs
        self._nodes = nodes
        self._base_state = base_state
        # Find which resource types are requested by at least one job.
        rtypes = sorted(set.union(*[set(job.resources) for job in jobs]))
        # Build array of job resources: <num_jobs> x <num_rtypes>. Each entry
        # [j, r] is the amount of resource r requested by a replica of job j.
        self._job_resources = np.zeros((len(jobs), len(rtypes)), np.int64)
        for j, job in enumerate(jobs):
            for r, rtype in enumerate(rtypes):
                self._job_resources[j, r] = job.resources.get(rtype, 0)
        # Build array of node resources: <num_nodes> x <num_rtypes>. Each
        # entry [n, r] is the amount of resource r available on node n.
        self._node_resources = np.zeros((len(nodes), len(rtypes)), np.int64)
        for n, node in enumerate(nodes):
            for r, rtype in enumerate(rtypes):
                self._node_resources[n, r] = node.resources.get(rtype, 0)
        # Calculate dominant per-replica resource shares for each job.
        shares = self._job_resources / np.sum(self._node_resources, axis=0)
        self._dominant_share = np.amax(shares, axis=1)
        # Change base goodput to fair-share goodput.
        fair_replicas = np.ceil(1.0 / self._dominant_share / len(self._jobs))
        fair_nodes = np.ceil(len(nodes) * self._dominant_share)
        # self.THR_OBJ = 'srtf' # 'srtf'
        self.THR_OBJ = 'fairness'
        self.power = 1
        self.max_allowed_replicas = sum([job.max_replicas for job in jobs])                
        if self.THR_OBJ == 'fairness':
            from simulator import args 
            self.power = args.power 
            empty_jobs = list() 
            # water filling 
            from simulator import args
            cluster_capacity = args.min_nodes * 4
            assigned_weight_jobs = [job for job in jobs]
            while len(assigned_weight_jobs) > 0: 
                fair_replicas = cluster_capacity / len(assigned_weight_jobs)
                remove_jobs = list() 
                if len(remove_jobs) == 0: 
                    for job in assigned_weight_jobs: 
                        if 'llama' in job.name: 
                            job.fair_replicas = max(fair_replicas // 4 * 4, 4)
                        else: 
                            job.fair_replicas = fair_replicas

                    assigned_weight_jobs = list() 
                
                for job in remove_jobs: 
                    assigned_weight_jobs.remove(job)
                

            for job in jobs: 
                if not hasattr(job.speedup_fn, "_goodput_fn"):
                    empty_jobs.append(job)
                    continue
                fair_replicas = int(math.ceil(job.fair_replicas))
                
                job.fair_replicas = job.max_replicas
                fair_replicas = job.replica_upper_bound


                fair_nodes = fair_replicas // 4
                if fair_replicas % 4 > 0: fair_nodes += 1
                fair_goodput = job.speedup_fn._goodput_fn.optimize(
                        num_nodes=fair_nodes, num_replicas=fair_replicas,
                        max_batch_size=job.speedup_fn._max_batch_size,
                        atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                        accumulation=job.speedup_fn._accumulation)[0]

                effective_throughput_list = list() 
                fair_goodput = 1e-3 
                # import pdb; pdb.set_trace() 
                effective_throughput_list.append((0, 1e-3))
                for i in range(job.replica_lower_bound, job.replica_upper_bound + 1): 

                    if "llama" in job.name and i % 4 != 0: 
                        continue 
                    
                    num_replicas = i 
                    num_nodes = num_replicas // 4 
                    if num_replicas % 4 > 0: num_nodes += 1
                    goodput = job.speedup_fn._goodput_fn.optimize(
                        num_nodes=num_nodes, num_replicas=num_replicas,
                        max_batch_size=job.speedup_fn._max_batch_size,
                        atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                        accumulation=job.speedup_fn._accumulation)[0]
                    effective_throughput = goodput 
                    effective_throughput_list.append((i, effective_throughput))
                    fair_goodput = max(goodput, fair_goodput)
                
                effective_throughput_list = [(gpu, (effective_throughput / fair_goodput)** self.power) for gpu, effective_throughput in effective_throughput_list]
                job.effective_throughput_list = effective_throughput_list 
            
            
            for job in empty_jobs: 
                effective_throughput_list = list() 
                effective_throughput_list.append((0, (1e-3) ** self.power))
                if 'llama' in job.name: 
                    for i in range(4, job.max_replicas//4*4+1, 4): 
                        effective_throughput_list.append((i, i ** self.power))
                else: 
                    for i in range(1, job.max_replicas + 1): 
                        effective_throughput_list.append((i, i ** self.power))
                job.effective_throughput_list = effective_throughput_list 
            job.effective_throughput_list = effective_throughput_list 
        else: 
            raise NotImplementedError 

        self.jobs = jobs 
        self.nodes = nodes 
    
    def solve(self, max_seconds=5): 
        from simulator import args 
        cluster_capacity = args.min_nodes * 4
        # model = Model(solver_name=GRB)
        model = Model(solver_name = CBC)
        var_len = sum([len(job.effective_throughput_list) for job in self.jobs])
        X = [model.add_var(var_type=BINARY) for i in range(var_len)]
        obj_list = list() 
        required_resource_list = list() 
        cnt = 0 
        for job in self.jobs: 
            for gpu, effective_throughput in job.effective_throughput_list: 
                obj_list.append(X[cnt] * effective_throughput) # whether add gpu weight, think for a while 
                required_resource_list.append(gpu)
                cnt += 1
        
        if self.max_allowed_replicas >= cluster_capacity: 
            model += xsum(X[i] * required_resource_list[i] for i in range(var_len)) <= cluster_capacity
        else: 
            # model += xsum(X[i] * required_resource_list[i] for i in range(var_len)) <= cluster_capacity
            model += xsum(X[i] * required_resource_list[i] for i in range(var_len)) <= self.max_allowed_replicas

        # cluster_capacity_lower_bound = min(cluster_capacity)
        # model += xsum(X[i] * required_resource_list[i] for i in range(var_len)) >= cluster_capacity
        cnt = 0
        for job in self.jobs: 
            length = len(job.effective_throughput_list)
            model.add_constr(xsum(X[i+cnt] for i in range(length)) == 1)
            cnt += length 
            


        # model.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list))))
        if self.power < 0: 
            model.objective = minimize(xsum(obj_list[i] for i in range(len(obj_list))))
        else: 
            model.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list))))
        model.optimize()
        cnt = 0 
        allocated_gpu = [0 for _ in range(len(self.jobs))]
        for idx, job in enumerate(self.jobs): 
            length = len(job.effective_throughput_list)
            for i, (gpu, effective_throughput) in enumerate(job.effective_throughput_list): 
                if X[i+cnt].x > 0.5:
                    allocated_gpu[idx] = gpu
                    if 'llama' in job.name and gpu < 4 and gpu > 0: 
                        import pdb; pdb.set_trace() 
                    # if job.name == 'cifar10-ResNet50-39': 
                    #     import pdb; pdb.set_trace() 
                    
            cnt += length

        # if (sum(allocated_gpu)) == 0 and len(allocated_gpu) >= 2: 
        #     import pdb; pdb.set_trace()
        #     [job.name for job in self.jobs]
        #     [job.max_replicas for job in self.jobs]
        #     print([X[i].x for i in range(len(X))])
        #     self.jobs[0].effective_throughput_list
        return allocated_gpu


        
