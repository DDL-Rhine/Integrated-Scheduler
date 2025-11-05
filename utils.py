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


class JobInfo(object):
    def __init__(self, name, resources, speedup_fn, creation_timestamp, attained_service,
                min_replicas, max_replicas, staying_time, deserved_service=None, prior_weight=1, max_node_count=-1, preemptible=True, \
                benefit_func=None, progress=0, max_progress=0, frozen_alpha=0, frozen_layer=-1, total_layer=-1, replica_lower_bound=0, replica_upper_bound=0):
        """
        Args:
            resources (dict): Requested resources (eg. GPUs) of each replica.
            speedup_fn (SpeedupFunction): Speedup function for this job.
            creation_timestamp (datetime): Time when this job was created.
            min_replicas (int): Minimum number of replicas job's guaranteed.
            max_replicas (int): Maximum number of replicas. Maximum should be
                                greater or equal to Minimum
            preemptible (bool): Is the job preemptible?
        """
        assert max_replicas > 0
        assert max_replicas >= min_replicas
        self.name = name
        self.resources = resources
        self.speedup_fn = speedup_fn
        self.creation_timestamp = creation_timestamp
        self.attained_service = attained_service
        self.deserved_service = deserved_service
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas
        self.preemptible = preemptible
        self.benefit_func = benefit_func
        self.max_node_count = max_node_count 
        # new added
        self.max_progress = max_progress 
        self.progress = progress
        self.frozen_alpha = frozen_alpha
        self.frozen_layer = frozen_layer 
        self.total_layer = total_layer 
        self.prior_weight = prior_weight
        self.staying_time = staying_time
        
        self.replica_lower_bound = replica_lower_bound
        self.replica_upper_bound = replica_upper_bound


class NodeInfo(object):
    def __init__(self, resources, preemptible):
        """
        Args:
            resources (dict): Available resources (eg. GPUs) on this node.
            preemptible (bool): Whether this node is pre-emptible.
        """
        self.resources = resources
        self.preemptible = preemptible
