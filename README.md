<h1 align=center> KayDiDL </h1>

# Papers 📜:

<ul>

<li>
<details><summary><b>Parallel and Distributed Deep Learning</b></summary>
<p>
	
##### Analysis (empirically) the speedup in training a CNN using conventional _*single core CPU*_ and _*GPU*_ and provide practical suggestions to improve training times.

- **Synchronous Update Methods**: {Parallel SGD, Alternating Direction Method of Multipliers SGD (ADMM.SGD)}
- **Asynchronous Update Methods**: {Downpour SGD}

</p>
</details>
</li>


<li>
<details><summary><b>Decentralized Distributed Deep Learning in Heterogeneous WAN Environments</b></summary>
<p>
	
##### decentralized distributed deep learning framework for such heterogeneous WAN-based infrastructures
**The framework dynamically and automatically adjusts**:
- The frequency of parameter sharing
- The size of parameters shared depending on individual network bandwidth and data processing power
- Introduces a new scaling factor to control the degree of contribution to parameter updates by considering the amount of data trained during unit time in each device

**Result**:
_**Sharing small size of parameters (partial params)**_ is more effective to increase the accuracy faster when machines are highly network bandwidth-constrained during training.

</p>
</details>
</li>


<li>
<details><summary><b>Orchestrating Deep Learning Workloads on Distributed Infrastructure</b></summary>
<p>
	
#####  Deep Learning Workloads requirements to support GPUs in container management systems and describe solutions in Kubernetes

**Issues**:
- GPU’s are _**unique**_ quantities (GPU 0, GPU 1, ...) and they must be allocated accordingly
- _**GPU topology**_, will heavily affect the bandwidth of _(GPU to GPU communication)_
- GPU topology even affects GPU capabil- ities. In some systems, for example, GPUs on different CPU socket cannot have Peer to Peer communication capability.

**Solutions**:
- [x] Enabled GPU support on Kubernetes
- [x] Implemented GPU allocator module:
	- record GPU number-to-device mapping
	- maps the number to actual GPU devices according to required scheduling policy and expose the allocated GPUs to application inside the container
- [x] Developed **two** advanced GPU schedulers:
	- _bin-packing scheduler_: tries to bundle GPU jobs to fewer servers, so that other idle servers can be reserved for potentially large jobs
	- _topology-aware scheduler_:  automatically collect GPU topology informa- tion of each worker node, and assign nodes that deliver the highest possible bandwidth to the application
- [x] Enhanced Kubernetes to gather the device drivers on kubelet startup and mount these drivers into the container automatically
- [x] Enabled GPU liveness check on Kubernetes 
- [x] Added GPU quota support in Kubernetes => support multiple users

</p>
</details>
</li>

<li>
<details><summary><b>Decentralized and Distributed Machine Learning Model Training with Actors</b></summary>
<p>
	
##### Explore a more experimental form of _decentralized training_ that removes bottleneck{centralized parameter server introduces a bottleneck and single-point of failure during training}

**Actor-Based Concurrency Model**:
> implemented as actors using the Akka2 actor framework written in Scala, based off the work done by Alex Minnaar3 in implementing Google’s DistBelief framework in Akka

**Network Architecture**: => XOR
- **Asynchronous centralized training**: attain similar accuracy with much higher throughput by using soft synchronization
- **Fully asynchronous and decentralized training**: net the greatest overall training speed, but at a cost to model accuracy. _(This cost is configurable based on the setting of **τ**)_

_**New Keywords**_:
- **threshold parameter (τ)**: tunes the frequency with which updates are sent out to all other data shards in the system.
- **gradient residual​**:


</p>
</details>
</li>


</ul>

# Frameworks 👨‍💻:

- **[TF.AKO](https://www-users.cs.umn.edu/~chandra/tfako/home.html)**: ([GitHub](https://github.com/mesh-umn/TF.AKO))
