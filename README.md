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


<li>
<details><summary><b>DIANNE- Distributed Artificial Neural Networks for the Internet of Things</b></summary>
<p>
	
##### DIANNE middleware framework is presented that is optimized for single sample feed-forward execution and facilitates distributing artificial neural networks across multiple IoT devices

Cloud is often the natural choice to train and evaluate neural networks, benefiting from the huge compute power and scalability, but IoT applications with sensors sending a continuous stream of data, 
the Cloud introduces additional complications:
- connection to the Cloud is required at all times, having to deal with limitations in **bandwidth and a high and variable latency**
- sending sensor data to the Cloud may introduce **security holes and privacy issues**


_**The first experiment’s results**_  prove that large neural networks, which can not fit on small embedded devices, can benefit from distributing the slow convolutional modules to other devices in the IoT environment preferable equipped with GPU acceleration. 
_**The second experiment**_ shows that the DIANNE middleware performs excellently on GPU accelerated devices, outperforming all tested frameworks when only a single image is forwarded through the network.

**Result**:
DIANNE actually performs on par or better than the other frameworks.


</p>
</details>
</li>

<li>
<details><summary><b>Optimizing Network Performance in Distributed Machine Learning</b></summary>
<p>
	
##### MLNET, a host-based communication layer that aims to improve the network performance of distributed machine learning systems through a combination of _traffic reduction_ techniques (to diminish network load in the core and at the edges) and _traffic management_ (to reduce average training time).

MLNET inherits the standard commu- nication APIs from the Parameter Server
- Distributed Aggregation and Multicast
- Network Prioritization

**Result**:
overall training time can be reduced by up to 78%. 

</p>
</details>
</li>

<li>
<details><summary><b>Open Fabric for Deep Learning Models</b></summary>
<p>
	
##### The FfDL platform uses a microservices architecture to reduce coupling between components, keep each component simple and as stateless as possible, isolate component failures, and allow each component to be developed, tested, deployed, scaled, and upgraded independently. 

**Tools Used**:
- **Adversarial Robustness Toolbox _(ART)_**: To provide robustness for models
- **AI Fairness 360 toolkit _(AIF360)_**: to find and remove bias in datasets and models
- **Model Asset Exchange _(MAX)_**: an app store to discover, share and rate models

Differences between the projects in terms of _job scheduling and distribution, framework support, ecosystem and general architecture_.

</p>
</details>
</li>


</ul>

# Articles 📖:

- **[Distributed Deep Learning, Part 1: An Introduction to Distributed Training of Neural Networks](https://blog.skymind.ai/distributed-deep-learning-part-1-an-introduction-to-distributed-training-of-neural-networks/)**
- **[Implementing the DistBelief Deep Neural Network Training Framework with Akka](http://alexminnaar.com/implementing-the-distbelief-deep-neural-network-training-framework-with-akka.html)**


# Software 👨‍💻:

### Parallel and Distributed Frameworks:

- **[AkkaDistBelief](https://github.com/alexminnaar/AkkaDistBelief)**: DistBelief is a framework for training deep neural networks with a cluster of machines rather than GPUs _**(Scala)**_ | Google DistBelief Net

### Decentralized Distributed Deep Dearning:

- **[decentralizedsgd](https://github.com/tgaddair/decentralizedsgd)**: performing distributed training of machine learning models over a cluster of machines in parallel _**(Scale)**_
- **[TF.AKO](https://www-users.cs.umn.edu/~chandra/tfako/home.html)**: ([GitHub](https://github.com/mesh-umn/TF.AKO)): Decentralised Deep Learning with Partial Gradient Exchange

### Deep Learning Platform:

- **[FfDL](https://github.com/IBM/FfDL)**: Deep Learning Platform offering TensorFlow, Caffe, PyTorch etc. as a Service on Kubernetes
- **[kubeflow](https://github.com/kubeflow/kubeflow)**: Machine Learning Toolkit for Kubernetes

### Distributed Networks:
- **[DIANNE](http://dianne.intec.ugent.be/)**: DIstributed Artificial Neural NEtworks ([GitHub](https://github.com/ibcn-cloudlet/dianne))
