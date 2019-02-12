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
	
##### decentralized distributed deep learning framework for such heterogeneous WAN-based in- frastructures
**The framework dynamically and automatically adjusts**:
- The frequency of parameter sharing
- The size of parameters shared depending on individual network bandwidth and data processing power
- Introduces a new scaling factor to control the degree of contribution to parameter updates by considering the amount of data trained during unit time in each device

**Result**:
_**Sharing small size of parameters (partial params)**_ is more effective to increase the accuracy faster when machines are highly network bandwidth-constrained during training.

</p>
</details>
</li>


</ul>

# Frameworks 👨‍💻:

- **[TF.AKO](https://www-users.cs.umn.edu/~chandra/tfako/home.html)**: ([GitHub](https://github.com/mesh-umn/TF.AKO))
