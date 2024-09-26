---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

Hi there! 

Welcome to Xinyin Ma(马欣尹)’s website!
I am currently a Ph.D candidate in [Learning and Vision Lab @ NUS](http://www.lv-nus.org/) from August 2022, advised by [Prof.Xinchao Wang](https://sites.google.com/site/sitexinchaowang/). Previously I obtained my master degree in computer science from Zhejiang University where I was advised by [Prof.Weiming Lu](https://person.zju.edu.cn/en/lwm). I obtained my bachelor degree in software engineering also in Zhejiang University and got the honor degree from Chu Kochen Honors College.

I’m currently conducting some research in efficient learning, including:  
🌲 The efficiency of the Language Model and Diffusion Model.  
🌱 The acceleration of training: dataset distillation and coreset  
🌿 Compression with synthetic data, e.g., data-free distillation. 

I have published several papers in NeurIPS, CVPR, EMNLP, IJCAI. You can find more information about my publications in [Google Scholar](https://scholar.google.com/citations?user=jFUKS0oAAAAJ&hl=en)

**I‘m actively seeking internship and visiting opportunities. If you have any opportunities available, I would greatly appreciate it if you could reach out to me. Thank you😎!**


# 🔥 News
- *2024.09*: &nbsp;Four papers ([learning-to-cache](https://arxiv.org/abs/2406.01733), [AsyncDiff](https://arxiv.org/abs/2406.06911), [SlimSAM](https://arxiv.org/abs/2312.05284) and RemixDiT) accepted by NeurIPS'24!
- *2024.07*: &nbsp;One co-authored paper accepted by ECCV'24! 
- *2024.07*: &nbsp;⛵Passed my qualifying exam! 
- *2024.06*: &nbsp;We announce our new work, Learning-to-Cache, an improved version of DeepCache on the diffusion transformers. Check our [paper](https://arxiv.org/abs/2406.01733) and [code](https://github.com/horseee/learning-to-cache)! Also introduce another new work AsyncDiff, which seperates diffusion models into different devices to enable parallel computing. [Paper](https://arxiv.org/abs/2406.06911) and [code](https://github.com/czg1225/AsyncDiff) is available!
- *2024.06*: &nbsp;One co-authored paper accepted by Interspeech'24! 
- *2024.02*：&nbsp;DeepCache is accepted by CVPR'24! 
- *2023.12*: &nbsp;🌟Our new work, DeepCache, accelerates Diffusion Models for FREE! Check our [paper](https://arxiv.org/abs/2312.00858) and [code](https://github.com/horseee/DeepCache)! 
- *2023.09*: &nbsp;Two papers are accepted by NeurIPS'23. 
- *2023.06*: &nbsp;🎉🎉 Release LLM-Pruner🐏, the first structural pruning work of LLM. See our [paper](https://arxiv.org/abs/2305.11627) and [code](https://github.com/horseee/LLM-Pruner)! 
- *2023.02*: &nbsp;One paper 'DepGraph: Towards Any Structural Pruning' accepted by CVPR’23.
- *2022.08*: &nbsp;⛵Start my Ph.D. journey in NUS!
- *2022.04*: &nbsp; One paper ‘Prompting to distill: Boosting Data-Free Knowledge Distillation via Reinforced Prompt’ accepted by IJCAI’22.
- *2022.04*: &nbsp; Got my master degree from ZJU! Thanks to my supervisor and all my friends in ZJU!

# 📝 Publications 


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2024</div><img src='https://github.com/horseee/learning-to-cache/raw/main/assets/teaser.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching**](- A training-free paradigm that accelerates diffusion models
) <img src='https://img.shields.io/github/stars/horseee/Learning-to-Cache.svg?style=social&label=Star' alt="sym" height="100%">

**Xinyin Ma**, Gongfan Fang, Michael Bi Mi, Xinchao Wang

- A novel scheme that learns to conduct caching in a dynamic manner for diffusion transformers.
- A large proportion of layers in the diffusion transformer can be removed, without updating the model parameters.
- Learning-to-Cache largely outperforms samplers such as DDIM and DPM-Solver.

<div style="display: inline">
    <a href="https://arxiv.org/abs/2406.01733"> <strong>[paper]</strong></a>
    <a href="https://github.com/horseee/learning-to-cache"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Diffusion Transformers have recently demonstrated unprecedented generative capabilities for various tasks. The encouraging results, however, come with the cost of slow inference, since each denoising step requires inference on a transformer model with a large scale of parameters. In this study, we make an interesting and somehow surprising observation: the computation of a large proportion of layers in the diffusion transformer, through introducing a caching mechanism, can be readily removed even without updating the model parameters. In the case of U-ViT-H/2, for example, we may remove up to 93.68% of the computation in the cache steps (46.84% for all steps), with less than 0.01 drop in FID. To achieve this, we introduce a novel scheme, named Learning-to-Cache (L2C), that learns to conduct caching in a dynamic manner for diffusion transformers. Specifically, by leveraging the identical structure of layers in transformers and the sequential nature of diffusion, we explore redundant computations between timesteps by treating each layer as the fundamental unit for caching. To address the challenge of the exponential search space in deep models for identifying layers to cache and remove, we propose a novel differentiable optimization objective. An input-invariant yet timestep-variant router is then optimized, which can finally produce a static computation graph. Experimental results show that L2C largely outperforms samplers such as DDIM and DPM-Solver, alongside prior cache-based methods at the same inference speed. </p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2024</div><img src='https://github.com/horseee/DeepCache/blob/master/assets/intro.png?raw=true' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**DeepCache: Accelerating Diffusion Models for Free**](https://arxiv.org/abs/2312.00858) <img src='https://img.shields.io/github/stars/horseee/DeepCache.svg?style=social&label=Star' alt="sym" height="100%">

**Xinyin Ma**, Gongfan Fang, Xinchao Wang

- A training-free paradigm that accelerates diffusion models
- Utilizes the U-Net's properties to efficiently reuse high-level features and update low-level features
- 2.3× speedup for Stable Diffusion v1.5 and a 4.1× speedup for LDM-4-G, based upon DDIM/PLMS

<div style="display: inline">
    <a href="https://arxiv.org/abs/2312.00858"> <strong>[paper]</strong></a>
    <a href="https://github.com/horseee/DeepCache"> <strong>[code]</strong></a>
    <a href="https://horseee.github.io/Diffusion_DeepCache/"> <strong>[Project Page]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Diffusion models have recently gained unprecedented attention in the field of image synthesis due to their remarkable generative capabilities. Notwithstanding their prowess, these models often incur substantial computational costs, primarily attributed to the sequential denoising process and cumbersome model size. Traditional methods for compressing diffusion models typically involve extensive retraining, presenting cost and feasibility challenges. In this paper, we introduce DeepCache, a novel training-free paradigm that accelerates diffusion models from the perspective of model architecture. DeepCache capitalizes on the inherent temporal redundancy observed in the sequential denoising steps of diffusion models, which caches and retrieves features across adjacent denoising stages, thereby curtailing redundant computations. Utilizing the property of the U-Net, we reuse the high-level features while updating the low-level features in a very cheap way. This innovative strategy, in turn, enables a speedup factor of 2.3× for Stable Diffusion v1.5 with only a 0.05 decline in CLIP Score, and 4.1× for LDM-4-G with a slight decrease of 0.22 in FID on ImageNet. Our experiments also demonstrate DeepCache's superiority over existing pruning and distillation methods that necessitate retraining and its compatibility with current sampling techniques. Furthermore, we find that under the same throughput, DeepCache effectively achieves comparable or even marginally improved results with DDIM or PLMS. </p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2023</div><img src='images/papers/llm-pruner.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**LLM-Pruner: On the Structural Pruning of Large Language Models**](https://arxiv.org/abs/2305.11627) <img src='https://img.shields.io/github/stars/horseee/LLM-Pruner.svg?style=social&label=Star' alt="sym" height="100%">

**Xinyin Ma**, Gongfan Fang, Xinchao Wang

- Task-agnostic Compression: The compressed LLM retain its multi-task ability.
- Less Training Corpus: We use only 50k samples to post-train the LLM.
- Efficient Compression: 3 minutes for pruning and 3 hours for post-training. 
- Automatic Structural Pruning: Pruning new LLMs with minimal human effort.

<div style="display: inline">
    <a href="https://arxiv.org/abs/2305.11627"> <strong>[paper]</strong></a>
    <a href="https://github.com/horseee/LLM-Pruner"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Large language models (LLMs) have shown remarkable capabilities in language understanding and generation. However, such impressive capability typically comes with a substantial model size, which presents significant challenges in both the deployment, inference, and training stages. With LLM being a general-purpose task solver, we explore its compression in a task-agnostic manner, which aims to preserve the multi-task solving and language generation ability of the original LLM. One challenge to achieving this is the enormous size of the training corpus of LLM, which makes both data transfer and model post-training over-burdensome. Thus, we tackle the compression of LLMs within the bound of two constraints: being task-agnostic and minimizing the reliance on the original training dataset. Our method, named LLM-Pruner, adopts structural pruning that selectively removes non-critical coupled structures based on gradient information, maximally preserving the majority of the LLM's functionality. To this end, the performance of pruned models can be efficiently recovered through tuning techniques, LoRA, in merely 3 hours, requiring only 50K data. We validate the LLM-Pruner on three LLMs, including LLaMA, Vicuna, and ChatGLM, and demonstrate that the compressed models still exhibit satisfactory capabilities in zero-shot classification and generation. </p>
    </div>
</div>

</div>
</div>



<ul>
  <li>
   <a href="https://www.ijcai.org/proceedings/2022/0596.pdf"> Prompting to distill: Boosting Data-Free Knowledge Distillation via Reinforced Prompt</a>. <strong>Xinyin Ma</strong>, Xinchao Wang, Gongfan Fang, Yongliang Shen, Weiming Lu. <strong>IJCAI 2022</strong>. 
    <div style="display: inline">
        <a href="https://www.ijcai.org/proceedings/2022/0596.pdf"> [paper]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Data-free knowledge distillation (DFKD) conducts knowledge distillation via eliminating the dependence of original training data, and has recently achieved impressive results in accelerating pre-trained language models. At the heart of DFKD is toreconstruct a synthetic dataset by invertingthe parameters of the uncompressed model. Prior DFKD approaches, however, havelargely relied on hand-crafted priors of the target data distribution for the reconstruction, which can be inevitably biased and often incompetent to capture the intrinsic distributions. To address this problem, we propose a prompt-based method, termed as PromptDFD, that allows us to take advantage of learned language priors, which effectively harmonizes the synthetic sentences to be semantically and grammatically correct. Specifically, PromptDFD leverages a pre-trained generative model to provide language priors and introduces a reinforced topic prompter to control data synthesis, making the generated samples thematically relevant and semantically plausible, and thus friendly to downstream tasks. As shown in our experiments, the proposed method substantially improves the synthesis quality and achieves considerable improvements on distillation performance. In some cases, PromptDFD even gives rise to results on par with those from the data-driven knowledge distillation with access to the original training data. </p>
        </div>
    </div>
  </li>
  
  <li>
   <a href="https://aclanthology.org/2021.emnlp-main.205.pdf"> MuVER: Improving First-Stage Entity Retrieval with Multi-View Entity Representations</a>. <strong>Xinyin Ma</strong>, Yong Jiang, Nguyen Bach, Tao Wang, Zhongqiang Huang, Fei Huang, Weiming Lu. <strong>EMNLP 2021(short)</strong>. 
    <div style="display: inline">
        <a href="https://aclanthology.org/2021.emnlp-main.205.pdf"> [paper]</a>
        <a href="https://github.com/alibaba-nlp/muver"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Entity retrieval, which aims at disambiguating mentions to canonical entities from massive KBs, is essential for many tasks in natural language processing. Recent progress in entity retrieval shows that the dual-encoder structure is a powerful and efficient framework to nominate candidates if entities are only identified by descriptions. However, they ignore the property that meanings of entity mentions diverge in different contexts and are related to various portions of descriptions, which are treated equally in previous works. In this work, we propose Multi-View Entity Representations (MuVER), a novel approach for entity retrieval that constructs multi-view representations for entity descriptions and approximates the optimal view for mentions via a heuristic searching method. Our method achieves the state-of-the-art performance on ZESHEL and improves the quality of candidates on three standard Entity Linking datasets. </p>
        </div>
    </div>
  </li>

  <li>
   <a href="https://aclanthology.org/2020.emnlp-main.499.pdf"> Adversarial Self-Supervised Data-Free Distillation for Text Classification</a>. <strong>Xinyin Ma</strong>, Yongliang Shen, Gongfan Fang, Chen Chen, Chenghao Jia, Weiming Lu. <strong>EMNLP 2020</strong>. 
    <div style="display: inline">
        <a href="https://aclanthology.org/2020.emnlp-main.499.pdf"> [paper]</a>
        <a href="https://slideslive.com/38938706/adversarial-selfsupervised-datafree-distillation-for-text-classification"> [video]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Large pre-trained transformer-based language models have achieved impressive results on a wide range of NLP tasks. In the past few years, Knowledge Distillation(KD) has become a popular paradigm to compress a computationally expensive model to a resource-efficient lightweight model. However, most KD algorithms, especially in NLP, rely on the accessibility of the original training dataset, which may be unavailable due to privacy issues. To tackle this problem, we propose a novel two-stage data-free distillation method, named Adversarial self-Supervised Data-Free Distillation (AS-DFD), which is designed for compressing large-scale transformer-based models (e.g., BERT). To avoid text generation in discrete space, we introduce a Plug & Play Embedding Guessing method to craft pseudo embeddings from the teacher’s hidden knowledge. Meanwhile, with a self-supervised module to quantify the student’s ability, we adapt the difficulty of pseudo embeddings in an adversarial training manner. To the best of our knowledge, our framework is the first data-free distillation framework designed for NLP tasks. We verify the effectiveness of our method on several text classification datasets. </p>
        </div>
    </div>
  </li>

  <li>
    <a href="https://arxiv.org/abs/2406.06911"> AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoising</a>. Zigeng Chen, <strong>Xinyin Ma</strong>, Gongfan Fang, Zhenxiong Tan, Xinchao Wang. <strong>NeurIPS 2024</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2406.06911"> [paper]</a>
        <a href="https://github.com/czg1225/AsyncDiff"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Diffusion models have garnered significant interest from the community for their great generative ability across various applications. However, their typical multi-step sequential-denoising nature gives rise to high cumulative latency, thereby precluding the possibilities of parallel computation. To address this, we introduce AsyncDiff, a universal and plug-and-play acceleration scheme that enables model parallelism across multiple devices. Our approach divides the cumbersome noise prediction model into multiple components, assigning each to a different device. To break the dependency chain between these components, it transforms the conventional sequential denoising into an asynchronous process by exploiting the high similarity between hidden states in consecutive diffusion steps. Consequently, each component is facilitated to compute in parallel on separate devices. The proposed strategy significantly reduces inference latency while minimally impacting the generative quality. Specifically, for the Stable Diffusion v2.1, AsyncDiff achieves a 2.7x speedup with negligible degradation and a 4.0x speedup with only a slight reduction of 0.38 in CLIP Score, on four NVIDIA A5000 GPUs. Our experiments also demonstrate that AsyncDiff can be readily applied to video diffusion models with encouraging performances. </p>
        </div>
    </div>
  </li>

  <li>
    <a href="https://arxiv.org/abs/2312.05284"> SlimSAM: 0.1% Data Makes Segment Anything Slim</a>. Zigeng Chen, Gongfan Fang, <strong>Xinyin Ma</strong>, Xinchao Wang. <strong>NeurIPS 2024</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2312.05284"> [paper]</a>
        <a href="https://github.com/czg1225/SlimSAM"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Current approaches for compressing the Segment Anything Model (SAM) yield commendable results, yet necessitate extensive data to train a new network from scratch. Employing conventional pruning techniques can remarkably reduce data requirements but would suffer from a degradation in performance. To address this challenging trade-off, we introduce SlimSAM, a novel data-efficient SAM compression method that achieves superior performance with extremely less training data. The essence of SlimSAM is encapsulated in the alternate slimming framework which effectively enhances knowledge inheritance under severely limited training data availability and exceptional pruning ratio. Diverging from prior techniques, our framework progressively compresses the model by alternately pruning and distilling distinct, decoupled sub-structures. Disturbed Taylor pruning is also proposed to address the misalignment between the pruning objective and training target, thereby boosting the post-distillation after pruning. SlimSAM yields significant performance improvements while demanding over 10 times less training data than any other existing compression methods. Even when compared to the original SAM, SlimSAM achieves approaching performance while reducing parameter counts to merely 1.4% (9.1M), MACs to 0.8% (23G), and requiring only 0.1% (10k) of the SAM training data. </p>
        </div>
    </div>
  </li>

  <li>
    Remix-DiT: Mixing Diffusion Transformers for Multi-Expert Denoising. Gongfan Fang, <strong>Xinyin Ma</strong>,  Xinchao Wang. <strong>NeurIPS 2024</strong>. 
  </li>

  <li>
    <a href="https://arxiv.org/abs/2407.04616"> Isomorphic Pruning for Vision Models</a>. Gongfan Fang, <strong>Xinyin Ma</strong>, Michael Bi Mi, Xinchao Wang. <strong>ECCV 2024</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2407.04616"> [paper]</a>
        <a href="https://github.com/VainF/Isomorphic-Pruning"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Structured pruning reduces the computational overhead of deep neural networks by removing redundant sub-structures. However, assessing the relative importance of different sub-structures remains a significant challenge, particularly in advanced vision models featuring novel mechanisms and architectures like self-attention, depth-wise convolutions, or residual connections. These heterogeneous substructures usually exhibit diverged parameter scales, weight distributions, and computational topology, introducing considerable difficulty to importance comparison. To overcome this, we present Isomorphic Pruning, a simple approach that demonstrates effectiveness across a range of network architectures such as Vision Transformers and CNNs, and delivers competitive performance across different model sizes. Isomorphic Pruning originates from an observation that, when evaluated under a pre-defined importance criterion, heterogeneous sub-structures demonstrate significant divergence in their importance distribution, as opposed to isomorphic structures that present similar importance patterns. This inspires us to perform isolated ranking and comparison on different types of sub-structures for more reliable pruning. Our empirical results on ImageNet-1K demonstrate that Isomorphic Pruning surpasses several pruning baselines dedicatedly designed for Transformers or CNNs. For instance, we improve the accuracy of DeiT-Tiny from 74.52% to 77.50% by pruning an off-the-shelf DeiT-Base model. And for ConvNext-Tiny, we enhanced performance from 82.06% to 82.18%, while reducing the number of parameters and memory usage.  </p>
        </div>
    </div>
  </li>

  <li>
    <a href="https://arxiv.org/abs/2407.10468"> LiteFocus: Accelerated Diffusion Inference for Long Audio Synthesis</a>. Zhenxiong Tan, <strong>Xinyin Ma</strong>, Gongfan Fang, Xinchao Wang. <strong>Interspeech 2024</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2407.10468"> [paper]</a>
        <a href="https://github.com/Yuanshi9815/LiteFocus"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Latent diffusion models have shown promising results in audio generation, making notable advancements over traditional methods. However, their performance, while impressive with short audio clips, faces challenges when extended to longer audio sequences. These challenges are due to model's self-attention mechanism and training predominantly on 10-second clips, which complicates the extension to longer audio without adaptation. In response to these issues, we introduce a novel approach, LiteFocus that enhances the inference of existing audio latent diffusion models in long audio synthesis. Observed the attention pattern in self-attention, we employ a dual sparse form for attention calculation, designated as same-frequency focus and cross-frequency compensation, which curtails the attention computation under same-frequency constraints, while enhancing audio quality through cross-frequency refillment. LiteFocus demonstrates substantial reduction on inference time with diffusion-based TTA model by 1.99x in synthesizing 80-second audio clips while also obtaining improved audio quality. </p>
        </div>
    </div>
  </li>

  <li>
    <img src='https://img.shields.io/github/stars/VainF/Torch-Pruning.svg?style=social&label=Star' alt="sym" height="100%">
    <a href="https://arxiv.org/abs/2301.12900"> DepGraph: Towards Any Structural Pruning</a>. Gongfan Fang, <strong>Xinyin Ma</strong>, Mingli Song, Michael Bi Mi, Xinchao Wang. <strong>CVPR 2023</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2301.12900"> [paper]</a>
        <a href="https://github.com/VainF/Torch-Pruning"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Structural pruning enables model acceleration by removing structurally-grouped parameters from neural networks. However, the parameter-grouping patterns vary widely across different models, making architecture-specific pruners, which rely on manually-designed grouping schemes, non-generalizable to new architectures. In this work, we study a highly-challenging yet barely-explored task, any structural pruning, to tackle general structural pruning of arbitrary architecture like CNNs, RNNs, GNNs and Transformers. The most prominent obstacle towards this goal lies in the structural coupling, which not only forces different layers to be pruned simultaneously, but also expects all removed parameters to be consistently unimportant, thereby avoiding structural issues and significant performance degradation after pruning. To address this problem, we propose a general and fully automatic method, Dependency Graph(DepGraph), to explicitly model the dependency between layers and comprehensively group coupled parameters for pruning. In this work, we extensively evaluate our method on several architectures and tasks, including ResNe(X)t, DenseNet, MobileNet and Vision transformer for images, GAT for graph, DGCNN for 3D point cloud, alongside LSTM for language, and demonstrate that, even with a simple norm-based criterion, the proposed method consistently yields gratifying performances. </p>
        </div>
    </div>
  </li>

  <li>
    <a href="https://arxiv.org/abs/2305.10924"> Structural Pruning for Diffusion Models</a>. Gongfan Fang, <strong>Xinyin Ma</strong>, Xinchao Wang. <strong>NeurIPS 2023</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2305.10924"> [paper]</a>
        <a href="https://github.com/VainF/Diff-Pruning"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Generative modeling has recently undergone remarkable advancements, primarily propelled by the transformative implications of Diffusion Probabilistic Models (DPMs). The impressive capability of these models, however, often entails significant computational overhead during both training and inference. To tackle this challenge, we present Diff-Pruning, an efficient compression method tailored for learning lightweight diffusion models from pre-existing ones, without the need for extensive re-training. The essence of Diff-Pruning is encapsulated in a Taylor expansion over pruned timesteps, a process that disregards non-contributory diffusion steps and ensembles informative gradients to identify important weights. Our empirical assessment, undertaken across four diverse datasets highlights two primary benefits of our proposed method: 1) Efficiency: it enables approximately a 50% reduction in FLOPs at a mere 10% to 20% of the original training expenditure; 2) Consistency: the pruned diffusion models inherently preserve generative behavior congruent with their pre-trained progenitors. </p>
        </div>
    </div>
  </li>

  <li>
   <a href="https://aclanthology.org/2021.acl-long.216.pdf"> A Locate and Label: A Two-stage Identifier for Nested Named Entity Recognition</a>. Yongliang Shen, <strong>Xinyin Ma</strong>, Zeqi Tan, Shuai Zhang, Wen Wang, Weiming Lu. <strong>ACL 2021</strong>. 
    <div style="display: inline">
        <a href="https://aclanthology.org/2021.acl-long.216.pdf"> [paper]</a>
        <a href="https://github.com/tricktreat/locate-and-label"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Named entity recognition (NER) is a well-studied task in natural language processing. Traditional NER research only deals with flat entities and ignores nested entities. The span-based methods treat entity recognition as a span classification task. Although these methods have the innate ability to handle nested NER, they suffer from high computational cost, ignorance of boundary information, under-utilization of the spans that partially match with entities, and difficulties in long entity recognition. To tackle these issues, we propose a two-stage entity identifier. First we generate span proposals by filtering and boundary regression on the seed spans to locate the entities, and then label the boundary-adjusted span proposals with the corresponding categories. Our method effectively utilizes the boundary information of entities and partially matched spans during training. Through boundary regression, entities of any length can be covered theoretically, which improves the ability to recognize long entities. In addition, many low-quality seed spans are filtered out in the first stage, which reduces the time complexity of inference. Experiments on nested NER datasets demonstrate that our proposed method outperforms previous state-of-the-art models. </p>
        </div>
    </div>
  </li>

  <li>
   <a href="https://dl.acm.org/doi/abs/10.1145/3442381.3449895"> A Trigger-Sense Memory Flow Framework for Joint Entity and Relation Extraction</a>. Yongliang Shen, <strong>Xinyin Ma</strong>, Yechun Tang, Weiming Lu. <strong>WWW 2021</strong>. 
    <div style="display: inline">
        <a href="https://dl.acm.org/doi/abs/10.1145/3442381.3449895"> [paper]</a>
        <a href="https://github.com/tricktreat/trimf"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Joint entity and relation extraction framework constructs a unified model to perform entity recognition and relation extraction simultaneously, which can exploit the dependency between the two tasks to mitigate the error propagation problem suffered by the pipeline model. Current efforts on joint entity and relation extraction focus on enhancing the interaction between entity recognition and relation extraction through parameter sharing, joint decoding, or other ad-hoc tricks (e.g., modeled as a semi-Markov decision process, cast as a multi-round reading comprehension task). However, there are still two issues on the table. First, the interaction utilized by most methods is still weak and uni-directional, which is unable to model the mutual dependency between the two tasks. Second, relation triggers are ignored by most methods, which can help explain why humans would extract a relation in the sentence. They’re essential for relation extraction but overlooked. To this end, we present a Trigger-Sense Memory Flow Framework (TriMF) for joint entity and relation extraction. We build a memory module to remember category representations learned in entity recognition and relation extraction tasks. And based on it, we design a multi-level memory flow attention mechanism to enhance the bi-directional interaction between entity recognition and relation extraction. Moreover, without any human annotations, our model can enhance relation trigger information in a sentence through a trigger sensor module, which improves the model performance and makes model predictions with better interpretation. Experiment results show that our proposed framework achieves state-of-the-art results by improves the relation F1 to 52.44% (+3.2%) on SciERC, 66.49% (+4.9%) on ACE05, 72.35% (+0.6%) on CoNLL04 and 80.66% (+2.3%) on ADE. </p>
        </div>
    </div>
  </li>  
  
</ul>




# 🎖 Honors and Awards 
- *2019-2022(M.Eng.)*: Outstanding Graduate(2022), Tecent Scholarship(2021), CETC28 Scholarship(2021), Huawei Elite Scholarship(2020), Shenzhen Stock Exchange Scholarship(2020), Award of Honor for Graduate(2021, 2020)
- *2015-2019(B.Eng.)*: Outstanding Engineer Scholarship (2018), Outstanding Student of Zhejiang University (2018, 2017, 2016), Second-Class Academic Scholarship of Zhejiang University (2017, 2016), Second Class Scholarship of National Talent Training Base (2017), CASC Second Class Scholarship (2016)

# 📖 Educations
- *2022.08 - (now)*, Ph.D. Student in Electrical and Computer Engineering, College of Design and Engineering, National University of Singapore
- *2019.08 - 2022.04*, M.Eng. in Computer Science, College of Computer Science and Technology, Zhejiang University
- *2015.09 - 2019.06*, B.Eng. in Software Engineering, Chu Kochen Honors College, Zhejiang University

# 📋 Academic Service
- Conference: NeurIPS'24, EMNLP'24(ARR'24 June), ECCV'24, ACL'24(ARR'24 Feb), ICML'24, IJCAI'24, ICLR'24, NAACL'24(ARR'23 Dec), NeurIPS’23, EMNLP'23, ICML’23, ACL’23, EMNLP’22, ACL’22, EMNLP’21, ACL’21 and several ARRs
- Journal: JVCI

# 🔭 Teaching Experience
- Fall 2023, Spring 2023. TA for EE2211, Introduction to Machine Learning, NUS.


# 💻 Internships
- *2020.12 - 2021.6*, Alibaba DAMO Academy, Research Intern. Mentor: [Yong Jiang](https://jiangyong.site).
- *2018.07 - 2018.11*, Netease Thunderfire UX, Data Analyst Intern. Mentor: Lei Xia.


<!-- Statcounter code for personal website
http://horseee.github.io on Google Sites (new) -->
<script type="text/javascript">
var sc_project=12946013; 
var sc_invisible=1; 
var sc_security="08b61411"; 
</script>
<script type="text/javascript"
src="https://www.statcounter.com/counter/counter.js"
async></script>
<noscript><div class="statcounter"><a title="website
statistics" href="https://statcounter.com/"
target="_blank"><img class="statcounter"
src="https://c.statcounter.com/12946013/0/08b61411/1/"
alt="website statistics"
referrerPolicy="no-referrer-when-downgrade"></a></div></noscript>
<!-- End of Statcounter Code -->