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

Hi there! 👋

Welcome to **Bhanu Prakash Vangala**’s website!

I’m currently a Ph.D. candidate in Computer Science at the University of Missouri, Columbia (Aug 2023 – Jun 2027), co-advised by [Dr. Jianlin Cheng](https://calla.rnet.missouri.edu/cheng/) and [Dr. Tanu Malik](https://engineering.missouri.edu/faculty/tanu-malik/).

Previously, I earned my M.S. in Computer Science from the University of Missouri (GPA: 4.0/4.0), where I worked on deploying Large Language Models (LLMs) in distributed computing environments advised by [Dr. Grant J Scott](https://scottgs.mufaculty.umsystem.edu/). I completed my B.Tech in Computer Science and Engineering (Data Analytics) at Vellore Institute of Technology, India, where I explored multilingual NLP and large-scale social media analysis.

My core research interests include:

🌲 **Trustworthy and Interpretable AI**  
Developing systems that can reason transparently, self-reflect, and correct their own outputs—improving reliability and reducing harmful or misleading responses.

🌱 **Efficient and Scalable Language Models**  
Focusing on model compression, memory optimization, and large-scale deployment of LLMs for real-time inference in resource-constrained and HPC environments.

🌿 **Factuality and Evaluation in Language Models**  
Designing evaluation benchmarks and hybrid techniques to assess consistency, factual accuracy, and context-grounded reasoning in LLMs.

🍃 **AI for Scientific Discovery**  
Applying LLMs in scientific domains such as materials science, biomedical research, and policy modeling—empowering AI to support researchers in hypothesis generation and knowledge synthesis.

Beyond research, I enjoy mentoring students as a Teaching Assistant for full-stack web development courses, helping them connect theoretical concepts with real-world applications. I’m also passionate about guiding international students through blogging and sharing insights on pursuing higher education abroad.

Thanks for stopping by—feel free to explore my work on [GitHub](https://bhanuprakashvangala.github.io) or connect with me on [LinkedIn](https://www.linkedin.com/in/vangalabhanuprakash/)!

[//]: <> <span style="color:blue"> </span>

# 🔥 News

- *2025.05*: &nbsp;🎓 Earned my M.S. in Computer Science (GPA: 4.0/4.0) from the University of Missouri, Columbia.
- *2025.04*: &nbsp;🏆 Received the **Outstanding Master’s Student Award** from the MU Department of Computer Science.
- *2025.04*: &nbsp;📤 Submitted a thesis proposal: *"Trustworthy AI: Building Self-correcting and Self-evolving models for Scientific Discovery."*
- *2025.04*: &nbsp;🎉 Presented our work on Hallucination Detection at **AAAI Spring Symposium 2025 on AI for scientific Discovery track**
- *2025.03*: &nbsp;Started development of **ReflectMemory**, focused on persistent memory control for long-context LLM reasoning.
- *2025.03*: &nbsp; Deployed updated **KubeLLM** framework for multi-tenant LLM inference on GPU-based HPC clusters.
- *2025.02*: &nbsp;🥈 Achieved **Runner-Up** in the MUIDSI School for Generative AI for social good hackathon on VisionAI for Visually impaired project.
- *2025.01*: &nbsp;Released benchmarking tools for **hallucination detection in scientific LLMs**, supporting hybrid evaluation methods.
- *2024.09*: &nbsp;Initiated documentation work on scalable **LLM-as-a-Service infrastructure** using Helm charts and node affinity scheduling.
- *2024.01*: &nbsp;Working as a TA for over 100 students as a TA in web development course – guiding full-stack app development.
- *2023.12*: &nbsp;Led deployment of GPU-efficient LLM inference systems in the university’s Kubernetes-based HPC environment (Nautilus).
- *2023.08*: &nbsp;Began research on **faithfulness, interpretability, and robustness** in large generative language models.
- *2023.06*: &nbsp;🎉 Admitted to the Ph.D. program in Computer Science at the University of Missouri
- *2023.05*: &nbsp;Graduated with a B.Tech in CSE (Data Analytics) from VIT Vellore.
- *2023.04*: &nbsp;🏅 Honored with the **Excellence in Research** Award at VIT for multilingual NLP and social media analytics contributions.
- *2023.03*: &nbsp;Volunteered as an **AI Community Evangelist** at Adobe, contributing to community education and developer engagement.
- *2022.11*: &nbsp;Served as an **Internshala Student Partner (ISP)**, leading brand campaigns and peer mentoring on campus.
- *2020: &nbsp;Joined the **Brandiverse team** as a creative contributor, working on outreach and media strategy.
- *2021*: &nbsp;Collaborated with the **Synergy Team at VIT**, supporting student experience initiatives and university development programs.

# 📝 Publications 


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Preprint</div><img src='images/papers/publication-placeholder.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**HalluMat: Hallucination Detection in Scientific LLMs**]()

**Bhanu Prakash Vangala**, Jianlin Cheng

- A hybrid evaluation pipeline combining intrinsic and extrinsic techniques to flag hallucinations in domain-specific outputs.
- Applied to biomedical and scientific text generation tasks.

<div style="display: inline">
    <a href="#"> <strong>[paper]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">
        <p>This work introduces HalluMat, a benchmarking framework designed to assess hallucinations in domain-tuned large language models, especially in scientific domains such as biomedicine and material science. The method integrates intrinsic scoring with external factual checks using ground-truth corpora and curated datasets, enhancing the reliability of hallucination detection in practical applications.</p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Preprint</div><img src='images/papers/publication-placeholder.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**HalluFormer: Faithfulness Evaluation Framework**]()

**Bhanu Prakash Vangala**, Jianlin Cheng

- Transformer-based architecture for multi-dimensional consistency checking of LLM outputs.

<div style="display: inline">
    <a href="#"> <strong>[paper]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">
        <p>HalluFormer introduces a structured evaluation model to detect and categorize different types of hallucinations produced by LLMs. It leverages fine-tuned transformer models on domain-specific contexts to score consistency, factual overlap, and logical coherence.</p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Master's Thesis</div><img src='images/papers/publication-placeholder.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**Deploying LLM-as-a-Service in Kubernetes HPC Clusters**]()

**Bhanu Prakash Vangala**, Grant Scott, Jianlin Cheng

- Designed a Helm-based GPU-aware deployment pipeline for LLM inference in research clusters.

<div style="display: inline">
    <a href="#"> <strong>[thesis]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">
        <p>This work focuses on scalable and efficient deployment strategies for large language models in high-performance computing (HPC) environments. It outlines a Helm-chart-based approach for deploying containerized models with GPU affinity scheduling, resource throttling, and multi-user access configurations.</p>
    </div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Project</div><img src='images/papers/publication-placeholder.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**Brain Tumor Detection in MRI Images**]()

**Bhanu Prakash Vangala**

- Built a CNN-based pipeline to classify MRI scans into normal and tumor-positive cases.
- Used preprocessed image datasets and trained on Google Colab with Keras/TensorFlow.

<div style="display: inline">
    <a href="#"> <strong>[project]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">
        <p>This project implements deep convolutional architectures for identifying brain tumors from MRI scans. The pipeline includes data preprocessing, segmentation, and binary classification, helping radiologists prioritize diagnostic review.</p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Project</div><img src='images/papers/publication-placeholder.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**Pneumonia Detection in Chest X-rays Using Deep Learning**]()

**Bhanu Prakash Vangala**

- Applied CNN models to classify chest X-rays for pneumonia diagnosis.
- Trained on Kaggle datasets using transfer learning (ResNet, VGG).

<div style="display: inline">
    <a href="#"> <strong>[project]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">
        <p>This project uses transfer learning on standard CNN architectures to detect pneumonia in X-ray images. The system enables rapid, automated assessment for medical screening support, particularly in low-resource clinical environments.</p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Project</div><img src='images/papers/publication-placeholder.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**Image Colorization Using AI**]()

**Bhanu Prakash Vangala**

- Developed a deep learning model to convert grayscale images to color.
- Used convolutional autoencoders and GAN-based architectures for photorealistic results.

<div style="display: inline">
    <a href="#"> <strong>[project]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">
        <p>This project explores automatic colorization of black-and-white images using deep neural networks. By leveraging contextual feature extraction and GANs, the model generates plausible color mappings that restore image vibrancy and realism.</p>
    </div>
</div>

</div>
</div>

---

# 📊 Projects

- **ReflectMemory for Self-Correcting LLMs**  
  Built a memory module to store chain-of-thought embeddings and ensure reasoning consistency across inference rounds.

- **KubeLLM: LLM-as-a-Service Platform**  
  Deployed a scalable, GPU-optimized LLM inference service using Kubernetes, Helm, and horizontal pod autoscaling.

- **HalluMat & HalluFormer**  
  Developed benchmarking and evaluation pipelines for hallucination detection in scientific LLMs.

- **ChatMed: Medical Chatbot for Health Guidance**  
  Trained using BioGPT and PubMed literature to provide symptom-based support and healthcare guidance.

- **CropInsight: AI for Agriculture**  
  Combined computer vision and LSTM-based sequence modeling to analyze crop health and yield forecasts.

- **Hackathon Projects @ Mizzou**  
  Runner-up award winner for building a real-time LLM hazard indicator tool, VisionAI for visually impaired people.

- **SocialSift: Crisis-aware Sentiment Analysis**  
  Used multilingual transformers to analyze real-time sentiment from social media during natural disasters.

- **EmotionSense: Multi-modal Mental Health Model**  
  Built using text, audio, and facial signals to predict mental health markers like anxiety or depression.



# 🎖 Honors and Awards

- *2025.05*: **Outstanding Master’s Student Award**, College of Engineering, University of Missouri  
- *2025.03*: 🥈 **Runner-Up – MUIDSI Hackathon** for *VisionAI: AI-Powered Assistance for the Visually Impaired*, awarded $1,000  
- *2025.04*: Selected for **Google PhD Fellowship Nomination**, one of three University of Missouri nominees in NLP  
- *2024.10*: **Outstanding Volunteer**, EMNLP 2023   
- *2023*: 🏅 **Dean’s Research Excellence Award**, Vellore Institute of Technology (VIT)  
- *2023*: **Best Department Thesis Award**, VIT for B.Tech thesis on multilingual sentiment analysis  
- *2022*: **Top 2 Academic Performer**, CSE Department, VIT  
- *2022*: 🥈 **Runner-Up**, VIT AI Tech-Thon  
- *2020*: **Certificate of Outstanding Achievement**, Data Analyst Intern at Brandiverse  
- *2019–2023*: Multiple **Academic Merit Scholarships** and recognitions as **Internshala Student Partner (ISP)** and **Synergy Team Lead**, VIT  

# 🎩 Educations

- *2023.08 – 2027.06 (expected)*, **Ph.D. in Computer Science**, University of Missouri, Columbia  
  Co-advised by Dr. Jianlin Cheng and Dr. Tanu Malik  
  - Research focus: Trustworthy and Efficient LLMs, Self-Correcting and Evolving Language Models, Evaluation in LLMs  
  - Google PhD Fellowship nominee (NLP), Outstanding Student Award recipient
  - Supported by NASA, National Science Foundation, and Department of Defense grants for research in scientific LLMs and scalable AI infrastructure

- *2023.08 – 2025.05*, **M.S. in Computer Science**, University of Missouri, Columbia  
  Thesis: *Deploying LLM-as-a-Service in Kubernetes HPC Clusters*  
  Advisors: Dr. Grant Scott and Dr. Jianlin Cheng  
  - GPA: 4.0/4.0  
  - Built Helm/Kubernetes-based LLM inference pipelines in HPC environments  
  - TA for Full-Stack MERN Development (mentored 100+ students)

- *2019.05 – 2023.05*, **B.Tech in Computer Science and Engineering specialization in Data Analytics)**, Vellore Institute of Technology, India  
  - Awarded Excellence in Research and Best Department Thesis  
  - Thesis: *Multilingual Sentiment Analysis of Social Media Posts on KOO platform*  
  - Core member of Synergy Team, Internshala Student Partner and Student Ambassador at Internshala, Runner-up in VIT AI Tech-Thon  
  - Internship and volunteer work: Adobe (AI Evangelist), Brandiverse (Data Analyst)

- *2017.06 – 2019.04*, **Intermediate (+2) – MPC (Maths, Physics, Chemistry)**, Altitude College, Hyderabad, India  
  - Focused on engineering and analytical skill development through JEE/competitive preparation
  - 1554 in MIT Entrance Test, Secured 88% in JEE Mains qualified for JEE Advanced

- *2017.03*, **10th Standard – Secondary School Certificate (SSC)**, City Central School, India  


# 📋 Academic Service
- Conference Volunteer Reviewer: ICML (25, 24, 23), ACL (25, 24, 23), ICCV (25), CVPR (25), ICLR (25), AAAI (25), ICASSP (25), NeurIPS (24), EMNLP (24), ECCV (25), IJCAI (25), NAACL (25)
- Journal: TPAMI, JVCI, TIP, TMLR

# 🍞 Teaching Experience
- Fall 2025. Fall 2024, Spring 2024, Fall 2023. TA for Web Development.

# ☃️ Internships & Research Experience

- *May 2022 – Jan 2023*, **Adobe Research**, NLP Research Intern  
  Conducted research in Information Extraction, Web Mining, and Data Management for intelligent interfaces.  
  *Mentor*: [Nanda Kishore](https://research.adobe.com/person/nandakishore-kambhatla/)

- *Aug 2023 – Present*, **University of Missouri – Data Intensive Computing Lab**, Research and Teaching Assistant  
  - Developed a hallucination detection framework for scientific LLMs with 30% factual improvement in material science queries.  
  - Led deployment of LLM-as-a-Service using Kubernetes & Helm in HPC clusters (NSF-funded).  
  - Teaching Assistant for Web Development, supporting 115+ students in MERN Stack, grading, and project mentorship.

- *Aug 2023 – Jan 2024*, **University of Missouri – PAAL Lab**, Research Assistant  
  - Led a USDA-funded AI project on UAV-based crop monitoring.  
  - Designed real-time geospatial analysis workflows using QGIS for agricultural research.

- *May 2020 – Jul 2020*, **Brandiverse**, Data Analyst Intern  
  - Analyzed customer sentiment using NLP pipelines; contributed to marketing strategy improvements.  
  - *Recognition*: Certificate of Outstanding Achievement

- *May 2020 – Dec 2020*, **Internshala**, Internshala Student Partner (ISP)  
  Promoted internships, conducted career-building sessions, and facilitated student-industry interaction on campus.

- *2019 – 2020*, **VIT University – Synergy Team & Club Organizer**  
  Actively contributed to university tech initiatives and conducted AI/NLP workshops and events under various student bodies.


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
