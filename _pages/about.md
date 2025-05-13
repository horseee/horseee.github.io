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


<div class="container">

  <h1 id="about-me" class="text-primary"><i class="fas fa-user"></i> About Me</h1>

Hi there! 👋  
Welcome to **Bhanu Prakash Vangala**’s website!

I’m currently a <span class="badge">Ph.D. candidate</span> in Computer Science at the University of Missouri, Columbia <span class="text-accent">(Aug 2023 – Jun 2027)</span>, co-advised by [Dr. Jianlin Cheng](https://calla.rnet.missouri.edu/cheng/) and [Dr. Tanu Malik](https://engineering.missouri.edu/faculty/tanu-malik/). I earned my M.S. in Computer Science <span class="text-accent">(GPA 4.0/4.0)</span> from Missouri—deployed LLMs under [Dr. Grant J Scott](https://scottgs.mufaculty.umsystem.edu/). Completed my B.Tech in CSE (Data Analytics) at VIT Vellore, exploring multilingual NLP & large-scale social media analysis.

  ## <i class="fas fa-bullseye"></i> Core Research Interests

  - **Trustworthy & Interpretable AI** — transparent reasoning, self-reflection, self-correction  
  - **Efficient & Scalable LLMs** — model compression, memory optimization, real-time HPC deployment  
  - **Factuality & Evaluation** — benchmarks & hybrid techniques for factual accuracy  
  - **AI for Scientific Discovery** — LLMs in materials science, biomedicine, policy modeling

  ## <i class="fas fa-bolt"></i> News

  - <span class="text-accent">2025.05</span>: Earned M.S. in CS (GPA 4.0) from UM-Columbia  
  - <span class="text-accent">2025.04</span>: Outstanding Master’s Student Award, MU CS Dept.  
  - <span class="text-accent">2025.04</span>: Submitted thesis proposal: _Trustworthy AI: Building Self-correcting & Self-evolving Models_  
  - <span class="text-accent">2025.04</span>: Presented Hallucination Detection at AAAI Spring Symposium  
  - <span class="text-accent">2025.03</span>: Launched **ReflectMemory** for long-context LLM reasoning  
  - <span class="text-accent">2025.03</span>: Deployed **KubeLLM** on GPU-based HPC clusters  
  - <span class="text-accent">2025.02</span>: Runner-Up, MUIDSI AI for Social Good (VisionAI)  
  - <span class="text-accent">2025.01</span>: Released hallucination-detection benchmarking tools  
  - <span class="text-accent">2024.09</span>: Began LLM-as-a-Service docs (Helm + node-affinity)  
  - <span class="text-accent">2024.01</span>: TA for 100+ full-stack web-dev students  
  - <span class="text-accent">2023.12</span>: Deployed GPU-efficient LLM inference on Nautilus HPC  
  - <span class="text-accent">2023.08</span>: Started research on LLM faithfulness & robustness  
  - <span class="text-accent">2023.06</span>: Admitted to Ph.D. program, UM-Columbia  
  - <span class="text-accent">2023.05</span>: Graduated B.Tech CSE (Data Analytics), VIT Vellore  
  - <span class="text-accent">2023.04</span>: Excellence in Research Award, VIT  
  - <span class="text-accent">2023.03</span>: AI Community Evangelist, Adobe  
  - <span class="text-accent">2022.11</span>: Internshala Student Partner (ISP)  
  - <span class="text-accent">2021</span>: Synergy Team @ VIT  
  - <span class="text-accent">2020</span>: Joined Brandiverse

  ## <i class="fas fa-book"></i> Publications

  <div class="paper-box">
    <div class="paper-box-image">
      <div>
        <div class="badge">Preprint</div>
        <img src="images/papers/publication-placeholder.png" alt="HalluMat" width="100%">
      </div>
    </div>
    <div class="paper-box-text">
      **[HalluMat: Hallucination Detection in Scientific LLMs](#)**  
      _Bhanu Prakash Vangala, Jianlin Cheng_  
      - Hybrid evaluation combining intrinsic & extrinsic checks  
      - Applied to biomedical & scientific text generation  
      [paper] [abstract]
    </div>
  </div>

  <!-- Repeat .paper-box blocks for HalluFormer, Thesis, Projects… -->

  ## <i class="fas fa-project-diagram"></i> Projects

  - **ReflectMemory for Self-Correcting LLMs**: Chain-of-thought embeddings for reasoning consistency  
  - **KubeLLM**: GPU-optimized LLM-as-a-Service via Kubernetes & Helm  
  - **HalluMat & HalluFormer**: Benchmarks & evaluation pipelines for hallucination detection  
  - **ChatMed**: Medical chatbot trained on BioGPT & PubMed  
  - **CropInsight**: CV & LSTM for crop health & yield forecasting  
  - **SocialSift**: Crisis-aware sentiment analysis with multilingual transformers  
  - **EmotionSense**: Multi-modal mental health marker prediction

  ## <i class="fas fa-award"></i> Honors & Awards

  - <span class="text-accent">2025.05</span>: Outstanding Master’s Student Award, UM  
  - <span class="text-accent">2025.03</span>: Runner-Up – MUIDSI Hackathon (VisionAI)  
  - <span class="text-accent">2025.04</span>: Google PhD Fellowship Nomination  
  - <span class="text-accent">2024.10</span>: Outstanding Volunteer, EMNLP 2023  
  - <span class="text-accent">2023</span>: Dean’s Research Excellence Award, VIT  
  - <span class="text-accent">2023</span>: Best Department Thesis Award, VIT  
  - <span class="text-accent">2022</span>: Top 2 Academic Performer, VIT CSE  
  - <span class="text-accent">2022</span>: Runner-Up, VIT AI Tech-Thon  
  - <span class="text-accent">2020</span>: Certificate of Outstanding Achievement, Brandiverse  
  - <span class="text-accent">2019–2023</span>: Multiple Academic Merit Scholarships & ISP/Synergy leadership

  ## <i class="fas fa-graduation-cap"></i> Educations

  - **Ph.D. in Computer Science**, UM-Columbia (2023.08 – 2027.06 expected)  
    - Co-advised by Dr. Jianlin Cheng & Dr. Tanu Malik  
    - Research: Trustworthy LLMs, Self-correcting Models, Evaluation  
    - NASA, NSF & DoD funded

  - **M.S. in Computer Science**, UM-Columbia (2023.08 – 2025.05)  
    - Thesis: Deploying LLM-as-a-Service in Kubernetes HPC  
    - GPA 4.0/4.0; TA for MERN full-stack (100+ students)

  - **B.Tech in CSE (Data Analytics)**, VIT Vellore (2019.05 – 2023.05)  
    - Thesis: Multilingual Sentiment Analysis on KOO  
    - Excellence in Research & Best Thesis Award  
    - Synergy Team core member; AI Tech-Thon runner-up

  ## <i class="fas fa-tasks"></i> Academic Service

  - Reviewers: ICML, ACL, ICCV, CVPR, ICLR, AAAI, ICASSP, NeurIPS, EMNLP, ECCV, IJCAI, NAACL  
  - Journal Reviewer: TPAMI, JVCI, TIP, TMLR

  ## <i class="fas fa-chalkboard-teacher"></i> Teaching Experience

  - TA for Web Development (Fall 2023, Spring 2024, Fall 2024, Fall 2025)

  ## <i class="fas fa-briefcase"></i> Internships & Research Experience

  - **Adobe Research**, NLP Intern (May 2022 – Jan 2023): Info Extraction, Web Mining, Data Management  
  - **UM Data Intensive Computing Lab**, Research & TA (Aug 2023 – Present): Hallucination detection (+30% factual), LLM-as-a-Service (NSF), TA (115+ students)  
  - **UM PAAL Lab**, Research Assistant (Aug 2023 – Jan 2024): USDA UAV crop monitoring, geospatial workflows  
  - **Brandiverse**, Data Analyst Intern (May 2020 – Jul 2020): Customer sentiment NLP  
  - **Internshala ISP**, Student Partner (May 2020 – Dec 2020): Campus campaigns  
  - **VIT Synergy Team**, Club Organizer (2019 – 2020): AI/NLP workshops & events

</div>

<script type="text/javascript">
var sc_project=12946013; var sc_invisible=1; var sc_security="08b61411";
</script>
<script src="https://www.statcounter.com/counter/counter.js" async></script>
<noscript>
  <div class="statcounter">
    <a href="https://statcounter.com/" target="_blank"><img src="https://c.statcounter.com/12946013/0/08b61411/1/" alt="website statistics"></a>
  </div>
</noscript>
