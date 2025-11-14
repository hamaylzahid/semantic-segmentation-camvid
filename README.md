<h1 align="center">U-Net Semantic Segmentation on CamVid</h1>  
<p align="center" style="color:#555; font-size:16px;">
A complete deep learning pipeline for <strong>Semantic Segmentation</strong> using the U-Net architecture.<br>
Trained on the <strong>CamVid road-scene dataset</strong>, this project performs pixel-wise classification<br>
to segment real-world urban driving environments with high accuracy.
</p>

<!-- Tech Stack -->
<h4 align="center">Core Technology Stack</h4>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=flat&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Scientific-grey?style=flat&logo=numpy" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=flat&logo=matplotlib" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Metrics-orange?style=flat&logo=scikitlearn" />
</p>

<h4 align="center"> Project Info </h4>
<p align="center">
  <img src="https://img.shields.io/github/last-commit/hamaylzahid/semantic-segmentation-camvid?style=flat&color=orange&logo=github" />
  <img src="https://img.shields.io/badge/Status-Research%20Ready-success?style=flat" />
</p>

<hr>


<h2 align="center">📖 Table of Contents</h2>

<ul style="list-style:none; font-size:16px; line-height:2;">
  <li>🧠 <a href="#overview">Project Overview</a></li>
  <li>🎯 <a href="#Dataset"> Dataset</a></li>
  <li>⚙️ <a href="#installation">Setup & Installation</a></li>
  <li>🧰 <a href="#tools">Libraries & Tools</a></li>
  <li>🤝 <a href="#contribution">Contact & Contribution</a></li>
  <li>📜 <a href="#license">License</a></li>
</ul>

<hr>

<!-- Overview -->
<br><h2 id="overview" align="center">🧠 Project Overview</h2><br>

<p>
This project implements a high-performance <strong>U-Net</strong> model — a fully convolutional encoder-decoder architecture<br>
widely used for <strong>pixel-level semantic segmentation</strong> in modern computer vision.
</p>

<p>
Trained on the <strong>CamVid Urban Scene Understanding dataset</strong>, the model learns to classify every pixel into<br>
<strong>32 semantic classes</strong> such as buildings, roads, vehicles, pedestrians, sky, and vegetation.
</p>

<p>
The encoder captures spatial context, while the decoder restores fine-grained detail using skip connections,<br>
resulting in <strong>crisp segmentation masks</strong> that preserve object boundaries with high accuracy.
</p>

<p>
This implementation includes:
<ul>
  <li>📌 A clean and modular PyTorch training pipeline</li>
  <li>📌 Real-time data augmentation for robust learning</li>
  <li>📌 Custom dataset loaders for RGB images & label masks</li>
  <li>📌 Dice Loss + Cross Entropy for balanced optimization</li>
  <li>📌 Visualization of predictions, masks, and training curves</li>
  <li>📌 Easy reproducibility with requirements.txt</li>
</ul>
</p>

<p><em>
Perfect for autonomous driving research, segmentation-based CV projects, and deep learning portfolios.
</em></p>

<hr>
<!-- Dataset Section -->
<br><h2 align="center" id ="Dataset" >Dataset </h2><br>

### CamVid Urban Scene Segmentation

<p>
The <strong>CamVid dataset</strong> is a high-quality road-scene segmentation dataset widely used in computer vision research.<br>
It contains <strong>urban driving videos</strong> recorded at 30 FPS with <strong>pixel-level class annotations</strong>.
</p>

<p
The dataset includes <strong>32 semantic classes</strong> such as:<br>
Road, Building, Sky, Car, Pedestrian, Tree, Sign, Fence, Sidewalk, and more.
</p>

<p>
  <strong>Total Images:</strong> 701 &nbsp; | &nbsp;
  <strong>Resolution:</strong> 960×720 &nbsp; | &nbsp;
  <strong>Type:</strong> RGB + Pixel Annotations
</p>

<!-- Kaggle Button -->
<p align="center">
    <a href="https://www.kaggle.com/datasets/carlolepelaars/camvid?" target="_blank">
        <img src="https://img.shields.io/badge/Kaggle-View%20Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Badge" />
</a>
</p>

<hr>


<br><h2 id="objectives" align="center">🎯 Core Objectives</h2><br>
<ul>
  <li>Implement U-Net architecture from scratch or pretrained</li>
  <li>Train on CamVid dataset with proper augmentation</li>
  <li>Perform pixel-wise segmentation of urban scenes</li>
  <li>Visualize predictions, segmentation masks, and training curves</li>
  <li>Evaluate using IoU, Pixel Accuracy, and Loss curves</li>
</ul>
<hr>
<br><h2 align="center">🧩 Key Code Snippets</h2><br>

<p>
Below are essential code excerpts showcasing model architecture,<br>
training pipeline, evaluation workflow, and inference steps.  
These give readers a quick but powerful understanding of the implementation.
</p>

### Model Summary

<code>

   from torchsummary import summary
   summary(model, (3, CFG.IMG_SIZE, CFG.IMG_SIZE))

</code>

<br>

### Training Loop 

<code>
  
    for epoch in range(CFG.EPOCHS):
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{CFG.EPOCHS} | Loss: {train_loss/len(train_loader):.4f}")
    
</code>

<br>

### Validation Step

<code>
  
    model.eval()
    with torch.no_grad():
    total_iou, total_acc = 0, 0

    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        total_acc += pixel_accuracy(outputs, masks)
        total_iou += iou_score(outputs, masks, CFG.NUM_CLASSES)

        print("Pixel Accuracy:", total_acc / len(val_loader))
        print("Mean IoU:", total_iou / len(val_loader))

</code>
<hr>
<br><h2 id="installation" align="center">Setup & Installation > <br>
  
### Clone repository

git clone https://github.com/hamaylzahid/semantic-segmentation-camvid.git

### Navigate to project folder

cd semantic-segmentation-camvid

### Install required dependencies

pip install -r requirements.txt

### Launch Notebook

jupyter notebook "Semantic_Segmentation_with_U-Net.ipynb"

<hr>


<br><h2 id="tools" align="center">💼 Libraries & Tools</h2><br>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/NumPy-Numerical-blue?style=flat-square&logo=numpy" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-green?style=flat-square&logo=matplotlib" />
  <img src="https://img.shields.io/badge/Pillow-Images-yellow?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Metrics-orange?style=flat-square&logo=scikitlearn" />
</p>

<hr>

<br><h2 align="center">🤝 Contact & Contribution</h2><br><br>

<p align="center">
  Have feedback, want to collaborate, or just say hello?<br>
  <strong>Let’s connect and improve semantic segmentation together.</strong>
</p>

<p align="center">
  📬 <a href="mailto:maylzahid588@gmail.com">maylzahid588@gmail.com</a> &nbsp; | &nbsp;
  💼 <a href="https://www.linkedin.com/in/hamaylzahid">LinkedIn Profile</a> &nbsp; | &nbsp;
  🌐 <a href="https://github.com/hamaylzahid/semantic-segmentation-camvid">GitHub Repo</a>
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid/semantic-segmentation-camvid/stargazers">
    <img src="https://img.shields.io/badge/Star%20This%20Project-Give%20a%20Star-yellow?style=for-the-badge&logo=github" alt="Star Badge" />
  </a>
  <a href="https://github.com/hamaylzahid/semantic-segmentation-camvid/pulls">
    <img src="https://img.shields.io/badge/Contribute-Pull%20Requests%20Welcome-2ea44f?style=for-the-badge&logo=github" alt="PRs Welcome Badge" />
  </a>
</p>

<p align="center">
  ⭐ Found this project helpful? Give it a star on GitHub!<br>
  🤝 Want to improve it? Submit a PR and join the mission.<br>
  <sub><i>Your contributions help advance real-world scene understanding systems.</i></sub>
</p>

<br>
<h2 align="center">📜 License</h2><br>

<p align="center">
  <a href="https://github.com/hamaylzahid/semantic-segmentation-camvid/commits/main">
    <img src="https://img.shields.io/github/last-commit/hamaylzahid/semantic-segmentation-camvid?color=blue" alt="Last Commit">
  </a>
  <a href="https://github.com/hamaylzahid/semantic-segmentation-camvid">
    <img src="https://img.shields.io/github/repo-size/hamaylzahid/semantic-segmentation-camvid?color=lightgrey" alt="Repo Size">
  </a>
</p>

<p align="center">
  This project is licensed under the <strong>MIT License</strong> — open to use, modify, and expand.
</p>

<p align="center">
  ✅ <strong>Project Status:</strong> Complete & Portfolio-Ready<br>
  🧾 <strong>License:</strong> MIT — <a href="LICENSE">View License »</a>
</p>

<p align="center">
  <strong>Crafted with deep learning expertise & semantic segmentation research</strong> 🖼️✨
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid">
    <img src="https://img.shields.io/badge/GitHub-%40hamaylzahid-181717?style=flat-square&logo=github" alt="GitHub" />
  </a>
  •
  <a href="mailto:maylzahid588@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact%20Me-red?style=flat-square&logo=gmail&logoColor=white" alt="Email" />
  </a>
  •
  <a href="https://github.com/hamaylzahid/semantic-segmentation-camvid">
    <img src="https://img.shields.io/badge/Repo-Link-blueviolet?style=flat-square&logo=github" alt="Repo" />
  </a>
  <br>
  <a href="https://github.com/hamaylzahid/semantic-segmentation-camvid/fork">
    <img src="https://img.shields.io/badge/Fork%20This%20Project-Contribute%20to%20AI-2ea44f?style=flat-square&logo=github" alt="Fork Badge" />
  </a>
</p>

<p align="center">
  <sub><i>Designed for real-world urban scene segmentation and deep learning showcase.</i></sub>
</p>

<p align="center">
  🤖 <b>Use this project to demonstrate your expertise in computer vision and semantic segmentation</b><br>
  🧬 Clone it, modify it, expand it — and build advanced U-Net based segmentation systems.
</p>
