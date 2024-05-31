<!-- This template uses html code to offer a bit prettier formatting. This html
code is limited to the header and footer. The main body is and should be written
in markdown. -->

<h1 align="center" position="relative">
  <br>
  <img src=".images/top.jpg" alt="Decorative image">
  <br>
  <br>
  <span>Anomaly Detection in Security Logs using Sequence Modeling</span>
  <br>

</h1>

  This project focused on self-supervised sequence modeling as a tool for performing anomaly detection in network authentication logs. Building on existing work on this topic ([Tuor et al.](https://cdn.aaai.org/ocs/ws/ws0489/17039-75960-1-PB.pdf), [Brown et al.](https://dl.acm.org/doi/abs/10.1145/3217871.3217872)), we investigated changes to the model learning method, the anomaly detection method, and to the model architecture.

## Quick Info
Anomaly detection in security logs using sequence modeling info:

- Members:
  - Simon Gökstorp
  - Jakob Nyberg
  - Yeongwoo Kim
  - Pontus Johnson
  - György Dán
- Status: completed :orange_circle:
- Timeline: 2021-2024


## Publication

This work was presented at the [IEEE/IFIP Network Operations and Management Symposium 2024](https://noms2024.ieee-noms.org). Once the conference proceedings have been published this README will be updated with a link and bibtex. The abstract of the paper is as follows:

<p style="text-align: justify">
"As cyberattacks are becoming more sophisticated, automated activity logging and anomaly detection are becoming important tools for defending computer systems. Recent deep learning-based approaches have demonstrated promising results in cybersecurity contexts, typically using supervised learning combined with large amounts of labeled data. Self-supervised learning has seen growing interest as a method of training models because it does not require labeled training data, which can be difficult and expensive to collect. However, existing self-supervised approaches to anomaly detection in user authentication logs either suffer from low precision or rely on large pre-trained natural language models. This makes them slow and expensive both during training and inference. Building on previous works, we therefore propose an end-to-end trained self-supervised transformer-based sequence model for anomaly detection in user authentication events. Thanks in part to an adapted masked-language modeling (MLM) learning task and domain knowledge-based improvements to the anomaly detection method, our proposed model outperforms previous long short-term memory (LSTM)-based approaches at detecting red-team activity in the “Comprehensive, Multi-Source Cyber-Security Events” authentication event dataset, improving the area under the receiver operating characteristic curve (AUC) from 0.9760 to 0.9989 and achieving an average precision of 0.0410. Our work presents the first application of end-to-end trained self-supervised transformer models to user authentication data in a cybersecurity context, and demonstrates the potential of transformer-based approaches for anomaly detection."
</p>

## Instructions for using the code
This project was built with Python 3.8. The module and all requirements can most easily be installed by running:

`pip install .`

If you want to be able to edit the code, instead run:

`pip install -e .`

The dataset can be found [here](https://csr.lanl.gov/data/cyber1/). You will need to download the `auth.txt.gz` and `redteam.txt.gz` files and place them in `data/`.

Next the data must be preprocessed to the required format. Note that this may take a while. To do this most easily, with default settings, simply run:

`make prepare_data`

Or equivalently:

`python log_analyzer/data_utils/log_file_utils.py`

If you want to use a different subset of the dataset, or for other pre-processing options, have a look at the `log_analyzer/data_utils/log_file_utils.py` file.

The code is now ready to be run. The `makefile` provides a small set of useful shorthands for model training, e.g.:

`make transformer_word-global CUDA=True `

to train a transformer model using a global word-based vocabulary and default config (see `config/`) utilising the GPU (if available).

For more advanced uses have a look at the `log_analyzer/train_model.py` file.

<br>

  <a href="https://www.kth.se/nse/research/software-systems-architecture-and-security/" >
    <img src=".images/kth-round.png" alt="KTH logo" width=80 align="right" />
  </a>

- - - -
This is a project run by the [Software Systems Architecture and Security research
group](https://www.kth.se/nse/research/software-systems-architecture-and-security/)
within the [Division of Network and Systems Engineering](https://kth.se/nse) at
the Department of Computer Science at the School of [Electrical Engineering and
Computer Science](https://www.kth.se/en/eecs) @ [KTH university](https://www.kth.se).

For more of our projects, see the [SSAS page at github.com](https://github.com/KTH-SSAS).