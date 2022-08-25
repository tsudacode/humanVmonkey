# humanVmonkey

Model from:  
Tsuda B, Richmond BJ, Sejnowski TJ. *Exploring strategy differences between humans and monkeys with recurrent neural networks.*

**rnn_model.py** contains recurrent neural network model trained by reinforcement learning.  

**humanVmonkey_env.py** contains the specifications for the three working memory tasks from Wittig et al. 2016.

Organization of **rnn_model.py** is
  - helper fxns
  - definition of network class
  - definition of worker class
      - train fxn
      - get_experience fxn
      - test fxn
  - main
      - definition of parameters and output directories
      - creation of central network
      - creation of worker objects that run the network
      - script to deploy workers for training AND testing

Command to run DynaMoE:  
`python3 rnn_model.py [NETSZ] [EPS_TO_TRAIN_ON] [PERFTHRESH] [RUNNO] [GPU]`

%# Citation
%
%If you use this repo in your research, please cite:
%
%    @article{Tsuda_2022,
%    Author = {Tsuda, Ben and Richmond, Barry J. and Sejnowski, Terrence J.},  
%    Title = {Exploring strategy differences between humans and monkeys with recurrent neural networks},  
%    Journal = {},  
%    Publisher = {},  
%    Volume = {},  
%    Number = {},  
%    Pages = {--},  
%    Doi = {},  
    Year = {}}
