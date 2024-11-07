# The code for RR2QC (Leveraging Label Semantics and Meta-Label Refinement for Multi-Label Question Classification)

## Requirements
    conda create --name RR2QC python=3.11.7
    conda activate RR2QC
    conda install transformers==4.39.1
    conda install safetensors
    conda install scikit-learn
    conda install pandas
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    or
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

## Usage
### Pre-train
    pretrain/main.py
    necessary files:
        label_tree_index (the paths from root to leaf labels)
        train.txt split with '\t'. e.g. Id\tContent\t\labelname


### Downstream-RR2QC
    step:
    1.run train_Retrieval_model.py, set is_init_CCL to True, train the class_center_vector
    2.run test_Retrieval_model.py, set is_init_CCL to True, get the class_center_vector
    3.run train_Retrieval_model.py, set is_init_CCL to False, is_use_CCL to True, train with class centering learning
    4.run test_Retrieval_model.py, set is_init_CCL to False, get the Retrieval label sequence
    5.run train_Reranking_model.py
    6.run test_Reranking_model.py, get the Reranking metalabel sequence
    7.run refine_label_sequence.py, the last step of RR2QC, refining the label sequence by metalabel

    Please read parameters.py for more details.
