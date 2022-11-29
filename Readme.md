# CS-6340 Project
# Team: The Magicians [Nawazish Khan, Nikolay Lukyanchikov]

# Final Submission

## Methodology
- Train a feed forward Conv network
- Inputs fed to the deep network are Bert Embeddings
- No fine-tuning

### Libraries Used and References:
- [Pytorch](https://pytorch.org)
- Tensorboard - to save train/eval logs
- [Transformers]((https://github.com/huggingface/transformers)) - (Bert Base Uncased)
- Some helper functions from [HuggingFace Repository](https://github.com/huggingface/transformers) to evaluate the results

####  Build Virtual environment using *venv.csh*
### Training the model
- Run *train.py* 
(Note that, the pretrained model from our end is saved in *./output_dir*, Please delete that dir, to train a new model and save it's weights)

### Evaluating the model 
- Run *qa.py* system with *input_file.txt* as argument which contains the input in specified format
- Please wait few minutes (at least 3 to 4 minutes) for evaluation to complete

#### Tested on CADE Machine: lab1-7:1
