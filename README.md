# ErrorDection
This is text error detection code using BiLSTM.
### Input data
The shape of input data is as below <br>
**0\t0\t0\t1\t0\t'Thanks for your attentions .'** <br>
The left side (index 0 ~ -2) is labels which mean correct or incorrect. <br>
The right side (index -1) is the sentence splited by spaces.

In this repository, I used [FCE Dataset for error detection](https://ilexir.co.uk/datasets/index.html) <br>
By parsing this dataset through `parse.py`, you can get the corpus for train/dev/test. 

### Reference
- http://www.aclweb.org/anthology/P16-1112
- https://github.com/kanekomasahiro/grammatical-error-detection