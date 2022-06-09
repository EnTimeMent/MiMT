# MiMT

The *Movement in Multiple Time (MiMT)* neural network architecture models affective movement behaviour at multiple timescales by:
1. processing different body regions with separate but shared time encoding, to account for the independence as well as concomitance between joints during body movement; and
2. predicting the affective label of interest at multiple timescales, to address/explore the pertinent annotation and modelling questions, e.g., of: 
   * for the sparser timescale label, at what specific frames does the affective behaviour actually occur? 
   * for the more fine-grained label, how much evidence (over time) is needed to conclude that the affective behaviour is clearly present? 
		
		
The implemented MiMT architecture focuses on two label timescales, frame-level and window-level. The figure below gives an overview of this MiMT architecture.

![MiMT overview](https://user-images.githubusercontent.com/27019825/148803506-d30b59b7-9f45-4778-acc0-a3019fe09e30.png)

For further details on the use of the MiMT architecture for automatic detection of pain behaviour, please see:

```
@inproceedings{Olugbade2020MiMT,
  title={A Movement in Multiple Time Neural Network for Automatic Detection of Pain Behaviour},
  author={Olugbade, Temitayo and Gold, Nicolas and Williams, Amanda C de C and Bianchi-Berthouze, Nadia},
  booktitle={Companion Publication of the 2020 International Conference on Multimodal Interaction},
  pages={442--445},
  year={2020}
}
```

Please cite the above publication for any use of the MiMT architecture or code.


# Requirements
The MiMT architecture code is based on Python and Tensorflow. 

The network requires time-continuous 3D joint positions for five main body parts, head, left and right upper and lower limbs, together with the 3D positions for the spine region. For example, the left upper limb region input (one of the 5 inputs into the network) is a concatenation of the 3D positions of the lower arm, upper arm, left shoulder, top spine, mid spine, and bottom spine joints to make a `b x seqlen x 6 x 3` input, where `b` is the batch size of the input and `seqlen` is the number of timesteps in the input. The code can easily be modified to use fewer or more regions of the body and/or fewer or more joints within each region.
