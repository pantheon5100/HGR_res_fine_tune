# HGR_res_fine_tune
fine tune to recogonize 
+ Use ResNet18 with pretrain .
+ Learning rate change policy is mutilstep.
+ Feature extra from resnet layer4 with global Maxpooling.
+ Classifier use one block which is fc+bn+relu+fc
