# Comparative Analysis: Bertinetto et al.'s 'Making Better Mistakes' vs. Barz & Denzler's 'Hierarchy-based Image Embeddings

## Dataset choice

CIFAR-100 was chosen for all the experiments done for this comparison, due to its availability and it's size smaller size compared to the other datasets used in the two papers (ImageNet, iNaturalist'19). The hierarchy is extracted through a WordNet-based taxonomy proposed by Barz and Denzler (2019)  [cifar hierarchy](./Cifar-Hierarchy).

## Hierarchical structure

Due to Barz and Denzler (2019) already performing experiments on the Cifar dataset, my task was to get Cifar to work on the code base of the making better mistakes paper, to do that we need to conform to the hierarchy structure used, which is an nltk Tree object in contrast to the child-parent dictionaries computed in the Semantic Embeddings paper, we also have to compute the LCA(lowest common ancestor) distances in order to compute the hierarchical loss, as well as for hierarchical metrics calculations, the scripts for this conversion are in [Cifar hierarchy scripts](./data/scripts_asis/)

## Architecture used

the architecture used is one of the ones tested in Barz and Denzler (2019), it's a Plainet-11 a VGG like architecture adapted for quicker experiments compared to the other deep neural networks tested (resnet-x, pyramidnet...). 
<div>
  <img src="./assets/plain-11.png" alt="model architecture" style="height: 50vh;">
</div>
While Barz and Denzler (2019) achieved 74% accuracy on CIFAR-100, I've only managed around 55%. This discrepancy is primarily due to differences in the training process. Unlike Barz and Denzler's use of SGD with warm restarts and cosine annealing over 372 epochs, I trained with a fixed learning rate for only 25 epochs. This significant reduction in training time and lack of learning rate optimization likely accounts for the performance gap

## Appendix
### Cifar wordNet based hierarchy
![cifar hierarchy](./Cifar-Hierarchy/hierarchy.svg)