# Hybrid Images


A hybrid images has two coherent global image interpretations, one of which is of the low spatial frequencies, the other of high spatial frequencies. An example can be seen below:


![obj1](./12.jpeg) ![obj2](./13.jpeg)
![blended](./output.jpg)





## Usage


Code for generating hybird images:

e.g., for blending image 12 to image 13 use:

```

python hybrid.py -i0 ./data/12.jpeg -i1 ./data/13.jpeg

```


Use the mass_hybrid(config) function to generate the datase (please see the code).


This code is a modified version of the original code at:
https://github.com/rhthomas/hybrid-images



## Dataset

Dataset is available in DB.zip encompassing 10 categories (Banana, Custard Apple, Fig, Granny Smith, Jackfruit, Lemon, Orange, Pineapple, Pomegranate, and Strawberry) with 10 images per category.



## Citation

If you use this code in your research, please cite this project.

```
@article{Borji_Hybrid_2022,
  title={CNNs and Transformers Perceive Hybrid Images Similar to Humans},
  author={Borji, Al},
  journal={Arxiv},
  year={2022}
}
```

