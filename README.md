
> [!TIP]
> This project is a complementary to the __Attack AI Model__ project that can be found in the [repo](https://github.com/dahmansphi/attackai). To this end, to understand the entire idea you shall make sure to read both documentations.

# About the Package
## Author's Words
Welcome to **the first Edition of PROTECT AI MODEL SIMULATION- protectai Tool** official documentation. I am Dr. Deniz Dahman the creator of the BireyselValue algorithm and the author of this package. In the following section you will have a brief introduction on the principal idea of the __protectai__ tool. In addition, a reference to the academic publication on this potential type of __cyber attack__ and potential __defence__. Before going ahead, I would like to let you know that I have done this work as an independent scientist without any fund or similar capacity. 
I am dedicated to proceeding and seek further improvement on the proposed method at all costs. To this end if you wish to contribute in any way to this work, please find further details  in the contributing section.  
  
## Contributing 

If you wish to contribute to the creator of this project and the author, you may want to check possible ways on: 

> `To Contribute in any way possible, thank you, you can check` :

1. view options to subscribe on [Dahman's Phi Services Website](https://dahmansphi.com/subscriptions/)
2. subscribe to this channel [Dahman's Phi Services](https://www.youtube.com/@dahmansphi)     
3. you can support on [patreon](https://patreon.com/user?u=118924481) 


If you prefer *any other way of contribution*, please feel free to contact me directly on [contact](https://dahmansphi.com/contact/). 

*Thank you*

# Introduction

## The Data poisoning
The current revolution of the __AI framework__ has become almost the main element in every solution that we use daily. Many industries heavily rely on those AI models to generate responses accordingly. In fact, it has become a trend that once a product utilizes AI as its backend, then its potential to penetrate marketplace is substantially higher than the one doesn't.

This trend has pushed many industries to consider __the implementation of AI models__ in the tire of their business process. This rush is understandable from the way that those industries believe; for businesses to secure a place in today’s competitive market, they must catch up with the most recent advances in the realm of technology. However, one must ask what is this __AI problem solving paradigm__ anyway?  

In my published project [the Big Bang of Data Science](https://dahmansphi.com/the-big-bang-of-data-science/) I do provide a comprehensive answer to this question, from abstract and concrete perspective. But let me just summaries both perspectives in a few lines. 

Basically, AI model is a mathematical tool, so to speak. It mainly relies on an important stage, __the training stage__. As a metaphor, imagine it as a human brain that learns over time from the surroundings and the circumstances where it lives. Those surroundings and circumstances are the cultures, beliefs, people, etc. Once this brain is shaped and formed, it starts to make decisions and offers answers. Yet, we from the outside start to judge those decisions and answers and the brain would react to those judgements.  

AI models are mimicking such paradigm. The human brain is __the mathematical equation__ of the model, the surroundings and the circumstances are __the training samples__ that we feed to the mathematical equation to learn, the judgements by the surroundings are __the calculation of those misclassified cases__ which known as obtaining the derivatives. Obviously, we then __aim to have a model that can give accurate answers__ with a minimum level of mistakes.  

![training_ai_model](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/training_ai_model.gif) 

Once the technical workflow of the AI is understood, it should be clear then that the __training samples__ from which the AI model learns are the most important element of this entire flow. This element can be thought of as the __adjudicator__ whether the model will __succeed or fail__. 
To this end, such element is a target for __adversaries__ who aim to fail the model. If such __attack__ is successful, then it’s known as __data poisoning attack__.  

```Data poisoning is a type of cyberattack in which an adversary intentionally compromises a training dataset used by an AI or machine learning (ML) model to influence or manipulate the operation of that model``` 

Such type of attack can be done in several ways: 
- [x] Intentionally injecting false or misleading information within the training dataset, 
- [x] Modifying the existing dataset, 
- [x] Deleting a portion of the dataset.  

Unfortunately, such __cyber-attack__ could go undetected for so long due to the framework of the AI at the first place. Furthermore, __the lack of fundamental understanding__ of the AI black-box, and the __employing of ready-to-use AI models__ by industry practicians without the comprehensive understanding of the mathematics behind the entire framework.

However, there are some signs that might lead to the observation that the __AI model__ is compromised. Some of those signs are: 

1. __Model degradation__: Has the performance of the model inexplicably worsened over time?
Unintended outputs	Does the model behave unexpectedly and produce unintended results that cannot be explained by the training team?
2. __Increase in false positives/negatives__: Has the accuracy of the model inexplicably changed over time? Has the user community noticed a sudden spike in problematic or incorrect decisions?
3. __Biased results__: Does the model return results that skew toward a certain direction or demographic (indicating the possibility of bias introduction)?
4. __Breaches or other security events__: Has the organization experienced an attack or security event that could indicate they are an active target and/or that could have created a pathway for adversaries to access and manipulate training data?
5. __Unusual employee activity__: Does an employee show an unusual interest in understanding the intricacies of the training data and/or the security measures employed to protect it?

This [gif](https://github.com/dahmansphi/attackai/blob/main/assets/ai_attack_simulation.gif) illustrates the kind of data poisoning attack on __AI Model__. It basically shows how the __alphas or weights__ are influenced by the new training samples which the model uses __to update itself__.


To this end, such matters must be considered by the company AI division once they decide to employ the __AI problem-solving paradigm__.   

## protectai package __version__1.0
I have presented __two types of attacks__ that AI model can face; the first type is __corrupt data training sample__, and the second is __crazy the model__. Both are explained in detail in the repo project of the [attackai](https://github.com/dahmansphi/attackai?tab=readme-ov-file#attackai-package-__version__10). I have illustrated the image of both attacks. now I shall: 

1. first, illustrate a healthy image of AI model, in this case I will carry on the same simulation using the scenario of the [storyline](https://github.com/dahmansphi/attackai?tab=readme-ov-file#project-setup), so I will show the __first original creation of the x-ray-ai.h5__ model. Check the [details](https://github.com/dahmansphi/attackai?tab=readme-ov-file#attackai-package-__version__10). 

2. Second, I will show the effect of __both attacks__ that I illustrated using the [attackai](https://github.com/dahmansphi/attackai) tool. You may want to check the illustration of [attack type one](https://github.com/dahmansphi/attackai), and [attack type two](https://github.com/dahmansphi/attackai). As a result, will observe the performance after both attacks. 

3. Finally, I shall present the proposal of __defence__ strategy from both attacks.  

Here, I introduce the tool __protectai__, which basically is going to accomplish the __three outlines__ motioned above.  . 

> [!IMPORTANT]
> __This tool illustrates the prior three [objectives](#protectai-package-__version__10) for an educational purpose ONLY. The author provides NO WARRANTY OF ANY KIND, INCLUDING THE WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE__. 

# Installation 
> [!TIP]
> The simulation using __protectai__ is done on a __binary class__ images dataset, referenced in the below section. The _gif_ illustrations shows the [storyline](https://github.com/dahmansphi/attackai?tab=readme-ov-file#installation) as assumed.

## Data Availability
The NIH chest radiographs that support the findings of this project are publicly available at https://nihcc.app.box.com/v/ChestXray-NIHCC and https://www.kaggle.com/c/rsna-pneumonia-detection-challenge. The Indiana University Hospital Network database is available at https://openi.nlm.nih.gov/. The WCMC pediatric data that support the findings of this study are available in the identifier 10.17632/rscbjbr9sj.3

## Project Setup
### Story line
Basically, the storyline of the simulation is simulated on a __chest X-ray dataset__, outlined [above](#data-availability). The outline of the story goes as follows: 

1. A fictious healthcare center is operating as a __chest X-ray__ diagnosis place. 
2. In its workflow, called the __old paradigm__, the case _person_ has a chest X-ray scan; then the __scanned image__ is delivered to a domain expert, the Pulmonologist, to examine the image; and finally, the results are presented to the case person. 
3. The clinic decides to move to a __new paradigm__, by integrating a new layer for diagnosis that lays between the case input and the domain expert diagnosis. The new proposed block is to implement an __AI agent__ that essentially diagnosis the case and predict its label as __Normal or Pneumonia__, then that will pass to the domain expert to confirm. 

### Environment Setup
In the first phase the clinic creates the __AI Model__, that is by following: 
1. Main folder that contains subfolders __(train, and test)__; in each there are subfolders of classes for __(normal and pneumonia)__ cases 
2. The images basically are transformed into __tensors__ which then are fed into a neural network with x hidden layers, the system works until it produces the main predictive model __x-ray-ai.h5__ 
3. This model contains the valid __ratios weights__ that done the math to make the predictions 

### Pipeline Setup
The AI team then decides to create the pipeline to update the mode as follows: 
1. On a weekly basis the new cases are collected 
2. Then the same setup as mentioned above is created for the folders 
3. The original model then used to make the current update using the new batches of __x-ray images__ 
4. Technically speaking, that new input will update the model in a way that change the model weights values 

### AI Model Protection  
The AI team has decided to add a __security layer before the update of the AI model__. Basically, the new security layer shall __inspect__ the weekly batches before __feeding them to the AI model__. The new security layer shall perform inspection against: 

1. [Attack type one](https://github.com/dahmansphi/attackai?tab=readme-ov-file#type-one-attack).  

2. [attack type two](https://github.com/dahmansphi/attackai?tab=readme-ov-file#type-two-attack). 

## Install protectai

> [!TIP]
> make sure to create the project setup as outlined [above](#project-setup).

to install the package all what you have to do:
```
pip install protectai
```
You should then be able to use the package. You may want to confirm the installation

```
pip show protectai
```
The result then shall be as:

```
Name: protectai
Version: 1.0.0
Summary: Simulation of protecting AI model from poisoning attack
Home-page: https://github.com/dahmansphi/protectai
Author: Dr. Deniz Dahman's
Author-email: dahmansphi@gmail.com
```

## Employ the protectai -**Conditions**

> [!IMPORTANT]
> It’s mandatory, to use the first edition of protectai, to make sure the __update__ folder that have the subfolders of the __normal and Pneumonia__ as illustrated in the [gif](https://github.com/dahmansphi/attackai?tab=readme-ov-file#installation). 

## Detour in the protectai package- Build-in
Once your installation is done, and you have met all the conditions, then you may want to check 
the build-in functions of the protectai and understand each.  
Essentially, if you create an instance from the protectai as so: 

```
from protectai.protectai import ProtectAI
inst = ProtectAI()
```
now this **inst** instance offers you access to those build in functions that you need. 
this is a screenshot:

![Screenshot of build-in functions of the protectai tool.](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/functions_paim.png)

Once you have __protectai instance__, here are the details of the right sequence to employ the __defence__:

### Set the path for the main model:

first we shall call the `set_path_models()` function to prepare the training and testing dataset with the spec for the main model.

```inst.set_path_models(paths)```

the function takes __one argument__ that is the path to the __training and testing__ folders. or basically, any folders that shall want to be in tandem with creation a __predictive model__ of images. Here are some pictures illustrate the result after calling the function. 

![Screenshot of attack one with stamp.](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/set_path_model.png)

### make model function:
This function as the name implies, is to create the skelaton __AI predictive model__.  This is a __neural network__ model that accept __binary type of classess__ 

```inst.make_model()```

### train and test the model
In the standard workflow to create a neural network we use __training__ samples to train the model and eventually the model would have its set of __ratios/weights__. However, we should test that trained model. To this end, this function `train_test_model()` does that. It requires two arguments, the first is the __skeleton of the neural network model__ created in previous [function](#make-model-function), and the second is a __list of paths to the training and testing folders__. 

```inst.train_test_model(model, paths)```

The result of the training and testing can be seen in the figure below:
![Screenshot train test model.](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/train_test.png)

Once the result is fine and satisfactory then you can move and create the original model.

### Produce AI Model
Once these results from the __train and test__ are satisfactory then you can use the `produce_model()` function to output the model. This function takes __two arguments__, the first is the path to the __skeleton model__ we output from [above](#make-model-function), the second argument is a __list contains the path to the training samples, and path where to save the new model__. Once this function is executed then you have the original model as shown in the picture below. _notably_ you have options of the type of the saved model, e.g. h5, Keras, or any form of structure you wish to save the ratios and any other information.

```inst.produce_model(model=model, paths_folder=paths_produce)```

![original model](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/orginal_model.png)

### Prediction Task

Since the model is ready, it is time now to see the model in action. There is the use of the `predict()` function. This function takes two arguments; the first one is the __path to the folder of images__ to make prediction on their labels, and __the path to the original model__.  

```inst.predict(img_path=path_to_imgs, model_path=orignial_model)```

if we, just for testing purpose, point the model to folder of __normal imgages__ we could see the results as in the figure below with accuracy of 100%, and the same for the __Pneumonia images__ we could also see almost 100%.

![prediction by model for normal imgs](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/predict_1.png)
![prediction by model for pneumonia imgs](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/predict_1_pnemonia.png)

### Model Update Framework

I have discussed in detail the workflow of updating the existing model in the [pipeline workflow section](#pipeline-setup). To make that workflow in action the use of the ` inst.update_trained_model_with_test()` and `inst.update_trained_model_produce()` is the right order. As the name suggests, the first function __does the update with test__ and the second __does the update with save new mode updated__. Basically, both functions take __two arguments__; the __first argument is to the model which has existed to update__, the __second argument is path to the new weekly updated X-ray images__.  

```inst.update_trained_model_with_test(model=paths_to_existing_model, paths=paths_new_imgs)```
```inst.update_trained_model_produce(model=paths_to_existing_model, paths=paths_new_imgs)```

### Effect of poisoning attack 

Assume either type of attacks has happened as discussed in [types of attack](https://github.com/dahmansphi/attackai?tab=readme-ov-file#type-one-attack). We could see the effect of that attack on the accuracy of the model using:

```inst.update_trained_model_produce(model=paths_to_existing_model, paths=paths_new_imgs)``` 

and

```inst.predict(img_path=path_to_imgs, model_path=orignial_model)```

Observe that the prediction tasks which we hit 100% for both classes is now at the same level of accuracy for __normal images__, _however_ it is almost __0%__ for the other class of __pneumonia images__. that means if a person actually has cancer the model says they are not; as presented in the figures below. 

__result of updated model accuracy__

![prediction by attacked model](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/predict_attack_2.png)

__result of prediction on normal classes__

![prediction by model for normal imgs](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/predict_1.png)

__result of prediction on pneumonia classes__

![results by attacked model](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/predict_2_attacked_pnemonia.png)


> [!CAUTION]
> You can see the effect of such an attack how misleading it is. This is a sample illustration that we can see its effect after one round, however such idea could go unpredicted for some time by poisoning the model slowly until it loses its main core of ratios.

## Secure layer using the __norm culture__ solution

As I have discussed earlier in the simulative storyline example, if the AI team decided to add a security layer before the actual update happens. This is where I do introduce the __norm culture__ solution. The __mathematical__ intuition behind the method can be found in the academic published paper, however in this demonstration I will show how such layer could help to capture the images of which they have been attacked by either type one or two.   

![norm cuture logo](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/norm_culture.png)

In the following figure you can see where the location of this __norm culture__ box. It will be basically the stage where you can check if the __folder of the new training images__ are compromised in anyway. To employ the __norm culture box__ we shall be using `inst.protect_me()` function which in principle calculate four main elements:
1. __two numpy__ where each represents the number of classes called __alpha__
2. __two norms__ where each represents the number of classes called __norms__
3. __two flatten__ where each represents the number of classes called __flatten__

the figure illustrate that:

![norm images](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/norms_imgs.png)

the following functions illustrates the implementation with two arguments __the path of the original clean training dataset__ and __path to where save the three elements__:

```inst.protect_me(paths=paths, save_path_norms=save_folder)```


Here are the use in sequece two functions `inst.inspect_attack_one()`, and `inst.inspect_attack_two()`. 

### inspect_attack_one()
This function as the name implies is to inspect the new batch of images if they are compromised in anyway with the __first type of attack__. it takes __two arguments__ the first one is the __path to inspect__ and the second is the path to the model __norms__ which is discussed above. 
the requrecruitment of this function using the function:

```inst.inspect_attack_one(paths_to_inpect=paths_to_inspect, paths_to_norms=paths_to_norm)```

here is figure shows the result with inspecting the folder of which that had been attacked prviously.

[type one attack caught by inpsect](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/catch_type_one_1.png)
[type one attack caught by inpsect](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/catch_type_one_2.png)

### inspect_attack_two()
This function as the name implies is to inspect the new batch of images if they are compromised in anyway with the __second type of attack__. two important arguments, __first argument is path to the folder that need to be inspected__, and the second __path to original predictive model that has been proved to be clean orginally__. the requrecruitment of this function using the function:

```inst.inspect_attack_two(paths_to_inpect=paths_to_inspect, model_verified=updated_trained_model_path)```

here is figure shows the result with inspecting the folder of which that had been attacked prviously.

[type one attack caught by inpsect](https://raw.githubusercontent.com/dahmansphi/protectai/main/assets/type_two_image_result.png)


## Conclusion on installation and employing 
I hope this simulation shows the idea of such attack, its effect and a proposal for defence. 


# Reference

please follow up on the project page to find the academic published paper on the project
 
