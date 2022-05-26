# fruitCounting

In this work I mainly build on Adrian's research, the same datasets, same CAN model, but make an analysis of small batch size training in fruit counting.

https://github.com/adrianxsalazar/Deep_Regression_vs_Detection_for_Counting_in_Robotic_Phenotyping


## Dataset
In the batch size analysis, we use Apple, Mango and Almond to make experiments.


## Training / Transfer learning 
I will take the trainApple.py as example,

the batchSize() function in trainApple.py, we can use

checkpoint = torch.load(loadModel)
model.load_state_dict(checkpoint['model_state_dict'])

to load the coutning model in source doamin, then obtain the results with different batch size and learning rate.


## Linear Regression Model


