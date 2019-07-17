# Identify Numbers with Perceptrons
This approach identifies digits in the MNIST dataset using 10 perceptrons. This is not an optimal solution for this dataset as the data is not linearly separable. However, the 10 perceptrons do somewhat well, considering the data isn't ideal. You will find that 8 is commonly misinterpreted and 1 is usually categorized correctly. You can tell this by checking the diagonal of the confusion matrix (from top left to bottom right diagonal) and it's complement. What's not on the diagonal are miscategorized inputs. I used learning rates 1.0, 0.1, 0.01, and 0.001 and saw no better than 95% accuracy rates.

I will be creating a new solution using multi-layer neural nets next and we should see an improvement of accuracy with that approach, since I will be using a sigmoid function which doesn't rely on linearity in the data. 

##### Housekeeping:

Feel free to use this code by citing it if you are using it for something academic, otherwise you may be penalized by your instructor for plagerizm; as I am creating this approach (along with my neural net approach) for CS 445 at Portland State University.

Run using command:
mnist_perceptron.py 0.01 

Where 0.01 is your learning rate, which can be whatever you wish it to be.

Thank you!
