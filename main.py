import numpy as np
import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#function for updating the plot
def update_line(g,x,y):
    g.set_xdata(np.append(g.get_xdata(), x))
    g.set_ydata(np.append(g.get_ydata(), y))

    ax.relim()
    ax.autoscale_view(True,True,True)
    plt.draw()
    plt.pause(0.0001)

#--------CLASSES--------------
# Dense Layer
class Layer_Dense:
    # layer initialization. The constructor
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # method: forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # calculate output values from inputs, weight and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot( self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-inputs))
    def backward(self, dvalues):
        # self.dvalues = dvalues * (np.exp(-self.inputs)/( pow(1+ np.exp(-self.inputs),2) )) #derivata della sigmoide
        self.dvalues = dvalues*(1-self.output)*self.output

class Binary_Loss:
    def calculate(self, targets, predictions):
        predictions = np.clip(predictions, 1e-7, 1-1e-7) #si regola min e max tale che min-->o e max-->1 ma non ci arrivi mai
        # self.output = -targets*np.log(predictions) - (1-targets)*np.log(1-predictions)
        self.output = np.mean(-targets*np.log(predictions) - (1-targets)*np.log(1-predictions), axis=-1)
        #self.output = -targets * np.log(predictions) - (1 - targets) * np.log(1 - predictions)
    def backward(self, target, predictions):
        normalization = len(predictions)
        n_outputs = len(predictions[0])  # numero degli output
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7) #evitare divisioni per 0 su predictions
        dvalues = -((target/predictions)-((1-target)/(1-predictions)))/n_outputs
        self.dvalues = dvalues/normalization  # Normalizzazione per non far esplodere valori


class Optimizer:
    def __init__(self, lr=5.0):
        self.lr = lr

    def update(self, layer):
        layer.weights = layer.weights - self.lr*layer.dweights
        layer.biases = layer.biases - self.lr*layer.dbiases

#---------------------------------------------------------------------------

with open("diabetes.csv", 'r') as file:
    #reading
    csvFile = csv.reader(file)

    #creating list with labels
    label = file.readline()
    label = label.strip().split(',')
    X_label = label[:-1] #labels of the input values without the outcome label
    #print(X_label)
    y_label = label[8:]  #label with only "outcome"
    #print(y_label)

    #create empty input and target lists
    X = []
    y = []
    #filling X with all the inputs withouth the "outcome" column
    #filling y with the outputs quindi la colonna "outcome"
    for row in csvFile:
        X.append([row[label.index(column)] for column in X_label])
        y.append([row[label.index(column)] for column in y_label])

X_train = X[:256]
y_train = y[:256]
#converting lists in array with float casting
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

X_valid = X[256:512]
y_valid = y[256:512]
#converting lists in array with float casting
X_valid = np.array(X_valid, dtype=np.float32)
y_valid = np.array(y_valid, dtype=np.float32)

X_test = X[512:]
y_test = y[512:]
#converting lists in array with float casting
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

#preparing the plot
ax = plt.gca()
ax.set_autoscale_on(True)
#label of axes
steps = np.linspace(-1,2,100)
plt.ylabel('loss function')
plt.xlabel('epochs')
g_train_loss, = plt.plot([],[])
g_train_accuracy, = plt.plot([],[])
g_valid_loss, = plt.plot([],[])
g_valid_accuracy, = plt.plot([],[])
ax.legend([g_train_loss,g_train_accuracy,g_valid_loss,g_valid_accuracy], ['train_loss','train_accuracy','valid_loss','valid_accuracy'])



#-----> Now all the inputs X and targets y are ready!

#initialization
activation1 = Activation_Sigmoid()
activation2 = Activation_Sigmoid()
binary_loss = Binary_Loss()
layer1 = Layer_Dense(8,64)
layer2 = Layer_Dense(64, 1)
optimizer = Optimizer(lr=0.001)

#TRAINING
print("TRAINING:")
for epoch in range(10001):
    layer1.forward(X_train)
    activation1.forward(layer1.output)

    #forward su layer2
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    #activation_sigm.output

    #calcolo loss function
    binary_loss.calculate(y_train, activation2.output)

    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y_train)
    loss = np.mean(binary_loss.output)

    #backward
    binary_loss.backward(y_train,activation2.output)
    activation2.backward(binary_loss.dvalues)
    layer2.backward(activation2.dvalues)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dvalues)

    #ottimizzazione learning rate e pesi
    optimizer.update(layer1)
    optimizer.update(layer2)

    if not epoch % 100:
        print("Epoch:", epoch, "Loss:", loss, "Accuracy:", accuracy)
        update_line(g_train_loss, epoch, np.mean(binary_loss.output))
        update_line(g_train_accuracy, epoch, accuracy)
        # Validation
        layer1.forward(X_valid)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        binary_loss.calculate(y_valid, activation2.output)

        predictions = (activation2.output > 0.5) * 1
        accuracy = np.mean(predictions == y_valid)
        update_line(g_valid_loss, epoch, np.mean(binary_loss.output))
        update_line(g_valid_accuracy, epoch, accuracy)
        print("   Validation Loss:", np.mean(binary_loss.output), "Validation Accuracy:", accuracy)

#Forward test
print("\n")
print("TESTING:")
layer1.forward(X_test)
activation1.forward(layer1.output)

#forward su layer2
layer2.forward(activation1.output)
activation2.forward(layer2.output)
#activation_sigm.output

#calcolo loss function
binary_loss.calculate(y_test, activation2.output)

predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)
print("Loss:",np.mean(binary_loss.output),"Accuracy:",accuracy)
plt.show()
# for i in range(0, y_test.shape[0]):
#    print("network output:", activation2.output[i],"expected output:",y_test[i])
