#import matplotlib.pyplot as plt
import plotly.express as px
import torch
import torch.nn as nn

device = torch.accelerator.current_accelerator () .type if torch.accelerator.is_available () else "cpu"
print ( f"{torch.__version__=}" )
print ( f"{device=}" )

weight = 0.7 ; bias = 0.3
start = 0 ; end = 1 ; step = 0.02

# Input tensor (matrix) of shape (n_samples, n_features) = (50, 1)
X = torch.arange ( start , end , step ) .unsqueeze ( dim = 1 ) ; print ( f"{X[:5]=} {X.shape=} {X.grad=} {X.is_leaf=}" )
# Output tensor (matrix) of shape (n_samples, n_outputs) = (50, 1)
y = weight * X + bias ; print ( f"{y[:5]=} {y.shape=} {y.grad=} {y.is_leaf=}" )

# Create a train / test split
train_split = int ( 0.8 * len ( X ) )
X_train , y_train = X [ : train_split ] , y [ : train_split ]
X_test , y_test = X [ train_split : ] , y [ train_split : ]

def plot_predictions ( train_data = X_train , 
                     train_labels = y_train , 
                     test_data = X_test , 
                     test_labels = y_test , 
                     predictions = None ) :
    """
    Plots training data, test data and compares predictions.
    """
    train_x = train_data.detach().cpu().view(-1).tolist()
    train_y = train_labels.detach().cpu().view(-1).tolist()
    test_x = test_data.detach().cpu().view(-1).tolist()
    test_y = test_labels.detach().cpu().view(-1).tolist()

    x_values = train_x + test_x
    y_values = train_y + test_y
    series = ["Training data"] * len(train_x) + ["Testing data"] * len(test_x)

    if predictions is not None:
        pred_y = predictions.detach().cpu().view(-1).tolist()
        x_values += test_x
        y_values += pred_y
        series += ["Predictions"] * len(test_x)

    fig = px.scatter(
        x=x_values,
        y=y_values,
        color=series,
        labels={"x": "Input", "y": "Target", "color": "Series"},
        color_discrete_map={
            "Training data": "blue",
            "Testing data": "green",
            "Predictions": "red",
        },
        width=1000,
        height=700,
    )
    fig.update_traces(marker={"size": 8})
    fig.update_layout(legend_title_text="")
    fig.show()

plot_predictions ( X_train , y_train , X_test , y_test )

class LinearRegressionModel ( nn.Module ) :
    def __init__ ( self ) :
        super () .__init__ () # call the constructor of the parent class nn.Module
        self.linear_layer = nn.Linear ( in_features = 1 , out_features = 1 ) # create a linear layer with 1 input feature and 1 output feature

    def forward ( self , x : torch.Tensor ) -> torch.Tensor : # define the forward pass of the model
        return self.linear_layer ( x ) # pass the input through the linear layer and return the output

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
torch .manual_seed ( 42 ) # set the random seed for reproducibility
model = LinearRegressionModel ()
print ( f"{model=}, {list ( model.parameters () )=}" )

next ( model.parameters () ) .device
# Set model to GPU if it's available, otherwise it'll default to CPU
model .to ( device ) # the device variable was set above to be "cuda" if available or "cpu" if not
print ( f"{next ( model .parameters () ) .device=}" )

# Create loss function
loss_fn = nn.L1Loss ()
# Create optimizer
optimizer = torch.optim.SGD ( params = model .parameters () , lr = 0.01 )

torch.manual_seed ( 42 )

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 100

# Put data on the available device
# Without this, error will happen (not all model/data on device)
X_train = X_train .to ( device )
X_test = X_test .to ( device )
y_train = y_train .to ( device )
y_test = y_test .to ( device )

for epoch in range ( epochs ) :
    ### Training

    # Put model in training mode (this is the default state of a model)
    model.train ()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model ( X_train )
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn ( y_pred , y_train )

    # 3. Zero grad of the optimizer
    optimizer .zero_grad ()

    # 4. Loss backwards
    loss .backward ()

    # 5. Progress the optimizer
    optimizer .step ()

    ### Testing

    # Put the model in evaluation mode
    model .eval ()

    with torch.inference_mode () :
      # 1. Forward pass on test data
        test_pred = model ( X_test )

      # 2. Caculate loss on test data
        test_loss = loss_fn ( test_pred , y_test )

    if epoch % 10 == 0 :
        print(f"{epoch=} | {loss=} | {test_loss=} ")
        plot_predictions ( X_train , y_train , X_test , y_test , test_pred )

# Find our model's learned parameters
from pprint import pprint # pprint = pretty print, see: https://docs.python.org/3/library/pprint.html 
print ( f"{model .state_dict ()=}" )

# Turn model into evaluation mode
model .eval ()
# Make predictions on the test data
with torch.inference_mode():
    y_preds = model ( X_test )
print( f"{y_preds=}")