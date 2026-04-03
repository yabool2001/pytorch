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
model_0 = LinearRegressionModel ()

with torch.inference_mode () : # turn on inference mode (also known as evaluation mode) to speed up calculations and reduce memory usage
    y_preds = model_0 ( X_test ) # make predictions with the untrained model
print ( f"{y_preds=} {y_preds.grad_fn=}" )
plot_predictions ( predictions = y_preds )

loss_fn = nn.L1Loss () # create a loss function (also known as criterion) for regression problems
optimizer = torch.optim.SGD ( params = model_0.parameters () , lr = 0.01 ) # create an optimizer for the model's parameters with a learning rate of 0.01

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 100

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range ( epochs ) :
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train ()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0 ( X_train )
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn ( y_pred , y_train )

    # 3. Zero grad of the optimizer
    optimizer.zero_grad ()

    # 4. Loss backwards
    loss.backward ()

    # 5. Progress the optimizer
    optimizer.step ()

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.item())
        test_loss_values.append(loss.item())
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {loss} ")

    ### Testing

    # Put the model in evaluation mode
    model_0.eval ()

    with torch.inference_mode () :
      # 1. Forward pass on test data
        test_pred = model_0(X_test)

      # 2. Caculate loss on test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

        # Print out what's happening
        if epoch % 10 == 0:
            epoch_count.append ( epoch )
            train_loss_values.append ( test_loss.item () )
            test_loss_values.append ( test_loss.item () )
            print(f"Epoch: {epoch} | MAE Train Loss: {test_loss} | MAE Test Loss: {test_loss} ")

# Plot the loss curves
loss_fig = px.line(
    x=epoch_count + epoch_count,
    y=train_loss_values + test_loss_values,
    color=["Train loss"] * len(epoch_count) + ["Test loss"] * len(epoch_count),
    markers=True,
    labels={"x": "Epochs", "y": "Loss", "color": "Series"},
    title="Training and test loss curves",
    color_discrete_map={
        "Train loss": "blue",
        "Test loss": "orange",
    },
)
loss_fig.update_layout(legend_title_text="")
loss_fig.show()