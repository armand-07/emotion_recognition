import numpy as np
import matplotlib.pyplot as plt
import math
from src import AFFECTNET_CAT_EMOT, MODELS_DIR
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import torch
from torch.distributions import Categorical


def visualize_batch(img, cat_label = None, col = 8):
    img = img.numpy().transpose((0, 2, 3, 1)) # Convert the image from [B,C,H,W] to [B,H,W,C]
    row = math.ceil(img.shape[0]/col)
    fig = plt.figure(figsize=(10,1.5*row))

    # Set a general title for the figure
    if cat_label is not None:
        fig.suptitle('Batch visualization with categorical annotation', fontsize=12, y=0.95)
    
    for i in range(img.shape[0]):
        ax = fig.add_subplot(row, col, i+1, xticks=[], yticks=[])
        ax.imshow(img[i])
        if cat_label is not None:
            ax.set_title(AFFECTNET_CAT_EMOT[np.argmax(cat_label[i]).item()], fontsize=10)
    plt.show()

def create_conf_matrix(conf_matrix):
    """Create a confusion matrix using the plotly library."""
    fig = px.imshow(conf_matrix,
                labels=dict(y="True Emotion", x="Predicted Emotion", color="Percentage"),
                x=AFFECTNET_CAT_EMOT,
                y=AFFECTNET_CAT_EMOT,
                color_continuous_scale='Blues',
                zmin = 0.0,
                zmax = 1.0,
               )
    fig.update_xaxes(side="top", tickangle=0, title_font=dict(size=14), tickfont=dict(size=9))
    fig.update_yaxes(tickangle=0, title_font=dict(size=14), tickfont=dict(size=9))
    fig.update_layout(
    coloraxis_colorbar=dict(thickness=15, len = 0.8)
    )
    fig.update_traces(
    hovertemplate='True Emotion: %{y}<br>Predicted Emotion: %{x}<br>Percentage: %{z:.4f}<extra></extra>'
    )
    return fig


def compute_cat_label_batch_entropy(dataloader, NUMBER_OF_EMOT, title = 'Entropy through all batches'):
    """Compute the entropy of the categorical labels in a batch using the class AffectNetDatasetValidation.
    """
    results = []
    max_entropy = - np.log(1/NUMBER_OF_EMOT) # uniform distribution
    for i, (_, _ , cat_labels, _) in enumerate(dataloader):
        cat_labels = np.argmax(cat_labels, axis=-1) # select the max categorical emotion in case a soft encoding has been applied
        # Count the occurrences of each label
        counts = np.bincount(cat_labels)
        # Convert the counts into probabilities to get a distribution
        probs = counts / np.sum(counts)
        # Convert the numpy array to a PyTorch tensor
        probs_tensor = torch.from_numpy(probs)
        entropy = Categorical(probs = probs_tensor).entropy().item() # entropy is calculated using the natural logarithm
        results.append({'Batch': i, 'Entropy': entropy})
    
    # Display the results
    # Convert the list to a DataFrame
    df = pd.DataFrame(results)
    # Create a barplot 
    bar_plot = alt.Chart(df).mark_bar(color="steelblue").encode(
        alt.X("Batch:N", axis=alt.Axis(labelAngle=0, labelFlush = True, title="Batch id", values=list(range(0, df['Batch'].max(), 5)))),
        alt.Y('Entropy:Q', title="Entropy of emotion ocurrences"),
        tooltip=[
            alt.Tooltip('Batch', title='Batch id'),
            alt.Tooltip('Entropy:Q', title = 'Entropy', format='.3f')]
    ).properties(
        title=alt.TitleParams(text= title, fontSize=16),
        width=300,
        height=300
    )
    # Create a horizontal line chart
    line = alt.Chart(pd.DataFrame({'y': [max_entropy]})).mark_rule(color='black').encode(y=alt.Y('y', title =''))
    # Add text to the line

    # Create a text chart
    text = alt.Chart(pd.DataFrame({'y': [max_entropy]})).mark_text(dy = -10).encode(
        y=alt.Y('y:Q', title =''),
        text=alt.value('Uniform Distribution Entropy: ' + str(round(max_entropy, 2)))  # Redondear el valor promedio a 2 decimales
    )
    # Combine the bar plot and the line chart
    chart = bar_plot + text + line

    return chart

