
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from flask import render_template_string



# Assuming POISONED_WORKER_IDS is defined somewhere in your code
#POISONED_WORKER_IDS = [15, 35, 44, 16, 26, 18, 27, 8, 41, 28]


def extract_gradients(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Flatten the data and keep track of the worker ids
    gradients = []
    worker_ids = []
    for round_number, workers in data.items():
        for worker_id, gradient in workers.items():
            gradients.append(gradient)
            worker_ids.append(int(worker_id))
    return gradients, worker_ids


'''def visualize_3d(gradients, worker_ids,trace):
    grad = np.array(gradients)

    # PCA Reduction
    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(grad)


    # Determine color based on whether the worker is poisoned or not
    colors = ['blue' if worker_id in POISONED_WORKER_IDS else 'orange' for worker_id in worker_ids]

    # DataFrame Preparation
    df = pd.DataFrame({
        'x': vis_dims[:, 0],
        'y': vis_dims[:, 1],
        'z': vis_dims[:, 2],
        'worker_id': worker_ids,
        #'cluster': cluster_list,
        'color': colors  # Add color information for each point
    })

    # Visualization setup
    fig = go.Figure()

    # Add traces for general workers with adjusted opacity
    for worker_id in set(worker_ids) - {5, 3}:  # Exclude worker 5 and 3
        df_worker = df[df['worker_id'] == worker_id]
        fig.add_trace(go.Scatter3d(
            x=df_worker['x'],
            y=df_worker['y'],
            z=df_worker['z'],
            mode='markers',
            marker=dict(size=3, color=df_worker['color'].iloc[0], opacity=0.8),
            name=f'Worker {worker_id}'
        ))

    # Add traces for workers 5 and 3 with 100% opacity and lines
    if trace:
        for worker_id in [5, 3]:
            df_worker = df[df['worker_id'] == worker_id]
            fig.add_trace(go.Scatter3d(
                x=df_worker['x'],
                y=df_worker['y'],
                z=df_worker['z'],
                mode='markers+lines',
                marker=dict(size=5, color=df_worker['color'].iloc[0], opacity=1.0),  # Ensure visibility
                line=dict(width=2, color='black'),  # Distinct line color
                name=f'Traced Worker {worker_id}'
            ))

    # Update layout for better visualization
    fig.update_layout(width=900, height=900, uniformtext_minsize=12, uniformtext_mode='hide')
    fig.update_scenes(
        xaxis=dict(showbackground=True, backgroundcolor="rgb(211, 211, 211)", gridcolor="rgb(150, 150, 150)", zeroline=False),
        yaxis=dict(showbackground=True, backgroundcolor="rgb(211, 211, 211)", gridcolor="rgb(150, 150, 150)", zeroline=False),
        zaxis=dict(showbackground=True, backgroundcolor="rgb(211, 211, 211)", gridcolor="rgb(150, 150, 150)", zeroline=False))
    fig.show()'''

def visualize_3d(gradients, worker_ids,trace,POISONED_WORKER_IDS, wid):
    grad = np.array(gradients)

    # PCA Reduction
    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(grad)

    '''# KMeans Clustering
    kmeans_model = KMeans(n_clusters=2, n_init=10)
    kmeans_model.fit(vis_dims)
    cluster_list = kmeans_model.labels_'''

    # Determine color based on whether the worker is poisoned or not
    colors = ['blue' if worker_id in POISONED_WORKER_IDS else 'orange' for worker_id in worker_ids]

    # DataFrame Preparation
    df = pd.DataFrame({
        'x': vis_dims[:, 0],
        'y': vis_dims[:, 1],
        'z': vis_dims[:, 2],
        'worker_id': worker_ids,
        #'cluster': cluster_list,
        'color': colors  # Add color information for each point
    })

    # Visualization setup
    fig = go.Figure()

    # Add traces for general workers with adjusted opacity
    print("wid: ")
    print(wid)
    for worker_id in set(wid):
        df_worker = df[df['worker_id'] == worker_id]
        fig.add_trace(go.Scatter3d(
            x=df_worker['x'],
            y=df_worker['y'],
            z=df_worker['z'],
            mode='markers',
            marker=dict(size=3, color=df_worker['color'].iloc[0], opacity=0.8),
            name=f'Worker {worker_id}'
        ))

    # Add traces for workers 5 and 3 with 100% opacity and lines
    if trace:
        for worker_id in [5, 3]:
            df_worker = df[df['worker_id'] == worker_id]
            fig.add_trace(go.Scatter3d(
                x=df_worker['x'],
                y=df_worker['y'],
                z=df_worker['z'],
                mode='markers+lines',
                marker=dict(size=5, color=df_worker['color'].iloc[0], opacity=1.0),  # Ensure visibility
                line=dict(width=2, color='black'),  # Distinct line color
                name=f'Traced Worker {worker_id}'
            ))

    # Update layout for better visualization
    fig.update_layout(width=900, height=900, uniformtext_minsize=12, uniformtext_mode='hide',title="PCA Of Each Selected Worker")
    fig.update_scenes(
        xaxis=dict(showbackground=True, backgroundcolor="rgb(211, 211, 211)", gridcolor="rgb(150, 150, 150)", zeroline=False),
        yaxis=dict(showbackground=True, backgroundcolor="rgb(211, 211, 211)", gridcolor="rgb(150, 150, 150)", zeroline=False),
        zaxis=dict(showbackground=True, backgroundcolor="rgb(211, 211, 211)", gridcolor="rgb(150, 150, 150)", zeroline=False))

    fig.show()
    image_path = './static/css/pca.png'
    fig.write_image(image_path)
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    return plot_html,image_path

#gradients, worker_ids = extract_gradients('final/result/workers/10_20_0.5_gradients.json')
#visualize_3d(gradients, worker_ids,trace=False)
#visualize_3d_s(gradients, worker_ids, selected_worker_ids=[4],trace_l=[])
def visualize_3d_by_round_f(gradients_by_round,start_round, end_round):
    round_ids = []
    aggregated_gradients = []

    # Aggregate gradients by round
    for round_id, worker_gradients in gradients_by_round.items():
        if start_round <= int(round_id) <= end_round:
            # Sum up gradients for each worker to get a single gradient per round
            round_gradient = np.sum(np.array(list(worker_gradients.values())), axis=0)
            aggregated_gradients.append(round_gradient)
            round_ids.append(round_id)

    # Convert aggregated gradients to embeddings-like structure for PCA
    embs = np.array(aggregated_gradients)

    # PCA Reduction
    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(embs)

    # DataFrame Preparation
    df = pd.DataFrame({
        'x': vis_dims[:, 0],
        'y': vis_dims[:, 1],
        'z': vis_dims[:, 2],
        'round_id': round_ids,
        'frame': round_ids  # Add frame identifier for animation
    })

    # Create a figure with all points in grey to begin with
    fig = px.scatter_3d(df, x='x', y='y', z='z', hover_name='round_id', color_discrete_sequence=['grey'])

    # Setting initial range and aesthetics for the scatter plot
    fig.update_layout(
        scene=dict(
            xaxis=dict(**{'backgroundcolor': "rgb(211, 211, 211)",
                         'gridcolor': "rgb(150, 150, 150)", 'zeroline': False}),
            yaxis=dict(**{'backgroundcolor': "rgb(211, 211, 211)",
                         'gridcolor': "rgb(150, 150, 150)", 'zeroline': False}),
            zaxis=dict(**{'backgroundcolor': "rgb(211, 211, 211)",
                         'gridcolor': "rgb(150, 150, 150)", 'zeroline': False})
        ),
        width=900,
        height=900,
        title="Epoch Visualization"
    )

    # Update traces to set all points initially in grey
    fig.update_traces(marker=dict(size=3, color='grey'))

    # Define frames to highlight each epoch one by one
    frames = [go.Frame(
        data=[go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            mode='markers',
            marker=dict(size=3, color=['#1f77b4' if idx == frame else 'grey' for idx in df.index])
        )],
        name=str(rid)
    ) for frame, rid in enumerate(round_ids)]

    fig.frames = frames

    # Add slider for animation control
    sliders = [{
        'steps': [{'method': 'animate', 'args': [[frame.name],
                 {'mode': 'immediate', 'frame': {'duration': 500, 'redraw': True}, 'transition': {'duration': 0}}],
                 'label': frame.name} for frame in fig.frames],
        'transition': {'duration': 0},
        'x': 0.1,
        'len': 0.88,
        'xanchor': 'left',
        'y': 0,
        'yanchor': 'bottom',
        'currentvalue': {'visible': True, 'prefix': 'Round: ', 'xanchor': 'right'},
        'pad': {'b': 10, 't': 50}
    }]

    fig.update_layout(sliders=sliders)

    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                 "label": "Play",
                 "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate",
                                   "transition": {"duration": 0}}],
                 "label": "Pause",
                 "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": True,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )

    fig.show()
    image_path = './static/css/round.png'
    fig.write_image(image_path)
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    return plot_html,image_path
