import igraph
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
import plotly.io as pio

def create_tree_plot(parent_child, labels):
    # Create a directed graph
    G = Graph(directed=True)
    
    # Add vertices
    vertices = set([p for p, c in parent_child] + [c for p, c in parent_child])
    G.add_vertices(list(vertices))
    
    # Add edges
    G.add_edges(parent_child)
    
    # Layout
    lay = G.layout('rt')

    position = {k: lay[k] for k in range(len(G.vs))}
    Y = [lay[k][1] for k in range(len(G.vs))]
    M = max(Y)

    es = EdgeSeq(G) # sequence of edges
    E = [e.tuple for e in G.es] # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2*M-position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

    labels = [labels.get(v.index, str(v.index)) for v in G.vs]

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                       y=Ye,
                       mode='lines',
                       line=dict(color='rgb(210,210,210)', width=1),
                       hoverinfo='none'
                       ))
    fig.add_trace(go.Scatter(x=Xn,
                      y=Yn,
                      mode='markers+text',
                      name='',
                      marker=dict(symbol='circle-dot',
                                    size=10,  # Reduced marker size
                                    color='#6175c1',
                                    line=dict(color='rgb(50,50,50)', width=1)
                                    ),
                      text=labels,
                      hoverinfo='text',
                      opacity=0.8,
                      textposition="top center",
                      textfont=dict(size=8)  # Reduced font size
                      ))

    # Update layout for a larger graph
    fig.update_layout(
        showlegend=False,
        width=2000,  # Increased width
        height=1000,  # Increased height
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # Save as interactive HTML
    pio.write_html(fig, file='tree_interactive.html', auto_open=True)

    print("Interactive HTML file 'tree_interactive.html' has been generated.")

# Parse the input data
parent_child = []
with open('modified_cifar.parent-child.txt', 'r') as f:
    for line in f:
        parent, child = map(int, line.strip().split())
        parent_child.append((parent, child))

labels = {}
with open('class_names.txt', 'r') as f:
    for line in f:
        number, label = line.strip().split(' ', 1)
        labels[int(number)] = label

# Create the tree plot
create_tree_plot(parent_child, labels)