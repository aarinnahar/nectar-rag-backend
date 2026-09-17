import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.orchestration.agent_state import AgentState


def prepare_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Safely converts specified DataFrame columns to numeric types, coercing invalid values to NaN."""
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def prepare_scatter_dataframe(data: dict) -> pd.DataFrame:
    """Normalizes state evaluation metrics into a clean DataFrame for scatter plotting."""
    df = pd.DataFrame(data).T.reset_index().rename(columns={"index": "strategy"})

    # Key Aliasing: Normalize column names from cost and latency pipelines
    column_aliases = {
        "avg_input_tokens_per_query": "avg_tokens_per_query",
        "latency_ms": "avg_vector_search_latency_ms",
        "vector_search_latency_ms": "avg_vector_search_latency_ms"
    }
    df = df.rename(columns=column_aliases)

    # Defaults for missing optional keys
    if "avg_tokens_per_query" not in df.columns:
        df["avg_tokens_per_query"] = 0.0
    if "avg_vector_search_latency_ms" not in df.columns:
        df["avg_vector_search_latency_ms"] = 1.0  # Fallback for marker sizing

    numeric_cols = ["avg_tokens_per_query", "context_recall", "avg_vector_search_latency_ms"]
    df = prepare_numeric_columns(df, numeric_cols)

    # Ensure marker sizing column has a positive minimum threshold (prevents invisible points)
    df["marker_size"] = df["avg_vector_search_latency_ms"].apply(lambda x: max(15.0, float(x) if pd.notnull(x) else 15.0))

    return df.dropna(subset=["avg_tokens_per_query", "context_recall"])


def scatter_plot(plot_df: pd.DataFrame) -> str:
    """Generates a Quality vs. Cost & Latency scatter plot with robust marker sizing and hover templates."""
    if plot_df.empty:
        return "<div class='error'>No valid data available for Scatter Plot generation.</div>"

    fig_scatter = px.scatter(
        plot_df,
        x="avg_tokens_per_query",
        y="context_recall",
        color="strategy",
        hover_name="strategy",
        size="marker_size",
        size_max=30,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Prism,
        labels={
            "strategy": "Chunking Strategy",
            "avg_tokens_per_query": "Avg Tokens / Query (Cost Proxy)",
            "context_recall": "Context Recall",
            "avg_vector_search_latency_ms": "Search Latency (ms)"
        }
    )

    fig_scatter.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>" +
            "Avg Tokens: %{x:,.0f}<br>" +
            "Context Recall: %{y:.1%}<br>" +
            "<extra></extra>"
        )
    )

    fig_scatter.update_layout(
        title=dict(
            text="Quality vs. Cost Trade-off",
            x=0.5,
            font=dict(size=18, family="Inter", color="white")
        ),
        font=dict(family="Inter", color="white"),
        legend=dict(
            title="Strategy",
            bgcolor="rgba(30,30,30,0.8)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            x=1.02,
            y=1
        ),
        margin=dict(l=60, r=160, t=60, b=60)
    )

    fig_scatter.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
    fig_scatter.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)", range=[0, 1.05], tickformat=".0%")

    return fig_scatter.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})


def bargraph(data: dict) -> str:
    """Generates a grouped bar chart comparing structural and quality metrics across strategies."""
    df = pd.DataFrame(data).T.reset_index().rename(columns={"index": "strategy"})

    numeric_columns = ["context_recall", "context_precision", "boundary_health", "intra_chunk_coherence"]
    df = prepare_numeric_columns(df, numeric_columns)

    colors = ["#00E5FF", "#00FF88", "#FF9F43", "#E040FB"]

    fig_bar = go.Figure()

    metric_labels = [
        ("context_recall", "Context Recall", colors[0]),
        ("context_precision", "Context Precision", colors[1]),
        ("boundary_health", "Boundary Health", colors[2]),
        ("intra_chunk_coherence", "Intra-Chunk Coherence", colors[3]),
    ]

    for col_key, display_name, color in metric_labels:
        if col_key in df.columns:
            fig_bar.add_trace(
                go.Bar(
                    name=display_name,
                    x=df["strategy"],
                    y=df[col_key],
                    marker_color=color,
                    text=df[col_key],
                    texttemplate="%{y:.1%}",  # Formats raw 0.64 float directly as 64.0%
                    textposition="outside",
                    textfont=dict(color="white", size=10)
                )
            )

    fig_bar.update_layout(
        template="plotly_dark",
        barmode="group",
        title=dict(
            text="Multi-Dimensional Quality & Structural Health",
            x=0.5,
            font=dict(size=18, family="Inter", color="white")
        ),
        yaxis=dict(title="Score", range=[0, 1.2], tickformat=".0%"),
        xaxis=dict(title="Chunking Strategy", tickangle=-30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(t=100, b=60, l=50, r=50)
    )

    return fig_bar.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})


def create_graphs(state: AgentState) -> dict:
    """LangGraph node execution function to generate and return Plotly HTML components."""
    data = state.get("evaluation_scores", {})

    if not data:
        print("Warning: No evaluation metrics found in state for graph generation.")
        return {"scatter_html": "", "bar_html": ""}

    # 1. Generate Figures
    bar_html = bargraph(data)
    plot_df = prepare_scatter_dataframe(data)
    scatter_html = scatter_plot(plot_df)

    # 2. Portable Path Saving (Relative to module location)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "..", "report_design")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "scatterplot.html"), "w", encoding="utf-8") as f:
        f.write(scatter_html)

    with open(os.path.join(output_dir, "bargraph.html"), "w", encoding="utf-8") as f:
        f.write(bar_html)

    print("Success! Evaluation graphs generated and saved.")

    # 3. Return updated keys to AgentState
    return {
        "scatter_html": scatter_html,
        "bar_html": bar_html
    }



































# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from src.orchestration.agent_state import AgentState
# import os






# def scatter_plot(plot_df):
#     fig_scatter = px.scatter(
#     plot_df,
#     x="total_tokens",
#     y="context_recall",
#     color="strategy",                # Categorical column → Legend
#     hover_name="strategy",           # Strategy name on hover
#     size="latency_seconds",
#     title="Quality vs. Cost Trade-off",
#     size_max=35,
#     template="plotly_dark",
#     color_discrete_sequence=px.colors.qualitative.Prism,
#     labels={
#         "strategy": "Chunking Strategy",
#         "total_tokens": "Total Tokens (Cost)",
#         "context_recall": "Context Recall (Quality)",
#         "latency_seconds": "Latency (s)"
#     }
# )

#     # ---------------------------------------------------------
#     # Layout
#     # ---------------------------------------------------------
#     fig_scatter.update_layout(
#         title=dict(
#             text="Quality vs. Cost Trade-off",
#             x=0.5,
#             font=dict(size=20, family="Inter")
#         ),

#         font=dict(
#             family="Inter",
#             color="white"
#         ),

#         legend=dict(
#             title="Chunking Strategy",
#             bgcolor="rgba(40,40,40,0.8)",
#             bordercolor="gray",
#             borderwidth=1,
#             orientation="v",
#             x=1.02,
#             y=1,
#             xanchor="left",
#             yanchor="top"
#         ),

#         hoverlabel=dict(
#             bgcolor="white",
#             font_size=14,
#             font_family="Inter"
#         ),

#         margin=dict(l=70, r=170, t=70, b=70)
#     )

#     # ---------------------------------------------------------
#     # Grid Styling
#     # ---------------------------------------------------------
#     fig_scatter.update_xaxes(
#         showgrid=True,
#         gridwidth=1,
#         gridcolor="#ACA9A9"
#     )

#     fig_scatter.update_yaxes(
#         showgrid=True,
#         gridwidth=1,
#         gridcolor="#ACA9A9"
#     )
#     graph_html = fig_scatter.to_html(
#     full_html=False,
#     include_plotlyjs=False,
#     config={
#         "responsive": True,
#         "displaylogo": False
#     }
# )

#     return graph_html

#     # st.plotly_chart(fig_scatter, use_container_width=True)




# def bargraph(data):
#     # Define a vibrant color palette designed to stand out against a dark background
#     # Using neon/bright variants of the previous colors:
#     # Bright Cyan, Bright Green, Bright Orange
#     df = (pd.DataFrame(data)).T
#     df = df.reset_index()
#     df = df.copy()

#     df = df.reset_index().rename(columns={"index": "strategy"})
#     dark_theme_colors = ['#00E5FF', '#00FF88', '#FF9F43']

#     fig_bar_dark = go.Figure(data=[
#         go.Bar(
#             name='Context Recall', 
#             x=df["strategy"], 
#             y=df['context_recall'],
#             marker_color=dark_theme_colors[0],
#             text=df['context_recall'],
#             texttemplate='%{text:.2f}',
#             textposition='outside',
#             # Ensuring text on top of bars is bright (Plotly usually handles this
#             # with templates, but explicitly setting it helps)
#             textfont=dict(color='white', size=11) 
#         ),
#         go.Bar(
#             name='Context Precision', 
#             x=df["strategy"], 
#             y=df['context_precision'],
#             marker_color=dark_theme_colors[1],
#             text=df['context_precision'],
#             texttemplate='%{text:.2f}',
#             textposition='outside',
#             textfont=dict(color='white', size=11)
#         ),
#         go.Bar(
#             name='Faithfulness', 
#             x=df["strategy"], 
#             y=df['faithfullness'], 
#             marker_color=dark_theme_colors[2],
#             text=df['faithfullness'],
#             texttemplate='%{text:.2f}',
#             textposition='outside',
#             textfont=dict(color='white', size=11)
#         )
#     ])

#     # Updating the layout for the dark theme and polished aesthetics
#     fig_bar_dark.update_layout(
#         # --- The Key Change ---
#         template="plotly_dark", # Sets the entire canvas to dark mode
        
#         barmode='group', 
#         title="Retrieval vs. Generation Quality by Strategy",
#         title_x=0.5,             # Centers the title
#         title_font=dict(color='white', size=22, family="Inter"), # Classic dark-mode look

#         yaxis_title="Score (0 to 1)",
#         xaxis_title="Chunking Strategy",
#         xaxis_tickangle=-45,     # Keep the readable angle
#         uniformtext_minsize=10, 
#         uniformtext_mode='hide',

#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.05,              # Legend still on top
#             xanchor="center",
#             x=0.5,
#             font=dict(color='white'),
#             bgcolor='rgba(0,0,0,0)', # Full transparency against dark background
#             bordercolor='rgba(255,255,255,0.2)' # Subtle light border
#         ),
#         margin=dict(t=100)       # Margin so title/legend don't overlap
#     )

#     # Optional: Ensure gridlines are subtle but visible against the dark canvas
#     fig_bar_dark.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)',range=[0, 1.0])

#     return fig_bar_dark.to_html(full_html=False,include_plotlyjs=False)
#     # st.plotly_chart(fig_bar_dark, use_container_width=True)



# def create_graphs(state:AgentState):
#     data = state['combined_matrics']
#     # data = state['combined_matrics']
#     bar = bargraph(data)
#     df = (pd.DataFrame(data)).T
#     plot_df = df.reset_index().rename(columns={"index": "strategy"})
#     scatter  = scatter_plot(plot_df)

    
#     # 1. Set your directory
#     output_dir = r"D:\Practive Projects\Chunking_Eval\src\report_design"
#     scatter_file = f"scatterplot.html"
#     bar_file = f"bargraph.html"

#     # 2. CREATE the folder if it doesn't exist (the magic line)
#     os.makedirs(output_dir, exist_ok=True)

#     # 3. Combine them into a full path
#     final_destination1 = os.path.join(output_dir, scatter_file)
#     final_destination2 = os.path.join(output_dir, bar_file)

#     # 4. Save the file
#     with open(final_destination1, "w", encoding="utf-8") as f1:
#         f1.write(scatter)

#     # 5. Save the file
#     with open(final_destination2, "w", encoding="utf-8") as f2:
#         f2.write(bar)

#     print(f"Success! Graph Saved ")

#     return {}
