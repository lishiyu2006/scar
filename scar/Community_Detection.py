# module_analysis.py (Declarative Version - Final CPU Fix)
import os
import pandas as pd
import networkx as nx
import gseapy as gp
import matplotlib.pyplot as plt
from scar.configuration_manager import load_config
from typing import Set, Dict, Any, List


# ----------------------------------------------------------------------------
# 1. Configuration and Setup (No changes)
# ----------------------------------------------------------------------------

def setup_paths(config_path: str = "scar/project_settings.yaml") -> Dict[str, Any]:
    """Load configuration and define all necessary base paths."""
    config = load_config(config_path)
    paths = {
        "deg_base": config.get("com_deg_dir", "../DEG_results"),
        "ppi_base": config.get("com_ppi_output_dir", "../ppi_results"),
        "gene_base": config.get("co_data", "data"),
        "output_base": config.get("co_result", "module_results"),
        "go_genesets": config.get("go_gene_sets", "GO_Biological_Process_2021"),
    }
    os.makedirs(paths["output_base"], exist_ok=True)
    print("Configuration loaded and paths are set.")
    return paths


# ----------------------------------------------------------------------------
# 2. Data Loading Functions (Fixed)
# ----------------------------------------------------------------------------

def load_gene_list(filepath: str, col_index: int = 0) -> Set[str]:
    """Loads a single column of genes from a file into a cleaned, uppercase set."""
    if not os.path.exists(filepath):
        return set()

    df = pd.read_csv(
        filepath,
        sep="\t",  # Added this line to specify delimiter
        header=None,
        usecols=[col_index],
        dtype=str
    )
    return set(df[col_index].dropna().str.strip().str.upper())


def load_ppi_network(filepath: str) -> pd.DataFrame:
    """Loads a two-column PPI network into a cleaned, uppercase DataFrame."""
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=["gene1", "gene2"])

    # Fix #1: change separator from tab to comma to correctly parse CSV
    ppi_df = pd.read_csv(
        filepath,
        sep=",",
        header=None,
        names=["gene1", "gene2"],
        dtype=str
    )
    ppi_df["gene1"] = ppi_df["gene1"].str.strip().str.upper()
    ppi_df["gene2"] = ppi_df["gene2"].str.strip().str.upper()
    return ppi_df


# ----------------------------------------------------------------------------
# 3. Core Analysis Functions (Fixed)
# ----------------------------------------------------------------------------

def build_ppi_subgraph(ppi_df: pd.DataFrame, target_genes: Set[str]) -> nx.Graph:
    """Constructs a NetworkX graph from a PPI dataframe, filtered by a target gene set."""
    subgraph_edges = ppi_df[
        ppi_df['gene1'].isin(target_genes) & ppi_df['gene2'].isin(target_genes)
        ]
    return nx.from_pandas_edgelist(subgraph_edges, 'gene1', 'gene2')



# Paste this new function into your code
def detect_modules_with_louvain(graph: nx.Graph, seed: int = 42) -> pd.DataFrame:
    """Performs community detection using the Louvain method and returns a gene-module mapping."""
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["gene", "module"])

    print("      -> Step 1: Finding the largest connected component for Louvain...")
    # Operate only on the largest connected component; a good practice
    connected_components = list(nx.connected_components(graph))
    if not connected_components:
        return pd.DataFrame(columns=["gene", "module"])

    largest_component = max(connected_components, key=len)
    main_graph = graph.subgraph(largest_component).copy()
    print(
        f"      -> Original graph: {graph.number_of_nodes()} nodes. Largest component: {main_graph.number_of_nodes()} nodes.")

    print("      -> Step 2: Starting Louvain algorithm community detection...")
    # Use networkx's built-in community detection
    communities = nx.community.louvain_communities(main_graph, seed=seed)
    print(f"      -> Louvain algorithm finished successfully. Detected {len(communities)} modules.")

    # Convert results to the desired DataFrame format
    gene_module_map = {}
    for module_id, gene_set in enumerate(communities):
        for gene in gene_set:
            gene_module_map[gene] = module_id

    return pd.DataFrame(gene_module_map.items(), columns=["gene", "module"])


def run_go_enrichment_on_modules(
        module_df: pd.DataFrame,
        background_genes: Set[str],
        gene_sets: str,  # Ensure the type annotation is str
        min_module_size: int = 3
) -> pd.DataFrame:
    """Performs GO enrichment analysis for each module in the module DataFrame."""
    all_enrich_results = []

    # Iterate by module ID
    for mod_id, group in module_df.groupby("module"):
        genes_in_mod = group["gene"].tolist()

        # Skip if module too small
        if len(genes_in_mod) < min_module_size:
            continue

        # ==================== Final diagnostic block ====================
        # Only check the first encountered module (ID may not be 0)
        '''
        print("\n" + "=" * 50)
        print(f" Start debugging the first encountered module, ID: {mod_id}")
        print("-" * 50)

        print(f"Module {mod_id} gene count: {len(genes_in_mod)}")
        print(f"Top 20 genes in module {mod_id}: {genes_in_mod[:20]}")

        print("-" * 50)
        background_list = list(background_genes)
        print(f"Background gene count: {len(background_list)}")
        print(f"Top 20 background genes: {background_list[:20]}")
        print("=" * 50)

        # Exit after printing debug info
        print("Debug info printed. Please verify gene symbols format (e.g., 'TP53').")
        '''

        try:
            # Print a simple progress hint
            print(f"      -> Running GO enrichment for module {mod_id} ({len(genes_in_mod)} genes)...")

            enr = gp.enrichr(
                gene_list=genes_in_mod,
                gene_sets=gene_sets,
                background=list(background_genes),
                cutoff=0.05
            )

            if not enr.results.empty:
                enr.results["module"] = mod_id
                all_enrich_results.append(enr.results)

        # Catch possible network/server errors without crashing
        except Exception as e:
            print(f"GO enrichment failed for module {mod_id}: {e}")

    # Combine enrichment results across all modules into a single DataFrame
    return pd.concat(all_enrich_results, ignore_index=True) if all_enrich_results else pd.DataFrame()


# ----------------------------------------------------------------------------
# 4. Orchestration and Execution (No changes)
# ----------------------------------------------------------------------------

def find_analysis_tasks(paths: Dict[str, Any]) -> List[Dict[str, str]]:
    """Scans directories to find all tissue/celltype combinations to be analyzed."""
    tasks = []
    for tissue in os.listdir(paths["deg_base"]):
        tissue_deg_path = os.path.join(paths["deg_base"], tissue)
        if not os.path.isdir(tissue_deg_path):
            continue

        for f in os.listdir(tissue_deg_path):
            if not (f.startswith("intersection_results_") and f.endswith(".csv")):
                continue

            celltype = f.replace("intersection_results_", "").replace(".csv", "")
            ppi_file = os.path.join(
                paths["ppi_base"], tissue, f"results_{celltype}", "background_ppi_network.csv"
            )
            tasks.append({
                "tissue": tissue,
                "celltype": celltype,
                "scar_deg_file": os.path.join(tissue_deg_path, f),
                "background_genes_file": os.path.join(paths["gene_base"], tissue, "genes.tsv"),
                "ppi_file": ppi_file,
                "output_dir": os.path.join(paths["output_base"], f"{tissue}_{celltype}"),
            })
    print(f"Discovered {len(tasks)} analysis tasks.")
    return tasks


# Paste this new visualization function
# (Ensure import matplotlib.pyplot as plt and import networkx as nx at file top)
# (Ensure import matplotlib.pyplot as plt and import networkx as nx at file top)

# ✅ Use this enhanced visualization function
def visualize_network(graph_to_draw: nx.Graph, module_df: pd.DataFrame, output_path: str):
    """
    Visualize the network using module information and save as a file.
    This version is optimized to better show dense network structure.
    """
    if graph_to_draw.number_of_nodes() == 0:
        print("  - Skipping visualization for empty graph.")
        return

    print(f"  - Generating improved network visualization, saving to {output_path}...")

    # 1. Prepare node colors (unchanged)
    gene_to_module = {row['gene']: row['module'] for index, row in module_df.iterrows()}
    node_colors = [gene_to_module.get(node, -1) for node in graph_to_draw.nodes()]

    # 2. Set plotting parameters
    plt.figure(figsize=(20, 20))  # Use a larger canvas

    # ==================== Key adjustments ====================
    # a) Tune spring_layout 'k' parameter to increase node spacing
    #    Larger k yields wider spacing; scale based on node count
    k_value = 5 / (graph_to_draw.number_of_nodes() ** 0.5)

    # b) Increase iterations for a more stable layout
    pos = nx.spring_layout(graph_to_draw, seed=42, k=k_value, iterations=100)
    # ================================================

    # 3. Draw network
    nx.draw_networkx_nodes(
        graph_to_draw,
        pos,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=80,  # Slightly enlarge node size
        alpha=0.9
    )
    nx.draw_networkx_edges(
        graph_to_draw,
        pos,
        width=0.8,  # Slightly thicken edges
        alpha=0.2
    )

    # 4. Save image
    plt.title(os.path.basename(output_path).replace(".png", ""), fontsize=25)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  - Improved visualization saved successfully.")


def process_analysis_task(task: Dict[str, str], paths: Dict[str, Any]):
    """Executes the full analysis pipeline for a single tissue/celltype task."""
    print(f"\nProcessing Task: {task['tissue']} - {task['celltype']}")
    print(f"  Attempting to load PPI from: {task['ppi_file']}")
    os.makedirs(task["output_dir"], exist_ok=True)

    # 1. Load all necessary data
    background_genes = load_gene_list(task["background_genes_file"], col_index=1)
    ppi_df = load_ppi_network(task["ppi_file"])
    scar_deg_genes = load_gene_list(task["scar_deg_file"])

    if not background_genes or ppi_df.empty or not scar_deg_genes:
        print(f"  Skipping {task['tissue']}_{task['celltype']} due to missing data.")
        return

    print(
        f"  - Background genes: {len(background_genes)}, PPI edges: {len(ppi_df)}, ScAR ∩ DEG genes: {len(scar_deg_genes)}")

    # 2. Build PPI subgraph
    subgraph = build_ppi_subgraph(ppi_df, scar_deg_genes)
    print(f"  - Subgraph created with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
    if subgraph.number_of_nodes() == 0:
        print("  - Subgraph is empty. Skipping further analysis.")
        return

    # 3. Detect modules
    #module_df = detect_modules_with_leiden(subgraph)
    # Comment this out and use the line below:

    print("  - Switching to Louvain algorithm due to Leidenalg environment issues.")
    module_df = detect_modules_with_louvain(subgraph)

    print(f"  - Detected {module_df['module'].nunique()} modules.")
    module_df.to_csv(os.path.join(task["output_dir"], "gene_to_module.csv"), index=False)

    output_image_path = os.path.join(task["output_dir"], "network_visualization.png")
    visualize_network(subgraph, module_df, output_image_path)

    # 4. Run GO enrichment
    enrichment_results = run_go_enrichment_on_modules(
        module_df, background_genes, paths["go_genesets"]
    )

    if not enrichment_results.empty:
        print(f"  - GO enrichment found {len(enrichment_results)} significant terms.")
        enrichment_results.to_csv(os.path.join(task["output_dir"], "ScAR_DEGs_GO.csv"), index=False)
    else:
        print("  - No significant GO enrichment results found.")

    # 5. Save subgraph edges
    nx.write_edgelist(subgraph, os.path.join(task["output_dir"], "network_edges.csv"), delimiter=",", data=False)
    print(f"  Finished: {task['tissue']}_{task['celltype']}")



def main():
    """Main function to orchestrate the entire analysis."""
    paths = setup_paths()
    tasks = find_analysis_tasks(paths)

    for task in tasks:
        process_analysis_task(task, paths)

    print("\n All analysis tasks completed!")


if __name__ == "__main__":
    main()