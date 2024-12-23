import networkx as nx

# Initialize the knowledge graph
G = nx.Graph()

def build_knowledge_graph():
    # Add entities and relationships to the graph
    G.add_node("High Court", entity_type="court", location="Bengaluru")
    G.add_node("Petitioner", entity_type="person", name="Sri Joy Panakkal")
    G.add_node("Respondent", entity_type="organization", name="CBI")
    G.add_node("Impugned Order", entity_type="document", date="13.10.2023")
    G.add_edges_from([("Petitioner", "Impugned Order"), ("Respondent", "Impugned Order"), ("High Court", "Impugned Order")])

def graph_retrieve_entities(query):
    # Retrieve entities based on the query
    if "High Court" in query:
        return nx.get_node_attributes(G, "location")["High Court"]
    elif "Petitioner" in query:
        return nx.get_node_attributes(G, "name")["Petitioner"]
    elif "Impugned Order" in query:
        return nx.get_node_attributes(G, "date")["Impugned Order"]
    else:
        return "No relevant structured information found"
