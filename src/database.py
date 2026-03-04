"""
Neo4j database connection and operations module.
Handles connecting to Neo4j, fetching schema, and executing Cypher queries.
"""

import logging
import warnings

from neo4j import GraphDatabase
from src.config import Config

# Suppress Neo4j driver deprecation notifications about propertyTypes
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


class Neo4jDatabase:
    """Manages the Neo4j database connection and query execution."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD),
        )
        self._schema: str | None = None

    def close(self):
        """Close the database driver connection."""
        self.driver.close()

    def verify_connectivity(self) -> bool:
        """Test whether the Neo4j connection is alive."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def run_query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """
        Execute a Cypher query and return the results as a list of dicts.
        """
        parameters = parameters or {}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*propertyTypes.*will change output format.*",
            )
            with self.driver.session() as session:
                result = session.run(cypher, parameters)
                return [record.data() for record in result]

    def get_schema(self) -> str:
        """
        Retrieve the database schema (node labels, relationship types,
        and property keys) and cache it for reuse.
        """
        if self._schema is not None:
            return self._schema

        schema_parts: list[str] = []

        # Node labels and their properties
        node_props = self.run_query(
            "CALL db.schema.nodeTypeProperties() "
            "YIELD nodeType, propertyName, propertyTypes "
            "RETURN nodeType, propertyName, propertyTypes"
        )
        if node_props:
            schema_parts.append("Node Properties:")
            for row in node_props:
                prop_types = row.get("propertyTypes") or []
                types = ", ".join(str(t) for t in prop_types) if prop_types else "Unknown"
                prop_name = row.get("propertyName", "?")
                # Strip Neo4j backtick quoting so the LLM sees clean labels
                node_type = row['nodeType'].replace('`', '')
                schema_parts.append(
                    f"  {node_type}.{prop_name} : {types}"
                )

        # Relationship types and their properties
        rel_props = self.run_query(
            "CALL db.schema.relTypeProperties() "
            "YIELD relType, propertyName, propertyTypes "
            "RETURN relType, propertyName, propertyTypes"
        )
        if rel_props:
            schema_parts.append("\nRelationship Properties:")
            for row in rel_props:
                prop_types = row.get("propertyTypes") or []
                types = ", ".join(str(t) for t in prop_types) if prop_types else "None"
                prop = row.get("propertyName") or ""
                # Strip Neo4j backtick quoting
                rel_type = row['relType'].replace('`', '')
                if prop:  # Only show rels that have properties
                    schema_parts.append(f"  {rel_type}.{prop} : {types}")
                else:
                    schema_parts.append(f"  {rel_type} (no properties)")

        # Relationship patterns — use a simpler approach that avoids
        # schema.visualization() which returns non-serializable objects.
        try:
            pattern_query = self.run_query(
                "MATCH (a)-[r]->(b) "
                "RETURN DISTINCT labels(a)[0] AS start_label, type(r) AS rel_type, "
                "labels(b)[0] AS end_label LIMIT 50"
            )
            if pattern_query:
                schema_parts.append("\nRelationship Patterns:")
                seen = set()
                for row in pattern_query:
                    pattern = f"(:{row['start_label']})-[:{row['rel_type']}]->(:{row['end_label']})"
                    if pattern not in seen:
                        seen.add(pattern)
                        schema_parts.append(f"  {pattern}")
        except Exception:
            pass  # Non-critical, skip if it fails

        # Fallback: grab labels and rel types directly
        if not node_props:
            labels = self.run_query("CALL db.labels() YIELD label RETURN label")
            schema_parts.append("Node Labels:")
            for row in labels:
                schema_parts.append(f"  :{row['label']}")

            # Get sample properties for each label
            for row in labels:
                label = row["label"]
                sample = self.run_query(
                    f"MATCH (n:`{label}`) RETURN keys(n) AS props LIMIT 1"
                )
                if sample and sample[0].get("props"):
                    props = ", ".join(sample[0]["props"])
                    schema_parts.append(f"  :{label} properties: {props}")

        if not rel_props:
            rels = self.run_query(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            )
            if rels:
                schema_parts.append("\nRelationship Types:")
                for row in rels:
                    schema_parts.append(f"  :{row['relationshipType']}")

                # Get sample patterns
                for row in rels:
                    rtype = row["relationshipType"]
                    sample = self.run_query(
                        f"MATCH (a)-[r:`{rtype}`]->(b) "
                        f"RETURN labels(a)[0] AS start, labels(b)[0] AS end LIMIT 1"
                    )
                    if sample:
                        s = sample[0]
                        schema_parts.append(
                            f"  (:{s['start']})-[:{rtype}]->(:{s['end']})"
                        )

        self._schema = "\n".join(schema_parts) if schema_parts else "Schema unavailable"
        return self._schema

    def check_vector_index_exists(self, index_name: str = "stackoverflow_embeddings") -> bool:
        """Check if a vector index exists in the database."""
        try:
            result = self.run_query(
                "SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name"
            )
            return any(r["name"] == index_name for r in result)
        except Exception:
            return False
