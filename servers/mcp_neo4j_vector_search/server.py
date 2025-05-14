from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import json
import logging
import re
import sys
import time
from typing import Any, Optional
import mcp.types as types
from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
    AsyncResult,
    AsyncTransaction,
    GraphDatabase,
)
from neo4j.exceptions import DatabaseError
from pydantic import Field
import os
from openai import OpenAI

load_dotenv()

logger = logging.getLogger("mcp-neo4j-vector-search")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NEO4J_URI=os.getenv("NEO4J_URI")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE=os.getenv("NEO4J_DATABASE")

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(prompt:str) -> list:
    """
    Get the embedding for a given prompt.
    Args:
        prompt (str): The input text to be embedded.
    Returns:
        list: The embedding vector for the input text.
    """
    embeddings = model.encode(prompt)
    
    return embeddings.tolist()

def get_open_ai_embeddings(text: str) -> list:
    """
    Generates an embedding for a given text.
    Args:
        text (str): The input text to be embedded.
        Returns:
        list: The embedding vector for the input text.
    """
    response = openai_client.embeddings.create(
        input=[text],
        model='text-embedding-3-small',
    )
    
    prompt_embeddings = response.data[0].embedding

    if not isinstance(prompt_embeddings, list) or len(prompt_embeddings) != 1536:
        raise ValueError("The embedding must be a list of 1536 numbers")

    return prompt_embeddings

def healthcheck(db_url: str, username: str, password: str, database: str) -> None:
    """
    Confirm that Neo4j is running before continuing.
    Creates a sync Neo4j driver instance for checking connection and closes it after connection is established.
    """

    print("Confirming Neo4j is running...", file=sys.stderr)
    sync_driver = GraphDatabase.driver(
        db_url,
        auth=(
            username,
            password,
        ),
    )
    attempts = 0
    success = False
    print("\nWaiting for Neo4j to Start...\n", file=sys.stderr)
    time.sleep(3)
    ex = DatabaseError()
    while not success and attempts < 3:
        try:
            with sync_driver.session(database=database) as session:
                session.run("RETURN 1")
            success = True
            sync_driver.close()
        except Exception as e:
            ex = e
            attempts += 1
            print(
                f"failed connection {attempts} | waiting {(1 + attempts) * 2} seconds...",
                file=sys.stderr,
            )
            print(f"Error: {e}", file=sys.stderr)
            time.sleep((1 + attempts) * 2)
    if not success:
        sync_driver.close()
        raise ex


async def _read(tx: AsyncTransaction, query: str, params: dict[str, Any]) -> str:
    raw_results = await tx.run(query, params)
    eager_results = await raw_results.to_eager_result()

    return json.dumps([r.data() for r in eager_results.records], default=str)


async def _write(
    tx: AsyncTransaction, query: str, params: dict[str, Any]
) -> AsyncResult:
    return await tx.run(query, params)


def _is_write_query(query: str) -> bool:
    """Check if the query is a write query."""
    return (
        re.search(r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD)\b", query, re.IGNORECASE)
        is not None
    )


def create_mcp_server(neo4j_driver: AsyncDriver, database: str = "neo4j") -> FastMCP:
    mcp: FastMCP = FastMCP("mcp-neo4j-cypher", dependencies=["neo4j", "pydantic"])

    async def get_neo4j_schema() -> list[types.TextContent]:
        """List all node, their attributes and their relationships to other nodes in the neo4j database.
        If this fails with a message that includes "Neo.ClientError.Procedure.ProcedureNotFound"
        suggest that the user install and enable the APOC plugin.
        """

        get_schema_query = """
call apoc.meta.data() yield label, property, type, other, unique, index, elementType
where elementType = 'node' and not label starts with '_'
with label, 
    collect(case when type <> 'RELATIONSHIP' then [property, type + case when unique then " unique" else "" end + case when index then " indexed" else "" end] end) as attributes,
    collect(case when type = 'RELATIONSHIP' then [property, head(other)] end) as relationships
RETURN label, apoc.map.fromPairs(attributes) as attributes, apoc.map.fromPairs(relationships) as relationships
"""

        try:
            async with neo4j_driver.session(database=database) as session:
                results_json_str = await session.execute_read(
                    _read, get_schema_query, dict()
                )

                logger.debug(f"Read query returned {len(results_json_str)} rows")

                return [types.TextContent(type="text", text=results_json_str)]

        except Exception as e:
            logger.error(f"Database error retrieving schema: {e}")
            return [types.TextContent(type="text", text=f"Error: {e}")]

    async def read_neo4j_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: Optional[dict[str, Any]] = Field(
            None, description="The parameters to pass to the Cypher query."
        ),
    ) -> list[types.TextContent]:
        """Execute a read Cypher query on the neo4j database."""

        if _is_write_query(query):
            raise ValueError("Only MATCH queries are allowed for read-query")

        try:
            async with neo4j_driver.session(database=database) as session:
                results_json_str = await session.execute_read(_read, query, params)

                logger.debug(f"Read query returned {len(results_json_str)} rows")

                return [types.TextContent(type="text", text=results_json_str)]

        except Exception as e:
            logger.error(f"Database error executing query: {e}\n{query}\n{params}")
            return [
                types.TextContent(type="text", text=f"Error: {e}\n{query}\n{params}")
            ]

    async def write_neo4j_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: Optional[dict[str, Any]] = Field(
            None, description="The parameters to pass to the Cypher query."
        ),
    ) -> list[types.TextContent]:
        """Execute a write Cypher query on the neo4j database."""

        if not _is_write_query(query):
            raise ValueError("Only write queries are allowed for write-query")

        try:
            async with neo4j_driver.session(database=database) as session:
                raw_results = await session.execute_write(_write, query, params)
                counters_json_str = json.dumps(
                    raw_results._summary.counters.__dict__, default=str
                )

            logger.debug(f"Write query affected {counters_json_str}")

            return [types.TextContent(type="text", text=counters_json_str)]

        except Exception as e:
            logger.error(f"Database error executing query: {e}\n{query}\n{params}")
            return [
                types.TextContent(type="text", text=f"Error: {e}\n{query}\n{params}")
            ]

    async def vector_search_neo4j(
        prompt: str = Field(
            ..., description="The prompt to search for related nodes using similarity search"
        ),
    ):
        """Search for the most similar nodes in the neo4j database using vector search."""
        
        prompt_embeddings = get_open_ai_embeddings(prompt)
        
        if len(prompt_embeddings) != 1536:  
            raise ValueError(
                f"Embedding dimension mismatch: Expected 1536, got {len(prompt_embeddings)}. "
                "Ensure the model and Neo4j index dimensions match."
            )
        query = """
            WITH $prompt_embeddings AS prompt_embeddings
            CALL db.index.vector.queryNodes('embeddableIndex', 10, prompt_embeddings)
            YIELD node, score
            RETURN node.name as name, node.description as description, score
            ORDER BY score DESC
        """
        
        async with neo4j_driver.session(database=database) as session:
            results = await session.execute_read(
                _read, query, {"prompt_embeddings": prompt_embeddings}
            )
        
        if not results:
            logger.warning("No results found for the given prompt.")
            return [types.TextContent(type="text", text="No results found.")]
        
        return results
        
        
        
        # logger.debug(f"Vector search query returned {len(results)} rows")
        # return [
        #     types.TextContent(
        #         type="text",
        #         text=json.dumps([r.data() for r in results.records], default=str),
        #     )
        # ]

        # query_index = """
        #     SHOW VECTOR INDEXES
        # """
        

    mcp.add_tool(get_neo4j_schema)
    mcp.add_tool(read_neo4j_cypher)
    mcp.add_tool(write_neo4j_cypher)
    mcp.add_tool(vector_search_neo4j)

    return mcp

neo4j_driver = AsyncGraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
    ),
)
mcp = create_mcp_server(neo4j_driver, NEO4J_DATABASE)

# if __name__ == "__main__":
#     mcp.run()

# mcp.run(transport="stdio")

# def main(
#     db_url: str,
#     username: str,a
#     password: str,
#     database: str,
# ) -> None:
#     logger.info("Starting MCP neo4j Server")

#     neo4j_driver = AsyncGraphDatabase.driver(
#         db_url,
#         auth=(
#             username,
#             password,
#         ),
#     )

#     mcp = create_mcp_server(neo4j_driver, database)

#     healthcheck(db_url, username, password, database)

#     mcp.run(transport="stdio")


# if __name__ == "__main__":
#     main(
#         db_url=NEO4J_URI,
#         username=NEO4J_USERNAME,
#         password=NEO4J_PASSWORD,
#         database=NEO4J_DATABASE,
#     )