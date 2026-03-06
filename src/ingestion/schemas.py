from __future__ import annotations

import os
import sqlite3
from contextlib import closing

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Dynamically find the project root and point to the database folder
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
DB_DIR = os.path.join(root_dir, "database")
SQLITE_DB_PATH = os.path.join(DB_DIR, "risk_data.db")


def init_sqlite(db_path: str = SQLITE_DB_PATH) -> None:
    """Initialize SQLite database and create required tables if they do not exist."""
    # Ensure the databases folder exists before connecting
    os.makedirs(DB_DIR, exist_ok=True)
    
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS Clients (
                ClientID   INTEGER PRIMARY KEY,
                Name       TEXT,
                Turnover   REAL,
                RiskRating TEXT
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS Loans (
                LoanID       INTEGER PRIMARY KEY,
                ClientID     INTEGER,
                Amount       REAL,
                InterestRate REAL,
                Status       TEXT,
                FOREIGN KEY (ClientID) REFERENCES Clients(ClientID)
            );
            """
        )

        conn.commit()


def init_neo4j(uri: str = os.environ.get("NEO4J_URI")) -> None:
    """Initialize Neo4j constraints for Company and Person nodes."""
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")

    if not user or not password:
        raise RuntimeError(
            "NEO4J_USER and NEO4J_PASSWORD environment variables must be set."
        )

    driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_constraints(tx):
        # Neo4j 4+ syntax with IF NOT EXISTS.
        tx.run(
            """
            CREATE CONSTRAINT company_id_unique IF NOT EXISTS
            FOR (c:Company)
            REQUIRE c.id IS UNIQUE
            """
        )
        tx.run(
            """
            CREATE CONSTRAINT person_id_unique IF NOT EXISTS
            FOR (p:Person)
            REQUIRE p.id IS UNIQUE
            """
        )

    try:
        with driver.session() as session:
            session.execute_write(create_constraints)
    finally:
        driver.close()


def main() -> None:
    load_dotenv(override=True)

    init_sqlite()
    print(f"SQLite database initialized at '{SQLITE_DB_PATH}'.")

    init_neo4j()
    print("Neo4j constraints for Company.id and Person.id ensured.")


if __name__ == "__main__":
    main()

